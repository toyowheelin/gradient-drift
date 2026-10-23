"""
Decrypts an Assetto Corsa car's data.acd archive into its individual plain-text
INI files (tyres.ini, suspensions.ini, drivetrain.ini, engine.ini, car.ini, ...),
exactly analogous to tools/track_triangles.py extracting real track geometry from a
.kn5 - same idea, different real data source, for the same reason: grounding
the new physics model in this car's ACTUAL numbers instead of invented ones.

The archive format and its decryption (a key derived from the car's folder name, each byte
shifted by a cycling character of the key) follow acd_extractor by Bovis
(https://github.com/bovis/acd_extractor, MIT licence; notice below). Before it touches your car
file, the self-test below decrypts acd_extractor's own test archive and checks the result byte
for byte; if that ever fails, this stops rather than writing wrong numbers.

  One known variant, not yet relevant here: some later/modded cars use
  uppercase-preserving folder names in the key derivation rather than forcing
  lowercase (some acd tools make this distinction explicitly).
  gravygarage_street_miata is already all-lowercase, so that variant is a
  non-issue for this specific car - flagged here only so it isn't forgotten if
  this script is ever pointed at a car with a mixed-case folder name and the
  self-test still passes but the real output looks like garbage.

Usage:
    python acd_to_ini.py <path to data.acd> <car folder name> [output dir]

  <car folder name> is the EXACT name of the folder data.acd normally lives in
  on the server (e.g. "gravygarage_street_miata", from content/cars/<this>/
  data.acd) - it's what the decryption key is derived from, per the real
  format. It's passed explicitly rather than inferred from the uploaded file's
  own path, because uploaded files in this environment land under a hashed,
  renamed path that has nothing to do with the car's real folder name (as seen
  all session with every other upload) - inferring it from the wrong path
  would silently derive the wrong key and produce garbage, not an error.

  [output dir] defaults to "<car folder name>_data" in the current directory.
"""
import os
import struct
import sys


def get_key(dirname: str) -> str:
    """Ports acd_extractor's Cipher.get_key (Ruby) line-for-line. dirname is
    the car's own folder name (e.g. 'gravygarage_street_miata')."""
    n = len(dirname)
    o = [ord(c) for c in dirname]

    # PART 1
    part1 = sum(o[0:n]) & 0xff

    # PART 2
    part2 = 0
    i = 0
    length = n - 1
    while i < length:
        part2 = part2 * o[i]
        i += 1
        part2 = part2 - o[i]
        i += 1
    part2 &= 0xff

    # PART 3 - explicit truncating division, mirroring the reference source's
    # manual negative-case branch exactly. Both Ruby's '/' and Python's '//'
    # FLOOR for negative operands (unlike C-style truncation), so this branch
    # must stay as its own case, not get simplified into a plain '//'.
    part3 = 0
    i = 1
    length = n - 3
    while i < length:
        part3 = part3 * o[i]
        i += 1
        divisor = o[i] + 0x1b
        if part3 < 0:
            part3 = -(abs(part3) // divisor)
        else:
            part3 = part3 // divisor
        i -= 2
        part3 = part3 + (-(0x1b) - o[i])
        i += 4
    part3 &= 0xff

    # PART 4
    part4 = 0x1683
    i = 1
    while i < n:
        part4 = part4 - o[i]
        i += 1
    part4 &= 0xff

    # PART 5
    part5 = 0x42
    i = 1
    length = n - 4
    while i < length:
        a = part5 * (o[i] + 0xf)
        i -= 1
        b = o[i]
        i += 1
        b = b + 0xf
        b = b * a
        b = b + 0x16
        part5 = b
        i += 4
    part5 &= 0xff

    # PART 6
    part6 = 0x65
    i = 0
    length = n - 2
    while i < length:
        part6 = part6 - o[i]
        i += 2
    part6 &= 0xff

    # PART 7
    part7 = 0xab
    i = 0
    length = n - 2
    while i < length:
        part7 = part7 % o[i]
        i += 2
    part7 &= 0xff

    # PART 8
    part8 = 0xab
    i = 0
    length = n - 1
    while i < length:
        part8 = part8 // o[i]
        i += 1
        part8 = part8 + o[i]
    part8 &= 0xff

    return f"{part1}-{part2}-{part3}-{part4}-{part5}-{part6}-{part7}-{part8}"


def decrypt_acd(acd_path: str, key: str):
    """Yields (filename, decrypted_bytes) for each entry in a data.acd archive."""
    keylen = len(key)
    key_ords = [ord(c) for c in key]

    with open(acd_path, "rb") as f:
        data = f.read()

    pos = 0
    while pos < len(data):
        namesize = struct.unpack_from("<i", data, pos)[0]  # signed read first
        pos += 4
        if namesize == -1111:
            # Newer archives (some Kunos DLC cars) open with -1111 and one more
            # 4-byte value before the first entry; skip both.
            pos += 4
            continue

        filename = data[pos:pos + namesize].decode("ascii")
        pos += namesize

        contentsize = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        raw = data[pos:pos + contentsize * 4]
        pos += contentsize * 4

        decrypted = bytearray()
        j = 0  # runs continuously across the whole entry's content, not reset
               # per 4-byte group
        for k in range(contentsize):
            # Each content byte is stored as a 4-byte little-endian integer:
            # the value is in the first byte and the other three are zero.
            # The value itself CAN be zero (a plain byte equal to 256 minus
            # the key character) - that happens in binary entries, e.g. the
            # stray Windows shortcut inside gravygarage_street_e36_touring.
            group = raw[k * 4:(k + 1) * 4]
            if len(group) != 4 or group[1] or group[2] or group[3]:
                raise ValueError(
                    f"{filename}: byte-group {k} is {list(group)} (expected "
                    f"one value byte followed by three zeros) - the format "
                    f"assumption this script relies on doesn't hold here."
                )
            raw_byte = group[0]
            shift = key_ords[j % keylen]
            decrypted.append((raw_byte - shift) % 256)
            j += 1

        yield filename, bytes(decrypted)


# The decryption above follows acd_extractor (https://github.com/bovis/acd_extractor), by Bovis,
# and the test archive below is from its test suite; both under its MIT licence (as stated in its
# gemspec):
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify, merge,
#   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#   to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#   FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
# The test archive: acd_extractor's test/testcar/ data.acd and the aero.ini it decrypts to.
_FIXTURE_ACD = (
    b'\x08\x00\x00\x00aero.ini$\x00\x00\x00\x8d\x00\x00\x00|\x00\x00\x00'
    b'{\x00\x00\x00n\x00\x00\x00z\x00\x00\x00r\x00\x00\x00\x83\x00\x00\x00'
    b'\x8e\x00\x00\x00=\x00\x00\x00\x83\x00\x00\x00v\x00\x00\x00\x7f\x00\x00\x00'
    b'\x85\x00\x00\x00y\x00\x00\x00\x85\x00\x00\x00{\x00\x00\x00o\x00\x00\x00'
    b'h\x00\x00\x00[\x00\x00\x00g\x00\x00\x00U\x00\x00\x00M\x00\x00\x00'
    b'l\x00\x00\x00\xa7\x00\x00\x00\x9a\x00\x00\x00\xa4\x00\x00\x00\xa7\x00\x00\x00'
    b'\x9f\x00\x00\x00\x9c\x00\x00\x00\xa4\x00\x00\x00M\x00\x00\x00\x9a\x00\x00\x00'
    b'\x9f\x00\x00\x00\x99\x00\x00\x00\x9c\x00\x00\x00;\x00\x00\x00'
)
_FIXTURE_PARENT_DIR = "testcar"
_FIXTURE_EXPECTED = {"aero.ini": b"[HEADER]\nVERSION=1.2  ;version info\n"}


def run_self_test():
    import tempfile
    key = get_key(_FIXTURE_PARENT_DIR)
    with tempfile.NamedTemporaryFile(suffix=".acd", delete=False) as tmp:
        tmp.write(_FIXTURE_ACD)
        tmp_path = tmp.name
    try:
        results = dict(decrypt_acd(tmp_path, key))
    finally:
        os.unlink(tmp_path)

    if results != _FIXTURE_EXPECTED:
        print("SELF-TEST FAILED - this script does NOT currently reproduce a "
              "known-good data.acd correctly. Refusing to run against real "
              "data until this is fixed.", file=sys.stderr)
        print(f"  expected: {_FIXTURE_EXPECTED}", file=sys.stderr)
        print(f"  got:      {results}", file=sys.stderr)
        sys.exit(1)
    print("Self-test passed (reproduced acd_extractor's real test fixture "
          "byte-for-byte). Proceeding.\n")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    run_self_test()

    acd_path = sys.argv[1]
    car_folder_name = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else f"{car_folder_name}_data"

    key = get_key(car_folder_name)
    print(f"Derived key for '{car_folder_name}': {key}")

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename, content in decrypt_acd(acd_path, key):
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "wb") as f:
            f.write(content)
        print(f"  wrote {out_path} ({len(content)} bytes)")
        count += 1

    print(f"\nExtracted {count} files from {acd_path} into {output_dir}/")


if __name__ == "__main__":
    main()