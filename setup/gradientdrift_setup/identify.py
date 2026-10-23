"""Which server program a server folder runs: AssettoServer (the plugin needs it) or the game's own
dedicated server, and AssettoServer's version.

AssettoServer is a .NET program published as one file, and its name is written inside it many
times, with the version beside it ("AssettoServer/0.0.54-g51737e2c2e"). The game's own acServer
has neither. File names prove nothing: AssettoServer is often renamed acServer.exe so existing
start scripts keep working, with the game's original kept beside it.
"""
import glob
import mmap
import os
import re
from dataclasses import dataclass

# the AssettoServer version the plugin is built against and tested on
TESTED_VERSION = '0.0.54'
EXECUTABLES = ('AssettoServer.exe', 'AssettoServer', 'acServer.exe', 'acServer')
_MARK = b'AssettoServer'
_VERSION_IN_PROGRAM = re.compile(rb'"AssettoServer/(\d+\.\d+\.\d+)')
_VERSION_IN_LOG = re.compile(r'AssettoServer (\d+\.\d+\.\d+)')


@dataclass
class ServerProgram:
    kind: str            # 'assettoserver', 'kunos' (the game's own server) or 'unknown'
    program: str = ''    # the file that told
    version: str = ''    # AssettoServer's, '' if not found
    note: str = ''


def _is_assettoserver(path):
    """(True/False, version) by looking inside the file."""
    try:
        with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m.find(_MARK) < 0:
                return False, ''
            found = _VERSION_IN_PROGRAM.search(m)
            return True, found.group(1).decode() if found else ''
    except (OSError, ValueError):          # unreadable, or empty (mmap of 0 bytes)
        return False, ''


def _version_from_logs(folder):
    logs = sorted(glob.glob(os.path.join(folder, 'logs', 'log-*.txt')), key=os.path.getmtime, reverse=True)
    for path in logs[:3]:
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                for line in f:
                    found = _VERSION_IN_LOG.search(line)
                    if found:
                        return found.group(1)
        except OSError:
            continue
    return ''


def identify(folder):
    present = [os.path.join(folder, name) for name in EXECUTABLES if os.path.isfile(os.path.join(folder, name))]
    for path in present:
        yes, version = _is_assettoserver(path)
        if yes:
            return ServerProgram('assettoserver', path, version or _version_from_logs(folder))
    if present:
        return ServerProgram('kunos', present[0],
                             note="this is the game's own dedicated server; the drift bot needs AssettoServer "
                                  "(https://assettoserver.org)")
    version = _version_from_logs(folder)
    if version:
        return ServerProgram('assettoserver', '', version, note='no server program in the folder; its log says AssettoServer')
    return ServerProgram('unknown', note='no server program found in the folder (AssettoServer.exe or acServer.exe)')


def version_tuple(v):
    return tuple(int(x) for x in v.split('.')) if v else ()
