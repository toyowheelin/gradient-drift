"""
centre_line.py - writes an AI line down the middle of a track, from the track's own edge files
(data/side_l.csv and data/side_r.csv in the layout folder), as a fast_lane.ai the game and
AssettoServer read.

    py centre_line.py <layout data folder> <output fast_lane.ai> [--like <existing fast_lane.ai>]

Why: a track's own AI line is often a racing line, apex to apex, that runs within a metre of the
edge in places. The drift bot steers by its line and slides within a few metres either side of it,
so it needs the middle of the road.

The edge files hold X, Y, Z and progress round the lap (0 - 1). Their Z is often negated against the
model; --like (the track's own AI line) settles which way, the direction of travel and the line's
height above the edges. Without it, Z is taken as negated (as on every track checked so far).
The line is resampled every 1.5 m (as the game records them) and smoothed over about 6 m.
Written in the game's own format (version 7) with the radius, the half widths and the length the
game and the server use; speeds and pedals are left at zero.
"""
import argparse
import os
import struct

import numpy as np

SPACING_M = 1.5
SMOOTH_M = 6.0


def read_side(path):
    rows = np.loadtxt(path, delimiter=',')
    rows = rows[np.argsort(rows[:, 3])]
    return rows


def at(side, s):
    return np.stack([np.interp(s, side[:, 3], side[:, k], period=1.0) for k in range(3)], 1)


def read_fast_lane(path):
    """Positions of a fast_lane.ai (version 7 or -1)."""
    with open(path, 'rb') as f:
        version, n = struct.unpack('<ii', f.read(8))
        if version == 7:
            f.read(8)                                       # lap time, sample count
            return np.array([struct.unpack('<fff8x', f.read(20)) for _ in range(n)])   # position, length, id
        return np.array([struct.unpack('<fffff', f.read(20))[:3] for _ in range(n)])


def smooth_closed(p, k):
    pad = np.r_[p[-k:], p, p[:k]]
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, j], kern, mode='valid') for j in range(p.shape[1])], 1)


def radius_3pt(xz):
    a, b, c = np.roll(xz, 1, 0), xz, np.roll(xz, -1, 0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cr = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    r = np.minimum(ab * bc * ca / np.maximum(2.0 * cr, 1e-9), 1e4)
    return smooth_closed(r[:, None], 2)[:, 0]


def write_v7(path, pos, half_left, half_right):
    n = len(pos)
    nxt = np.roll(pos, -1, 0)
    length = np.linalg.norm(nxt - pos, axis=1)
    fwd = (nxt - pos) / np.maximum(length[:, None], 1e-9)
    radius = radius_3pt(pos[:, [0, 2]])
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiii', 7, n, 0, 0))
        for i in range(n):
            f.write(struct.pack('<ffffi', *pos[i], length[i], i))
        f.write(struct.pack('<i', n))
        for i in range(n):
            f.write(struct.pack('<4f', 0.0, 0.0, 0.0, 0.0))                    # speed, gas, brake, lat g
            f.write(struct.pack('<3f', radius[i], half_left[i], half_right[i]))
            f.write(struct.pack('<2f', 0.0, 1.0))                              # camber, its direction
            f.write(struct.pack('<3f', 0.0, 1.0, 0.0))                         # normal
            f.write(struct.pack('<f', length[i]))
            f.write(struct.pack('<3f', *fwd[i]))
            f.write(struct.pack('<2f', 0.0, 0.0))                              # tag, grade


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help="the layout's data folder (side_l.csv, side_r.csv)")
    ap.add_argument('output')
    ap.add_argument('--like', default=None, help="the track's own fast_lane.ai, to match")
    args = ap.parse_args()
    left = read_side(os.path.join(args.data, 'side_l.csv'))
    right = read_side(os.path.join(args.data, 'side_r.csv'))
    lap = np.linalg.norm(np.diff(left[:, [0, 2]], axis=0), axis=1).sum()
    s = np.linspace(0.0, 1.0, int(lap / 0.5), endpoint=False)
    l, r = at(left, s), at(right, s)
    zsign, rise, reverse = -1.0, 0.0, False
    if args.like:
        ref = read_fast_lane(args.like)
        from scipy.spatial import cKDTree
        tree = cKDTree(ref[:, [0, 2]])
        best = None
        for sign in (1.0, -1.0):
            mid = (l + r) / 2 * np.array([1, 1, sign])
            d, i = tree.query(mid[:, [0, 2]])
            if best is None or np.median(d) < best[0]:
                best = (np.median(d), sign, i, mid)
        _, zsign, i, mid = best
        rise = float(np.median(ref[i, 1] - mid[:, 1]))
        step = np.diff(i[::50])
        reverse = np.median((step + len(ref) // 2) % len(ref) - len(ref) // 2) < 0
        print(f"edges' Z {'negated' if zsign < 0 else 'as is'}; the track's own line is {rise:.2f} m above "
              f"the edges' middle and {'runs the other way' if reverse else 'runs the same way'}; middle of the road "
              f"{best[0]:.1f} m from it (median)")
    l[:, 2] *= zsign
    r[:, 2] *= zsign
    mid = (l + r) / 2 + np.array([0.0, rise, 0.0])
    half = np.linalg.norm((l - r)[:, [0, 2]], axis=1) / 2
    if reverse:
        mid, half = mid[::-1], half[::-1]
        l, r = r[::-1], l[::-1]
    # resample every SPACING_M along the middle, then smooth
    seg = np.linalg.norm(np.diff(np.r_[mid, mid[:1]][:, [0, 2]], axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(0.0, arc[-1], SPACING_M)
    pos = np.stack([np.interp(t, arc, np.r_[mid[:, k], mid[0, k]]) for k in range(3)], 1)
    hw = np.interp(t, arc, np.r_[half, half[0]])
    pos = smooth_closed(pos, max(1, int(SMOOTH_M / SPACING_M / 2)))
    write_v7(args.output, pos, hw, hw)
    print(f"wrote {args.output}: {len(pos)} points, {arc[-1]:.0f} m, road {2 * hw.min():.1f} - {2 * hw.max():.1f} m wide "
          f"(median {2 * np.median(hw):.1f}), tightest radius {radius_3pt(pos[:, [0, 2]]).min():.1f} m")


if __name__ == '__main__':
    main()
