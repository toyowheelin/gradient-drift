"""
line_tools.py - reads, checks and rebuilds a track's AI line (fast_lane.ai) for the drift bot.

The bot steers by the AI line and slides within a few metres either side of it, so it needs a line
down the middle of the road that closes cleanly. A track's own line is a recorded lap: often a
racing line, and sometimes with a knot where the recording closes (it runs back on itself for a few
metres), uneven spacing or a long gap. setup_track.py uses this; it can also be run on its own:

    python line_tools.py <track>           # report on tracks/<track>/<track>_fast_lane.ai

check() needs only the line. rebuild() also needs the track's geometry (a fast_sim World) to find
the paving's edges, the walls and the ground:

1. the recording's knots: where it turns back on itself (more than 60 deg in one point) the pieces
   running the other way are dropped, and points turning more than 30 deg are dropped with their
   bunched-up neighbours;
2. resampled every 1.5 m (heights, speed, gas and brake carried along by distance);
3. smoothed over about 22 m, and over about 45 m within 40 m of a gap the cleaning left;
4. moved onto the middle of the paving where the paving is less than 24 m wide (wider is a pad
   with a course marked on it, which the line already follows), at most 3 m;
5. moved into the middle of the room before the first wall or paving edge, at most 2 m, eased in
   and out over about 100 m (walls that stand on the paving);
6. resampled every 1.5 m again and set on the road with a ray from 1 m above the carried height,
   so a bridge's two decks never mix.

Written in the game's own format (version 7) with radius, distances to the road edges, and the
recording's speed, gas and brake.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SPACING_M = 1.5
SMOOTH_K = 7                  # points either side: about 22 m
GAP_SMOOTH_K = 15             # about 45 m, near a gap left by cleaning
PAD_WIDTH_M = 24.0            # paving wider than this is a pad: the line is not centred on it
MAX_PAVING_SHIFT_M = 3.0
MAX_WALL_SHIFT_M = 2.0


# ---- reading and writing -----------------------------------------------------------------------

def read_line(path):
    """(positions N x 3, speed/gas/brake N x 3 or None) of a fast_lane.ai, version 7 or -1."""
    with open(path, 'rb') as f:
        data = f.read()
    version, n = struct.unpack_from('<ii', data, 0)
    if version == 7:
        pos = np.array([struct.unpack_from('<fff', data, 16 + 20 * i) for i in range(n)])
        off = 16 + 20 * n
        m = struct.unpack_from('<i', data, off)[0]
        sgb = None
        if m == n:
            sgb = np.array([struct.unpack_from('<3f', data, off + 4 + 72 * i) for i in range(n)])
        return pos, sgb
    if version == -1:
        return np.array([struct.unpack_from('<fff', data, 8 + 20 * i) for i in range(n)]), None
    raise ValueError(f'{path}: fast_lane.ai version {version} (7 or -1 expected)')


def radius_3pt(xz):
    a, b, c = np.roll(xz, 1, 0), xz, np.roll(xz, -1, 0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cr = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    r = np.minimum(ab * bc * ca / np.maximum(2.0 * cr, 1e-9), 1e4)
    return smooth(r, 2)


def write_line(path, pos, sgb, left, right):
    """Version 7: per point position, length and id; then speed, gas, brake, radius, the distances
    to the road's left and right edges, no camber, normal up, length and forward."""
    n = len(pos)
    nxt = np.roll(pos, -1, 0)
    length = np.linalg.norm(nxt - pos, axis=1)
    fwd = (nxt - pos) / np.maximum(length[:, None], 1e-9)
    radius = radius_3pt(pos[:, [0, 2]])
    sgb = np.zeros((n, 3)) if sgb is None else sgb
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiii', 7, n, 0, n))
        for i in range(n):
            f.write(struct.pack('<ffffi', *pos[i], length[i], i))
        f.write(struct.pack('<i', n))
        for i in range(n):
            f.write(struct.pack('<18f', *sgb[i], 0.0, radius[i], left[i], right[i], 0.0, 1.0,
                                0.0, 1.0, 0.0, length[i], *fwd[i], 0.0, 0.0))


# ---- small helpers -------------------------------------------------------------------------------

def smooth(a, k):
    """Box average over 2k+1 points round a closed loop (a: N or N x m)."""
    a = np.asarray(a, dtype=float)
    pad = np.concatenate([a[-k:], a, a[:k]])
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    if a.ndim == 1:
        return np.convolve(pad, kern, 'valid')
    return np.stack([np.convolve(pad[:, j], kern, 'valid') for j in range(a.shape[1])], 1)


def resample(xz, attrs, spacing=SPACING_M):
    """Even spacing round the closed loop; attrs (N x m) carried by distance. Also returns, for each
    new point, the distance from it to the nearest old point (to find gaps)."""
    loop = np.r_[xz, xz[:1]]
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    n = max(int(round(arc[-1] / spacing)), 8)
    s = np.linspace(0.0, arc[-1], n, endpoint=False)
    out = np.c_[np.interp(s, arc, loop[:, 0]), np.interp(s, arc, loop[:, 1])]
    al = np.r_[attrs, attrs[:1]]
    out_attrs = np.stack([np.interp(s, arc, al[:, j]) for j in range(al.shape[1])], 1)
    k = np.clip(np.searchsorted(arc, s, side='right') - 1, 0, len(seg) - 1)
    gap = seg[k]
    return out, out_attrs, gap


def normals(xz):
    f = np.roll(xz, -1, 0) - np.roll(xz, 1, 0)
    f /= np.maximum(np.linalg.norm(f, axis=1)[:, None], 1e-9)
    return f, np.c_[-f[:, 1], f[:, 0]]


def turns_deg(xz):
    d = np.roll(xz, -1, 0) - xz
    h = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees((h - np.roll(h, 1) + np.pi) % (2 * np.pi) - np.pi)


# ---- checking ------------------------------------------------------------------------------------

def check(pos):
    """What is wrong with a line, as a dict; 'problems' lists the reasons to rebuild it."""
    xz = pos[:, [0, 2]]
    seg = np.linalg.norm(np.roll(xz, -1, 0) - xz, axis=1)
    med = float(np.median(seg))
    turn = turns_deg(xz)
    rep = dict(points=len(pos), length_m=float(seg.sum()), spacing_median=med, spacing_max=float(seg.max()),
               closing_gap_m=float(seg[-1]), turn_max_deg=float(np.abs(turn).max()),
               reversals=int((np.abs(turn) > 60).sum()), sharp=int((np.abs(turn) > 30).sum()))
    problems = []
    if rep['reversals']:
        problems.append(f"turns back on itself at {rep['reversals']} point(s)")
    elif rep['sharp']:
        problems.append(f"{rep['sharp']} point(s) turn more than 30 deg")
    if seg.max() > 4.0 * med:
        problems.append(f"a {seg.max():.1f} m gap between points (median {med:.2f} m)")
    if seg[-1] > 5.0:
        problems.append(f"the lap's two ends are {seg[-1]:.1f} m apart")
    rep['problems'] = problems
    return rep


def paving_and_walls(world, xz, y, nn, reach=20.0, step=0.25):
    """Distance from each point to the paving's edge and to the first wall, either side
    (columns: +normal, -normal)."""
    import fast_sim as fs
    R = world.ray_arrays()
    n = len(xz)
    edge = np.full((n, 2), reach)
    wall = np.full((n, 2), reach)
    for i in range(n):
        for c, sg in enumerate((1.0, -1.0)):
            for kk in range(1, int(reach / step)):
                x, z = xz[i] + sg * nn[i] * kk * step
                gd, gs = fs.ground_below(x, y[i] + 1.0, z, 4.0, *R)
                if gd < 0 or gs != fs.SURF_ROAD:
                    edge[i, c] = kk * step
                    break
            d, _ = fs.raycast(xz[i, 0], y[i] + 0.5, xz[i, 1], sg * nn[i, 0], 0.0, sg * nn[i, 1], reach, *R, 1)
            if d > 0:
                wall[i, c] = d
    return edge, wall


def centring(world, pos):
    """Median distance of the line from the middle of the paving (narrow paving only) and the
    narrowest room before a wall or the paving's edge."""
    xz = pos[:, [0, 2]]
    _, nn = normals(xz)
    edge, wall = paving_and_walls(world, xz, pos[:, 1], nn)
    narrow = edge.sum(1) < PAD_WIDTH_M
    off = np.abs(edge[:, 0] - edge[:, 1]) / 2
    room = np.minimum(edge, wall).min(1)
    return dict(off_centre_median_m=float(np.median(off[narrow])) if narrow.any() else 0.0,
                narrow_share=float(narrow.mean()), room_min_m=float(room.min()),
                room_p10_m=float(np.percentile(room, 10)))


# ---- rebuilding ----------------------------------------------------------------------------------

def clean(pos, sgb):
    """Drops the recording's knots: pieces running the wrong way between two reversals, and
    sharply turning points with their bunched-up neighbours. Returns the kept rows."""
    xz = pos[:, [0, 2]]
    n = len(xz)
    seg = np.linalg.norm(np.roll(xz, -1, 0) - xz, axis=1)
    med = np.median(seg)
    turn = np.abs(turns_deg(xz))
    keep = np.ones(n, bool)
    rev = []
    for i in np.nonzero(turn > 60)[0]:          # one reversal can spread over a few points
        if rev and i - rev[-1][-1] <= 3:
            rev[-1].append(i)
        else:
            rev.append([i])
    if len(rev) > 1 and (rev[0][0] + n) - rev[-1][-1] <= 3:
        rev[0] = rev.pop() + rev[0]
    rev = [g[len(g) // 2] for g in rev]
    if len(rev) >= 2:
        # pieces between reversals alternate direction; keep those going the longest one's way
        pieces = [(rev[i], rev[(i + 1) % len(rev)]) for i in range(len(rev))]
        lengths = [((b - a) % n) for a, b in pieces]
        main = int(np.argmax(lengths))
        for k, (a, b) in enumerate(pieces):
            if (k - main) % 2:
                idx = (a + np.arange((b - a) % n + 1)) % n
                keep[idx] = False
    for i in np.nonzero(turn > 30)[0]:
        keep[i] = False
        for step in (1, -1):
            j = i
            while True:
                j = (j + step) % n
                s = seg[j] if step > 0 else seg[(j - 1) % n]
                if s >= 0.75 * med or j == i:
                    break
                keep[j] = False
    return pos[keep], (None if sgb is None else sgb[keep])


def rebuild(world, pos, sgb, centre=True, log=print):
    """The cleaned, centred, evenly spaced line: (positions, speed/gas/brake, left, right)."""
    import fast_sim as fs
    R = world.ray_arrays()
    pos, sgb = clean(pos, sgb)
    attrs = np.c_[pos[:, 1], sgb if sgb is not None else np.zeros((len(pos), 3))]
    xz, attrs, gap = resample(pos[:, [0, 2]], attrs)
    xz = smooth(xz, SMOOTH_K)
    # harder near any gap the cleaning left, eased in over 15 m
    near = np.zeros(len(xz))
    for i in np.nonzero(gap > 3.0)[0]:
        d = np.abs((np.arange(len(xz)) - i + len(xz) // 2) % len(xz) - len(xz) // 2) * SPACING_M
        near = np.maximum(near, np.clip(1.0 - (d - 25.0) / 15.0, 0.0, 1.0))
    if near.any():
        xz = (1 - near[:, None]) * xz + near[:, None] * smooth(xz, GAP_SMOOTH_K)
    xz, attrs, _ = resample(xz, attrs)

    def ground(xz, y0):
        out = np.empty(len(xz))
        for i, ((x, z), yy) in enumerate(zip(xz, y0)):
            d, _ = fs.ground_below(x, yy + 1.0, z, 4.0, *R)
            out[i] = yy + 1.0 - d if d >= 0 else yy
        return out

    if centre:
        y = ground(xz, attrs[:, 0])
        _, nn = normals(xz)
        edge, _ = paving_and_walls(world, xz, y, nn)
        o = np.where(edge.sum(1) < PAD_WIDTH_M, (edge[:, 0] - edge[:, 1]) / 2, 0.0)
        o = smooth(np.clip(o, -MAX_PAVING_SHIFT_M, MAX_PAVING_SHIFT_M), SMOOTH_K)
        xz = xz + nn * o[:, None]
        log(f"  moved onto the middle of the paving: median {np.median(np.abs(o)):.2f} m, at most {np.abs(o).max():.2f} m")
        xz, attrs, _ = resample(xz, attrs)
        for _ in range(2):
            y = ground(xz, attrs[:, 0])
            _, nn = normals(xz)
            edge, wall = paving_and_walls(world, xz, y, nn)
            room = np.minimum(edge, wall)
            o = np.clip((room[:, 0] - room[:, 1]) / 2, -MAX_WALL_SHIFT_M, MAX_WALL_SHIFT_M)
            for _ in range(3):
                o = smooth(o, 15)
            xz = xz + nn * o[:, None]
        xz, attrs, _ = resample(xz, attrs)
    y = ground(xz, attrs[:, 0])
    _, nn = normals(xz)
    edge, _ = paving_and_walls(world, xz, y, nn)
    pos = np.c_[xz[:, 0], y, xz[:, 1]]
    return pos, (attrs[:, 1:] if sgb is not None else None), edge[:, 1], edge[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track')
    tf.tracks_arg(ap)
    args = ap.parse_args()
    pos, _ = read_line(tf.path(args.tracks, 'fast_lane', args.track))
    rep = check(pos)
    for k, v in rep.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
