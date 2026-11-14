"""
zone_line.py - move a course's AI line into its drift zones, so the bots drive through them.

The policies follow the AI line closely (it is what their line inputs measure from), so where the
line runs decides where the bots run. This moves each AI-line point in a drift zone to INSIDE_M
inside that zone's edge (the car's centre; with the car at a drift angle its rear axle ends up
nearer the edge), eases it in and out over RAMP points either side and smooths the result, so
the line bends gently from the racing line into each zone and back. Nothing else changes: the
points keep their ids, so the zones and the course edges still fit them; the edges' side widths
are worked out again for the moved line, and the pace reference made for the old line is set
aside (training and setup make a new one from the moved line's shape).

Files changed in <tracks>/<track>/: <track>_fast_lane.ai and <track>_course_edges.csv (the
originals are kept beside them as .before_zone_line, and a later run always starts from those, so
running it again with another depth replaces the move rather than adding to it), and
<track>_pace_reference.csv moved aside. --undo puts the originals back.

Run setup afterwards to put the moved line, edges and pace reference in the server folder.

    python zone_line.py --tracks C:/dev/drift_rl/tracks --track driftplayground_2021
    python zone_line.py --tracks tracks --track spiritpeaksraceway --inside 3
    python zone_line.py --tracks tracks --track spiritpeaksraceway --undo
"""
import argparse
import os
import shutil
import struct
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import track_files as tf  # noqa: E402

BACKUP = '.before_zone_line'
INSIDE_M = 3.0
RAMP_POINTS = 5
SMOOTH_POINTS = 5         # each side: an 11-point moving average
MAX_SLOPE = 0.2           # the line moves sideways at most 0.2 m per metre along it (a short zone far
                          # from the racing line gets less of the move rather than a kink)
NEW_BEND_M = 80.0         # the moved line may not add a bend tighter than this that the racing line does
                          # not have (turning the other way, or on a straight): eased out where it does
MIN_SIDE_M = 3.0          # and keeps at least this much road between itself and either edge


def _layout(raw):
    """(points, offset of the first position) of a fast_lane.ai: version 7 (the game's) or -1 (the
    short form generated courses use); either way 20 bytes a point, the position first."""
    version, n = struct.unpack_from('<ii', raw, 0)
    if version == 7:
        return n, 16
    if version == -1:
        return n, 8
    raise ValueError(f'fast_lane.ai version {version}: only versions 7 and -1 can be moved')


def read_line(raw):
    """Positions (n, 3) of a fast_lane.ai."""
    n, start = _layout(raw)
    return np.array([struct.unpack_from('<fff', raw, start + 20 * i) for i in range(n)], dtype=np.float64)


def write_line(raw, xyz):
    """The line with its points at xyz, and each point's corner radius worked out again for them (the
    simulator and the plugin read it: straights, the inside of a corner, the tightest corner ahead)."""
    from line_tools import radius_3pt
    n, start = _layout(raw)
    radius = radius_3pt(np.asarray(xyz)[:, [0, 2]])
    out = bytearray(raw)
    for i, p in enumerate(xyz):
        struct.pack_into('<fff', out, start + 20 * i, *p)
        if start == 8:                                   # version -1: position, radius, camber
            struct.pack_into('<f', out, start + 20 * i + 12, radius[i])
        else:                                            # version 7: radius is the 5th of 18 extras
            struct.pack_into('<f', out, 16 + 20 * n + 4 + 72 * i + 16, radius[i])
    return bytes(out)


def zone_points(zones, n):
    """(side, point ids in order) for each zone, wrapping past the line's start."""
    out = []
    for _, z in zones.iterrows():
        a, b = int(z.FirstPointId), int(z.LastPointId)
        ids = np.arange(a, b + 1) if b >= a else np.r_[np.arange(a, n), np.arange(0, b + 1)]
        out.append((int(np.sign(z.Side)), ids))
    return out


def curvature(xz, h=3):
    """Signed curvature (1/m) of a closed line, from points h either side."""
    d1 = (np.roll(xz, -h, 0) - np.roll(xz, h, 0)) / 2
    d2 = np.roll(xz, -h, 0) - 2 * xz + np.roll(xz, h, 0)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-9)


def new_bends(xz, moved, new_bend_m=NEW_BEND_M):
    """Points where the moved line bends tighter than new_bend_m and the racing line there does not
    bend that way (it turns the other way, or runs nearly straight)."""
    k0, k1 = curvature(xz), curvature(moved)
    return (np.abs(k1) > 1 / new_bend_m) & ((np.sign(k0) != np.sign(k1)) | (np.abs(k0) < 1 / 300))


def move_line(xz, edges, zones, inside_m=INSIDE_M, ramp=RAMP_POINTS, smooth=SMOOTH_POINTS, max_slope=MAX_SLOPE,
              new_bend_m=NEW_BEND_M, min_side_m=MIN_SIDE_M):
    """The AI line's (x, z) moved into the zones: (n, 2)."""
    n = len(xz)
    min_side_m = min(min_side_m, inside_m)          # --inside nearer than MIN_SIDE_M still means it
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy(dtype=np.float64)
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy(dtype=np.float64)
    off = np.zeros((n, 2))
    weight = np.zeros(n)
    spacing = float(np.median(np.linalg.norm(np.roll(xz, -1, 0) - xz, axis=1)))
    for side, ids in zone_points(zones, n):
        edge = pos if side > 0 else neg
        m = len(ids)
        r = ramp
        if max_slope > 0:
            # a longer ramp for a bigger move, so the line eases over at no more than max_slope
            inward = (pos[ids] + neg[ids]) / 2 - edge[ids]
            inward /= np.linalg.norm(inward, axis=1)[:, None] + 1e-9
            peak = np.linalg.norm(edge[ids] + inward * inside_m - xz[ids], axis=1).max()
            r = max(ramp, int(np.ceil(peak / (max_slope * spacing))))
        for k in range(-r, m + r):
            i = (ids[0] + k) % n
            w = 1.0 if 0 <= k < m else 1.0 - ((-k) if k < 0 else k - m + 1) / (r + 1)
            if w <= weight[i]:
                continue
            inward = (pos[i] + neg[i]) / 2 - edge[i]
            inward /= np.linalg.norm(inward) + 1e-9
            off[i] = (edge[i] + inward * inside_m - xz[i]) * w
            weight[i] = w
    if max_slope > 0:
        # how far each point may move: no more than max_slope per metre from its neighbours' allowance,
        # both ways round the lap (a closed loop: three passes settle it)
        step = np.linalg.norm(np.roll(xz, -1, 0) - xz, axis=1)
        want = np.linalg.norm(off, axis=1)
        allow = want.copy()
        for _ in range(3):
            for i in range(n):
                allow[i] = min(allow[i], allow[i - 1] + max_slope * step[i - 1])
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n
                allow[i] = min(allow[i], allow[j] + max_slope * step[i])
        off *= (allow / np.maximum(want, 1e-9))[:, None]
    if smooth > 0:
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        off = np.stack([np.convolve(np.r_[off[-smooth:, j], off[:, j], off[:smooth, j]], kernel, 'valid')
                        for j in range(2)], 1)
    def keep_off_edges(off):
        # never nearer an edge than min_side_m (a curved edge and the smoothing can bring it closer),
        # unless the racing line itself runs nearer there: then the move there is cut back toward the
        # racing line until it is far enough (or the racing line's own distance)
        out_ok = np.ones(n, bool)
        scale = np.ones(n)
        for t in np.linspace(1, 0, 21):
            out = xz + off * scale[:, None]
            out_ok[:] = True
            for e, other in ((pos, neg), (neg, pos)):
                width = np.linalg.norm(other - e, axis=1)
                want = np.minimum(np.minimum(min_side_m, width / 2), np.linalg.norm(xz - e, axis=1))
                out_ok &= np.linalg.norm(out - e, axis=1) >= want - 1e-6
            if out_ok.all():
                break
            scale[~out_ok] = t
        return off * scale[:, None]

    def ease_new_bends(off, rounds):
        # bends the racing line does not have (on Spirit Peaks, where the zones are far from it, the
        # line swung across the road between zones and wobbled through long corners, and the bots
        # fell out of their drifts there): smooth the move more, only around them
        for _ in range(rounds):
            bad = new_bends(xz, xz + off, new_bend_m)
            if not bad.any():
                return off, True
            near = np.convolve(np.r_[bad[-8:], bad, bad[:8]].astype(float), np.ones(17), 'valid') > 0
            avg = (np.roll(off, 1, 0) + off + np.roll(off, -1, 0)) / 3
            off = off.copy()
            off[near] = avg[near]
        return off, False

    for _ in range(30):
        if min_side_m > 0:
            off = keep_off_edges(off)
        if new_bend_m <= 0:
            break
        off, clean = ease_new_bends(off, 20)
        if clean and (min_side_m <= 0 or np.array_equal(keep_off_edges(off), off)):
            break
    if new_bend_m > 0:
        off, _ = ease_new_bends(off, 500)       # no new bends wins over the last few cm from an edge
    return xz + off


def side_widths(xz, edges):
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy(dtype=np.float64)
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy(dtype=np.float64)
    out = edges.copy()
    out['PositiveSideM'] = np.linalg.norm(pos - xz, axis=1).round(3)
    out['NegativeSideM'] = np.linalg.norm(neg - xz, axis=1).round(3)
    return out


def write_file(path, data):
    """Write a real file in place of a symbolic link (never through it: track folders are often links
    into another copy's files)."""
    if os.path.islink(path):
        os.remove(path)
    with open(path, 'wb') as f:
        f.write(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--track', required=True)
    ap.add_argument('--inside', type=float, default=INSIDE_M, help='metres inside the zone edge for the line (car centre)')
    ap.add_argument('--ramp', type=int, default=RAMP_POINTS, help='points either side of a zone to ease in and out over')
    ap.add_argument('--undo', action='store_true', help='put the original line and edges back')
    ap.add_argument('--fresh', action='store_true', help='the line and edges there now are the originals (just '
                    'rebuilt): keep them as the originals, setting aside any older ones')
    args = ap.parse_args(argv)

    folder = tf.folder(args.tracks, args.track)
    if os.path.islink(folder):
        sys.exit(f'{folder} is a link to {os.path.realpath(folder)}: copy the folder itself (cp -rL) so the '
                 'line is not moved there too')
    line = tf.path(args.tracks, 'fast_lane', args.track)
    edges_path = tf.path(args.tracks, 'course_edges', args.track)
    zones_path = tf.path(args.tracks, 'course_zones', args.track)
    pace = tf.pace_reference_path(args.tracks, args.track)
    for p in (line, edges_path) + (() if args.undo else (zones_path,)):
        if not os.path.exists(p) and not os.path.exists(p + BACKUP):
            sys.exit(f'{p} not found')
    if args.undo:
        if not os.path.exists(line + BACKUP):
            sys.exit(f'{args.track}: the line was not moved (no {os.path.basename(line) + BACKUP})')
        for p in (line, edges_path):
            if os.path.islink(p):
                os.remove(p)
            os.replace(p + BACKUP, p)
        if os.path.exists(pace):
            os.replace(pace, pace + '.zone_line')
        print(f'{args.track}: original AI line and course edges back; the pace reference is made again from it')
        return
    for p in (line, edges_path):
        if args.fresh and os.path.exists(p + BACKUP):
            os.replace(p + BACKUP, p + BACKUP + '.old')
        if not os.path.exists(p + BACKUP):
            shutil.copy2(p, p + BACKUP)
    raw = open(line + BACKUP, 'rb').read()
    xyz = read_line(raw)
    edges = pd.read_csv(edges_path + BACKUP)
    zones = pd.read_csv(zones_path)
    if len(edges) != len(xyz):
        sys.exit(f'{edges_path} has {len(edges)} points, the line {len(xyz)}: build the course for this line first')
    moved = move_line(xyz[:, [0, 2]], edges, zones, args.inside, args.ramp)
    shift = np.linalg.norm(moved - xyz[:, [0, 2]], axis=1)
    new = xyz.copy()
    new[:, 0], new[:, 2] = moved[:, 0], moved[:, 1]
    write_file(line, write_line(raw, new))
    widths = side_widths(moved, edges)
    write_file(edges_path, widths.to_csv(index=False).encode())
    if os.path.exists(pace):
        os.replace(pace, pace + BACKUP)
    moved_pts = shift > 0.05
    print(f'{args.track}: {len(zones)} zones, {moved_pts.sum()} of {len(xyz)} AI-line points moved '
          f'(median {np.median(shift[moved_pts]) if moved_pts.any() else 0:.1f} m, most {shift.max():.1f} m); '
          f'narrowest side now {min(widths.PositiveSideM.min(), widths.NegativeSideM.min()):.1f} m. '
          'Run setup to put the moved line in the server.')


if __name__ == '__main__':
    main()
