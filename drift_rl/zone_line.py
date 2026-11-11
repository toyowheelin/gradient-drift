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
INSIDE_M = 2.5
RAMP_POINTS = 5
SMOOTH_POINTS = 5         # each side: an 11-point moving average


def read_line(raw):
    """Positions (n, 3) of a version 7 fast_lane.ai."""
    version, n = struct.unpack_from('<ii', raw, 0)
    if version != 7:
        raise ValueError(f'fast_lane.ai version {version}: only version 7 lines can be moved')
    return np.array([struct.unpack_from('<fff', raw, 16 + 20 * i) for i in range(n)], dtype=np.float64)


def write_line(raw, xyz):
    out = bytearray(raw)
    for i, p in enumerate(xyz):
        struct.pack_into('<fff', out, 16 + 20 * i, *p)
    return bytes(out)


def zone_points(zones, n):
    """(side, point ids in order) for each zone, wrapping past the line's start."""
    out = []
    for _, z in zones.iterrows():
        a, b = int(z.FirstPointId), int(z.LastPointId)
        ids = np.arange(a, b + 1) if b >= a else np.r_[np.arange(a, n), np.arange(0, b + 1)]
        out.append((int(np.sign(z.Side)), ids))
    return out


def move_line(xz, edges, zones, inside_m=INSIDE_M, ramp=RAMP_POINTS, smooth=SMOOTH_POINTS):
    """The AI line's (x, z) moved into the zones: (n, 2)."""
    n = len(xz)
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy(dtype=np.float64)
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy(dtype=np.float64)
    off = np.zeros((n, 2))
    weight = np.zeros(n)
    for side, ids in zone_points(zones, n):
        edge = pos if side > 0 else neg
        m = len(ids)
        for k in range(-ramp, m + ramp):
            i = (ids[0] + k) % n
            w = 1.0 if 0 <= k < m else 1.0 - ((-k) if k < 0 else k - m + 1) / (ramp + 1)
            if w <= weight[i]:
                continue
            inward = (pos[i] + neg[i]) / 2 - edge[i]
            inward /= np.linalg.norm(inward) + 1e-9
            off[i] = (edge[i] + inward * inside_m - xz[i]) * w
            weight[i] = w
    if smooth > 0:
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        off = np.stack([np.convolve(np.r_[off[-smooth:, j], off[:, j], off[:smooth, j]], kernel, 'valid')
                        for j in range(2)], 1)
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
