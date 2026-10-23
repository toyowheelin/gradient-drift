"""
build_zones.py - places Formula Drift-style OUTSIDE ZONES on the course and writes them as a CSV.

Why this exists: everything the trainer paid for was per metre along the racing
line, so the policy drove the racing line - apex on the inside third of every corner, never on
the outside third. A drift course is judged the other way round: the rear of the car is to be
kept in zones marked along the OUTSIDE edge (Formula Drift: "place the rear of their vehicle in
between the cones for the length of the zone while in drift"; 40 of a run's 100 points are how
completely the zones are filled, and anything past a zone's outer boundary is "No-Go").

Where the zones go is worked out from the course's own shape (the course edges build_course.py
wrote), nothing track-specific:
  1. The course centreline (midway between the two edges) is cut into corners: stretches that
     turn one way with a radius under CORNER_RADIUS_M.
  2. Each corner gets one zone on its outside edge, by how far the corner turns:
       150 deg or more   outside zone over 30% - 90% of the corner (a long sweeper: on it early)
       90 - 150 deg      outside zone over 45% - 95% (mid-corner to exit, where the car is carried)
       45 - 90 deg       touch and go over the last 30% (a short zone: the rear only has to reach it)
       under 45 deg      nothing (a kink)
  3. A zone is ZONE_DEPTH_M deep, measured in from the course edge.

Output (in drift_rl/tracks/<track>/): <track>_course_zones.csv, one row per
zone, in spline point ids. fast_sim.load_world() reads it; without it there are no zones.

    python build_zones.py                 # writes the file, prints the zones
    python build_zones.py --plot          # also writes course_zones.png
    python build_zones.py --track <name>    # another track (see build_course.py)
    python build_zones.py --score A.csv   # how well recorded laps fill the zones (plugin telemetry or
                                          # the in-game recorder's CSV; repeatable)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))

CORNER_RADIUS_M = 80.0          # the centreline turning tighter than this is a corner
CURVATURE_WINDOW_M = 12.0       # curvature is measured across this much centreline
MIN_TURN_DEG = 45.0             # a corner turning less than this gets no zone
# (corner turns at least this many degrees, zone kind, zone from / to as fractions of the corner)
ZONE_RULES = [(150.0, 'outside zone', 0.30, 0.90),
              (90.0, 'outside zone', 0.45, 0.95),
              (MIN_TURN_DEG, 'touch and go', 0.70, 1.00)]
ZONE_DEPTH_M = 3.0              # also drift_env.RewardConfig.zone_depth_m
REAR_AXLE_BEHIND_CENTRE_M = 1.3


def find_corners(mid):
    """Corners of a closed centreline (n, 2): list of (first point, last point, +1 left / -1 right,
    degrees turned, tightest radius m)."""
    n = len(mid)
    seg = np.linalg.norm(np.roll(mid, -1, 0) - mid, axis=1)
    k = max(2, int(round(0.5 * CURVATURE_WINDOW_M / seg.mean())))
    pad = np.r_[mid[-2 * k:], mid, mid[:2 * k]]
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    smooth = np.stack([np.convolve(pad[:, j], kernel, mode='valid') for j in range(2)], 1)[k:k + n]
    step = np.roll(smooth, -1, 0) - np.roll(smooth, 1, 0)
    heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
    turn = np.roll(heading, -k) - np.roll(heading, k)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    span = np.array([seg[np.arange(i - k, i + k) % n].sum() for i in range(n)])
    curvature = turn / span
    side = np.where(np.abs(curvature) > 1.0 / CORNER_RADIUS_M, np.sign(curvature), 0.0)
    start = int(np.argmax(side == 0))                      # begin the walk on a straight
    order = (np.arange(n) + start) % n
    corners, j = [], 0
    while j < n:
        if side[order[j]] == 0:
            j += 1
            continue
        e = j
        while e + 1 < n and side[order[e + 1]] == side[order[j]]:
            e += 1
        pts = order[j:e + 1]
        corners.append((int(pts[0]), int(pts[-1]), int(side[pts[0]]),
                        float(np.degrees(abs(np.sum(curvature[pts] * seg[pts])))),
                        float(1.0 / np.abs(curvature[pts]).max())))
        j = e + 1
    return sorted(corners), seg


def place_zones(edges):
    """edges: the course-edges table. Returns a list of zone dicts."""
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy()
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy()
    n = len(pos)
    corners, seg = find_corners(0.5 * (pos + neg))
    zones = []
    for number, (first, last, direction, turned, radius) in enumerate(corners, 1):
        rule = next((r for r in ZONE_RULES if turned >= r[0]), None)
        if rule is None:
            continue
        pts = np.arange(first, first + ((last - first) % n) + 1) % n
        along = np.cumsum(seg[pts]) / seg[pts].sum()
        inside = pts[(along >= rule[2]) & (along <= rule[3] + 1e-9)]
        edge = neg if direction > 0 else pos               # a left turn's outside is the right-hand edge
        zones.append({'corner': number, 'kind': rule[1], 'side': -direction, 'first': int(inside[0]),
                      'last': int(inside[-1]), 'turned': turned, 'radius': radius,
                      'length': float(np.linalg.norm(np.diff(edge[inside], axis=0), axis=1).sum()),
                      'points': inside})
    return zones


HELD_MIN_TURN_DEG = 90.0        # corners at least this long are drifted in one direction...
HELD_FROM, HELD_TO = 0.15, 0.90 # ...from this far into the corner to this far (entry and exit flips are free)
HELD_JOIN_GAP_M = 10.0          # two corners the same way with less than this between them count as one here


def held_side_per_point(pos, neg):
    """Per centreline point: the outside edge (+1 positive, -1 negative, as zone sides) where the
    tail has to stay out, 0 where a change of direction is fine. pos, neg: (n, 2) edge points."""
    n = len(pos)
    corners, seg = find_corners(0.5 * (np.asarray(pos) + np.asarray(neg)))
    corners = join_corners(corners, seg)
    side = np.zeros(n)
    for first, last, direction, turned, _ in corners:
        if turned < HELD_MIN_TURN_DEG:
            continue
        pts = np.arange(first, first + ((last - first) % n) + 1) % n
        along = np.cumsum(seg[pts]) / seg[pts].sum()
        side[pts[(along >= HELD_FROM) & (along <= HELD_TO)]] = -direction
    return side


def join_corners(corners, seg):
    """Corners in order round the lap with same-direction neighbours less than HELD_JOIN_GAP_M apart
    merged (a long corner whose radius opens out for a few metres in the middle is still one
    corner to hold the tail out through). Degrees add; the radius is the tighter one."""
    n = len(seg)
    out = []
    for c in corners:
        if out:
            first, last, direction, turned, radius = out[-1]
            gap = np.arange(last + 1, c[0]) % n
            if c[2] == direction and seg[gap].sum() < HELD_JOIN_GAP_M:
                out[-1] = (first, c[1], direction, turned + c[3], min(radius, c[4]))
                continue
        out.append(c)
    if len(out) > 1:
        first, last, direction, turned, radius = out[-1]
        c = out[0]
        gap = np.arange(last + 1, c[0] + n) % n
        if c[2] == direction and seg[gap].sum() < HELD_JOIN_GAP_M:
            out[0] = (first, c[1], direction, turned + c[3], min(radius, c[4]))
            out.pop()
    return out


def zone_side_per_point(zones, n):
    side = np.zeros(n)
    for z in zones:
        side[z['points']] = z['side']
    return side


def rear_depth(x, z, heading, edges, side):
    """For car positions and headings: nearest spline point, and how far inside that point's zone
    edge the rear axle is (m; negative = past the edge). NaN where the point has no zone."""
    from scipy.spatial import cKDTree
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy()
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy()
    line = np.stack([pos[:, 0] - (pos[:, 0] - neg[:, 0]) * edges.PositiveSideM / (edges.PositiveSideM + edges.NegativeSideM),
                     pos[:, 1] - (pos[:, 1] - neg[:, 1]) * edges.PositiveSideM / (edges.PositiveSideM + edges.NegativeSideM)], 1)
    rx = x - REAR_AXLE_BEHIND_CENTRE_M * np.cos(heading)
    rz = z - REAR_AXLE_BEHIND_CENTRE_M * np.sin(heading)
    _, point = cKDTree(line).query(np.c_[x, z])
    edge = np.where((side[point] > 0)[:, None], pos[point], neg[point])
    inward = line[point] - edge
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    depth = (rx - edge[:, 0]) * inward[:, 0] + (rz - edge[:, 1]) * inward[:, 1]
    return point, np.where(side[point] != 0, depth, np.nan)


def load_laps(path):
    """x, z, heading (radians, x toward z) and slip angle (deg) of a recorded run; plugin telemetry
    or the in-game recorder's CSV (whose heading is worked out from the path and the rear tires)."""
    head = pd.read_csv(path, nrows=1).columns
    if 'RealHeadingDeg' in head:
        d = pd.read_csv(path, usecols=['PosX', 'PosZ', 'RealHeadingDeg', 'RealSlipAngleDeg', 'SpeedKmh'])
        d = d[d.SpeedKmh > 25]
        return d.PosX.to_numpy(), d.PosZ.to_numpy(), np.radians(d.RealHeadingDeg.to_numpy()), d.RealSlipAngleDeg.abs().to_numpy()
    d = pd.read_csv(path, usecols=['X', 'Z', 'SpeedKMH', 'SlipAngle_RL', 'SlipAngle_RR']).iloc[::8]
    x, z = d.X.to_numpy(), d.Z.to_numpy()
    path_angle = np.arctan2(np.gradient(z), np.gradient(x))
    rear_slip = np.radians(0.5 * (d.SlipAngle_RL + d.SlipAngle_RR).to_numpy())
    # the rear tires' slip angle is the car's slip angle less a yaw-rate term: close enough to
    # point the body with. Sign: tried both ways below, the one that puts the rear outside wins.
    keep = (d.SpeedKMH > 25).to_numpy()
    return x[keep], z[keep], (path_angle[keep], rear_slip[keep]), np.abs(np.degrees(rear_slip[keep]))


def score(path, edges, zones):
    side = zone_side_per_point(zones, len(edges))
    x, z, heading, slip = load_laps(path)
    if isinstance(heading, tuple):
        tries = [rear_depth(x, z, heading[0] + s * heading[1], edges, side) for s in (1.0, -1.0)]
        point, depth = max(tries, key=lambda t: -np.nanmean(t[1]))
    else:
        point, depth = rear_depth(x, z, heading, edges, side)
    rows = []
    for zone in zones:
        here = np.isin(point, zone['points'])
        if here.sum() < 5:
            rows.append((np.nan, np.nan, np.nan))
            continue
        d = depth[here]
        filled = (d >= 0.0) & (d <= ZONE_DEPTH_M) & (slip[here] >= 20.0)
        rows.append((float(filled.mean()), float(np.median(d)), float((d < 0.0).mean())))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--score', action='append', default=[], metavar='CSV')
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    args = ap.parse_args(argv)
    data = args.tracks
    track = args.track or tf.TRACK
    edges = pd.read_csv(tf.path(data, 'course_edges', track))
    zones = place_zones(edges)
    out = tf.writable_path(data, 'course_zones', track)
    with open(out, 'w') as f:
        f.write('Zone,Kind,Side,FirstPointId,LastPointId,LengthM,Corner,CornerTurnsDeg,CornerRadiusM\n')
        for i, z in enumerate(zones, 1):
            f.write(f"{i},{z['kind']},{z['side']},{z['first']},{z['last']},{z['length']:.1f},{z['corner']},"
                    f"{z['turned']:.0f},{z['radius']:.0f}\n")
    lap = float(np.linalg.norm(np.diff(edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy(), axis=0), axis=1).sum())
    print(f"{len(zones)} zones, {sum(z['length'] for z in zones):.0f} m of a {lap:.0f} m outside edge; {ZONE_DEPTH_M:g} m deep")
    print("zone  kind           corner            edge                 points     length")
    for i, z in enumerate(zones, 1):
        print(f"{i:3d}   {z['kind']:13s}  {'left' if z['side'] < 0 else 'right'} {z['turned']:3.0f} deg, R {z['radius']:2.0f} m   "
              f"{'right (negative)' if z['side'] < 0 else 'left (positive) '}   {z['first']:3d} - {z['last']:3d}   {z['length']:4.0f} m")
    print(f"written: {out}")
    scores = {os.path.splitext(os.path.basename(p))[0].replace('_', ' '): score(p, edges, zones) for p in args.score}
    for name, rows in scores.items():
        print(f"\n{name}: share of each zone's length with the rear axle in it, sliding; median distance of the rear "
              "axle inside the edge; share past the edge")
        for i, (fill, depth, past) in enumerate(rows, 1):
            print(f"  zone {i:2d}  filled {fill:5.0%}   rear {depth:5.1f} m inside the edge   past it {past:4.0%}")
        print(f"  line score, Formula Drift style (mean fill x 40): {40 * np.nanmean([r[0] for r in rows]):.1f} / 40")
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy()
        neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy()
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.fill(np.r_[pos[:, 0], pos[0, 0], neg[0, 0], neg[::-1, 0]], np.r_[pos[:, 1], pos[0, 1], neg[0, 1], neg[::-1, 1]],
                color='#ecebe6', lw=0)
        for e in (pos, neg):
            ax.plot(np.r_[e[:, 0], e[0, 0]], np.r_[e[:, 1], e[0, 1]], color='#8a8983', lw=0.8)
        for i, z in enumerate(zones, 1):
            outer = (neg if z['side'] < 0 else pos)[z['points']]
            other = (pos if z['side'] < 0 else neg)[z['points']]
            inward = (other - outer) / np.linalg.norm(other - outer, axis=1)[:, None]
            inner = outer + ZONE_DEPTH_M * inward
            ax.fill(np.r_[outer[:, 0], inner[::-1, 0]], np.r_[outer[:, 1], inner[::-1, 1]], color='#eb6834',
                    alpha=0.85 if z['kind'] == 'outside zone' else 0.45, lw=0)
            label = outer[len(outer) // 2] - 5.0 * inward[len(outer) // 2]
            ax.text(label[0], label[1], f"{'OZ' if z['kind'] == 'outside zone' else 'TG'}{i}", ha='center', va='center',
                    fontsize=10, color='#3d3d3a')
        for path, colour, name in zip(args.score, ('#2a78d6', '#1baf7a', '#4a3aa7'), scores):
            x, z_, _, _ = load_laps(path)
            jump = np.r_[False, np.hypot(np.diff(x), np.diff(z_)) > 5.0]
            ax.plot(np.where(jump, np.nan, x), np.where(jump, np.nan, z_), color=colour, lw=1.0, alpha=0.8, label=name)
        mid = 0.5 * (pos + neg)
        for i in range(0, len(mid) - 10, 36):               # direction of travel, about every 60 m
            ax.annotate('', xy=mid[i + 9], xytext=mid[i], zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color='#3d3d3a', lw=2.0, mutation_scale=22))
        ax.set_aspect('equal')
        if args.score:
            for handle in ax.legend(frameon=False, loc='upper left').get_lines():
                handle.set_linewidth(2.5)
        ax.set_title('Outside zones (solid) and touch-and-go areas (light), placed from the course shape; arrows = direction of travel',
                     loc='left')
        png = os.path.join(_HERE, 'course_zones.png' if track == tf.TRACK else f'course_zones_{track}.png')
        fig.savefig(png, dpi=110, bbox_inches='tight')
        print(f"written: {png}")


if __name__ == '__main__':
    main()
