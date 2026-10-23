"""
judge.py - drivers on the same course side by side, and how they would score against Formula
Drift's judging, lap by lap: the bot from the plugin's telemetry, a human from the in-game
recorder.

    python judge.py drift_physics_telemetry_<date>_<time>.csv  <recording>.csv  [--track NAME] [--map map.png]

Each file can hold several drivers (the plugin's telemetry has every bot); each driver's full laps
are judged separately and the driver gets the median lap. The course's outside zones are
build_zones.py's (they need the track's course edges in tracks/<track>/).

The score follows Formula Drift's 2026 solo-run weights - line 40, angle 40, style 20 - measured
rather than judged by eye, so read it as a rough guide (README, "What a drift run is judged on"):

* line: per zone, the share of its time the rear axle is within 3 m inside the zone's edge while
  sliding (20 deg or more), cut by the share spent past the edge (No-Go); the zones averaged.
* angle: the average rear slip angle through the zones, against 60 deg as the course's maximum.
* style: the average of four 0-1 marks - steering that does not saw back and forth (turning back
  by more than 10% of lock, against 3 a second), a steady angle through each zone (spread
  against 20 deg), a steady pace while sliding (spread against 25%), and commitment (the share of
  zone time sliding).

The angle is the rear tires' slip angle for both (the recorder has no other), so a bot and a
human are measured the same way; the human recording's car need not be the bot's.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import track_files as tf

SLIDING_DEG = 20.0
ANGLE_MAX_DEG = 60.0
REVERSAL_SHARE = 0.10
RECORDER_STEP = 8           # the recorder's 167 samples a second, thinned to about the server's tick


def drivers_in(path):
    """[(name, DataFrame with t, x, z, heading (rad) or None, slip (deg, signed), speed (km/h),
    steer (share of lock), pedal (0-1))] for a file."""
    head = pd.read_csv(path, nrows=1).columns
    base = os.path.splitext(os.path.basename(path))[0]
    out = []
    if 'RealHeadingDeg' in head:
        d = pd.read_csv(path, on_bad_lines='skip', low_memory=False,
                        usecols=['Tick', 'SessionId', 'SpawnCounter', 'RealDtMs', 'PosX', 'PosZ', 'RealHeadingDeg',
                                 'RearSlipAngleDeg', 'SpeedKmh', 'SteerCommandDeg', 'PolicySpeedAction', 'PolicyActive'])
        d = d[d.PolicyActive.astype(str) == 'True']
        lock = max(float(d.SteerCommandDeg.abs().quantile(0.995)), 1.0)
        for (sid, spawn), g in d.sort_values('Tick').groupby(['SessionId', 'SpawnCounter']):
            g = g[(g.RealDtMs > 20) & (g.RealDtMs < 120)]
            if len(g) < 200:
                continue
            out.append((f'{base} bot {sid}', pd.DataFrame({
                't': (g.RealDtMs / 1000.0).cumsum().to_numpy(), 'x': g.PosX.to_numpy(), 'z': g.PosZ.to_numpy(),
                'heading': np.radians(g.RealHeadingDeg.to_numpy()), 'slip': g.RearSlipAngleDeg.to_numpy(),
                'speed': g.SpeedKmh.to_numpy(), 'steer': g.SteerCommandDeg.to_numpy() / lock,
                'pedal': g.PolicySpeedAction.clip(lower=0.0).to_numpy()})))
    else:
        d = pd.read_csv(path, usecols=['X', 'Z', 'SpeedKMH', 'Steer', 'Throttle', 'SlipAngle_RL', 'SlipAngle_RR']
                        ).iloc[::RECORDER_STEP].reset_index(drop=True)
        lock = max(float(d.Steer.abs().max()), 1.0)
        out.append((f'{base} human', pd.DataFrame({
            't': np.arange(len(d)) * RECORDER_STEP * 0.006, 'x': d.X.to_numpy(), 'z': d.Z.to_numpy(), 'heading': np.nan,
            'slip': 0.5 * (d.SlipAngle_RL + d.SlipAngle_RR).to_numpy(), 'speed': d.SpeedKMH.to_numpy(),
            'steer': d.Steer.to_numpy() / lock, 'pedal': d.Throttle.to_numpy()})))
    return out


def place(df, edges, side):
    """Adds each sample's nearest course point and the rear axle's depth inside the zone edge."""
    import build_zones as bz
    if df.heading.isna().all():
        # the recorder has no heading: the path's direction turned by the rear slip, whichever
        # sign puts the rear on the outside (as build_zones.score does)
        path = np.arctan2(np.gradient(df.z.to_numpy()), np.gradient(df.x.to_numpy()))
        headings = [path + s * np.radians(df.slip.to_numpy()) for s in (1.0, -1.0)]
        tries = [bz.rear_depth(df.x.to_numpy(), df.z.to_numpy(), h, edges, side) for h in headings]
        best = int(np.argmax([-np.nanmean(t[1]) for t in tries]))
        (point, depth), heading = tries[best], headings[best]
    else:
        heading = df.heading.to_numpy()
        point, depth = bz.rear_depth(df.x.to_numpy(), df.z.to_numpy(), heading, edges, side)
    df = df.copy()
    df['point'], df['depth'] = point, depth
    df['rear_x'] = df.x - bz.REAR_AXLE_BEHIND_CENTRE_M * np.cos(heading)
    df['rear_z'] = df.z - bz.REAR_AXLE_BEHIND_CENTRE_M * np.sin(heading)
    return df


def laps(df, n):
    """Full laps: (start, end) row ranges between two crossings of the start of the course."""
    p = df.point.to_numpy()
    cross = np.nonzero((p[:-1] > 0.85 * n) & (p[1:] < 0.15 * n))[0] + 1
    return [(a, b) for a, b in zip(cross[:-1], cross[1:]) if (df.t.iloc[b] - df.t.iloc[a]) > 20.0]


def judge_lap(lap, zones):
    in_zone = np.zeros(len(lap), bool)
    credits, spreads, per_zone = [], [], []
    for z in zones:
        here = np.isin(lap.point.to_numpy(), z['points'])
        in_zone |= here
        if here.sum() < 3:
            per_zone.append((np.nan,) * 5)
            continue
        d, s = lap.depth.to_numpy()[here], np.abs(lap.slip.to_numpy()[here])
        filled = float(((d >= 0.0) & (d <= 3.0) & (s >= SLIDING_DEG)).mean())
        past = float((d < 0.0).mean())
        credits.append(filled * (1.0 - past))
        spreads.append(float(np.std(s)))
        per_zone.append((float(lap.speed.to_numpy()[here].mean()), float(s.mean()), filled, past, float(np.median(d))))
    slip = np.abs(lap.slip.to_numpy())
    dt = np.gradient(lap.t.to_numpy())
    steer = lap.steer.to_numpy()
    # steering turning back on itself by more than REVERSAL_SHARE of lock
    reversals, ref, trend = 0, steer[0], 0
    for v in steer[1:]:
        if trend == 0:
            if abs(v - ref) > REVERSAL_SHARE:
                trend, ref = (1 if v > ref else -1), v
        elif (v - ref) * trend > 0:
            ref = v                                     # still turning the same way
        elif abs(v - ref) > REVERSAL_SHARE:
            reversals += 1
            trend, ref = -trend, v
    seconds = float(dt.sum())
    sliding = slip >= SLIDING_DEG
    speed_slide = lap.speed.to_numpy()[sliding]
    angle_zone = float(slip[in_zone].mean()) if in_zone.any() else 0.0
    marks = {
        'steering': 1.0 - min(1.0, reversals / max(seconds, 1e-6) / 3.0),
        'steady angle': 1.0 - min(1.0, float(np.mean(spreads)) / 20.0) if spreads else 0.0,
        'steady pace': 1.0 - min(1.0, float(np.std(speed_slide) / max(np.mean(speed_slide), 1.0)) / 0.25) if len(speed_slide) else 0.0,
        'commitment': float(sliding[in_zone].mean()) if in_zone.any() else 0.0,
    }
    line = 40.0 * (float(np.mean(credits)) if credits else 0.0)
    angle = 40.0 * min(1.0, angle_zone / ANGLE_MAX_DEG)
    style = 20.0 * float(np.mean(list(marks.values())))
    signed = lap.slip.to_numpy()
    side = np.where(signed >= SLIDING_DEG, 1, np.where(signed <= -SLIDING_DEG, -1, 0))
    held = side[side != 0]
    flips = int((np.diff(held) != 0).sum()) if len(held) else 0
    return {'lap s': seconds, 'km/h': float(lap.speed.mean()), 'sliding': float(sliding.mean()),
            '40+': float((slip >= 40.0).mean()), 'zone angle': angle_zone, 'flips': flips,
            'reversals/s': reversals / max(seconds, 1e-6), 'line': line, 'angle': angle, 'style': style,
            'score': line + angle + style, 'marks': marks, 'zones': per_zone}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    ap.add_argument('--map', default=None, metavar='PNG',
                    help="also draw the course, its zones and each driver's first full lap (the rear axle's path)")
    args = ap.parse_args(argv)
    import build_zones as bz
    track = args.track or tf.TRACK
    edges = pd.read_csv(tf.path(args.tracks, 'course_edges', track))
    zones = bz.place_zones(edges)
    side = bz.zone_side_per_point(zones, len(edges))
    results, first_laps = [], []
    for path in args.files:
        for name, df in drivers_in(path):
            df = place(df, edges, side)
            judged = [judge_lap(df.iloc[a:b].reset_index(drop=True), zones) for a, b in laps(df, len(edges))]
            if not judged:
                print(f'{name}: no full lap')
                continue
            results.append((name, judged))
            a, b = laps(df, len(edges))[0]
            first_laps.append((name, df.iloc[a:b]))
    if not results:
        sys.exit('no full laps in these files')

    cols = ['lap s', 'km/h', 'sliding', '40+', 'zone angle', 'flips', 'reversals/s', 'line', 'angle', 'style', 'score']
    print(f"{track}: {len(zones)} zones. Medians of each driver's full laps (FD weights: line 40, angle 40, style 20)\n")
    print(f"{'driver':44s} {'laps':>4s} " + ' '.join(f'{c:>11s}' for c in cols))
    for name, judged in results:
        med = {c: float(np.median([j[c] for j in judged])) for c in cols}
        fmt = {'sliding': '{:.0%}', '40+': '{:.0%}', 'flips': '{:.0f}', 'reversals/s': '{:.2f}'}
        print(f"{name:44s} {len(judged):4d} " + ' '.join(f"{fmt.get(c, '{:.1f}').format(med[c]):>11s}" for c in cols))
    print('\nstyle marks (0-1), medians:')
    for name, judged in results:
        marks = {k: float(np.median([j['marks'][k] for j in judged])) for k in judged[0]['marks']}
        print(f"  {name:42s} " + '  '.join(f'{k} {v:.2f}' for k, v in marks.items()))
    print('\nzone by zone (median over laps): speed km/h / angle deg / filled % / past the edge % / rear axle inside the edge, m')
    for i, z in enumerate(zones):
        print(f"  zone {i + 1:2d} ({z['kind']}, {z['turned']:.0f} deg corner)")
        for name, judged in results:
            rows = np.array([j['zones'][i] for j in judged], dtype=float)
            if np.isnan(rows).all():
                continue
            m = np.nanmedian(rows, axis=0)
            print(f"      {name:40s} {m[0]:5.1f} / {m[1]:4.1f} / {100 * m[2]:4.0f} / {100 * m[3]:4.0f} / {m[4]:5.1f}")
    if args.map:
        draw(args.map, edges, zones, first_laps, track, tf.path(args.tracks, 'course_markers', track))
        print(f'\nmap: {args.map}')


def draw(out, edges, zones, first_laps, track, markers_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos = edges[['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy()
    neg = edges[['NegativeEdgeX', 'NegativeEdgeZ']].to_numpy()
    fig, ax = plt.subplots(figsize=(16, 13), dpi=110)
    for e, by_marker in ((pos, edges.PositiveFromMarkers.to_numpy() > 0), (neg, edges.NegativeFromMarkers.to_numpy() > 0)):
        ax.plot(*np.r_[e, e[:1]].T, color='0.55', lw=1.0)
        ax.scatter(*e[~by_marker].T, s=3, color='0.75', zorder=1)
    if os.path.exists(markers_path):
        mk = pd.read_csv(markers_path)
        ax.scatter(mk.X, mk.Z, s=10, color='0.15', marker='s', zorder=3, label='tire stacks and cones')
    for i, z in enumerate(zones):
        edge = neg if z['side'] < 0 else pos
        other = pos if z['side'] < 0 else neg
        pts = z['points']
        inward = other[pts] - edge[pts]
        inward /= np.linalg.norm(inward, axis=1)[:, None]
        band = edge[pts] + 3.0 * inward
        ax.fill(np.r_[edge[pts][:, 0], band[::-1, 0]], np.r_[edge[pts][:, 1], band[::-1, 1]],
                color='#f59e0b', alpha=0.45, zorder=2, label='zones (3 m deep)' if i == 0 else None)
        mid = edge[pts[len(pts) // 2]] - 6.0 * inward[len(pts) // 2]
        ax.text(*mid, str(i + 1), fontsize=11, weight='bold', color='#b45309', ha='center', va='center', zorder=6)
    colours = ['#2563eb', '#16a34a', '#9333ea', '#0891b2', '#dc2626', '#ea580c']
    for (name, lap), c in zip(first_laps, colours):
        style = '-' if 'human' in name else '--'
        ax.plot(lap.rear_x, lap.rear_z, style, color='#dc2626' if 'human' in name else c, lw=1.2,
                label=f"{name} (rear axle)", zorder=4)
    ax.set_aspect('equal')
    ax.set_title(f'{track}: zones and where the rear axle went (one full lap each)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out)


if __name__ == '__main__':
    main()
