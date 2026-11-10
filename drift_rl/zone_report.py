"""
zone_report.py - how well policies fill each drift zone of a course, driving as the server does.

Per zone: the share of it with the rear axle in the zone while sliding (filled, the Formula Drift
style line score is 40 x the mean), where the rear axle runs (median distance inside the zone's
edge, with the 10th and 90th percentiles) and how far inside the edge the AI line itself is. With
--layout K the zones are one of the course's moved layouts (fast_sim.zone_variants, which training
with --zone-variants shows): a policy that reads the zone inputs should fill them about as well as
the zones it was trained on; one that does not, drives to where the old zones were.

    python zone_report.py runs/x/policy.json [more policies] --track spiritpeaksraceway --track driftplayground_2021
    python zone_report.py runs/x/policy.json --track spiritpeaksraceway --layout 3
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import build_zones as bz  # noqa: E402
import track_files as tf  # noqa: E402


def zones_of(side):
    """The zones in a per-point side column: list of point-index arrays with their side."""
    n = len(side)
    out = []
    start = int(np.argmax(side == 0)) if (side == 0).any() else 0
    j = 0
    while j < n:
        i = (start + j) % n
        if side[i] == 0:
            j += 1
            continue
        e = j
        while e + 1 < n and side[(start + e + 1) % n] == side[i]:
            e += 1
        out.append((int(side[i]), (start + np.arange(j, e + 1)) % n))
        j = e + 1
    return out


def drive(world, policy, ticks, cars, pace, grip):
    """Rear axle x, z, heading, slip angle (deg) and speed (km/h) of server-style runs."""
    import drift_env as de
    import fast_sim as fs
    cfg, never_reset = de.server_run_configs(policy.meta, ticks=ticks, sliding_grip=grip, pace=pace, skill=None)
    cfg = replace(cfg, recovery_start_fraction=0.0)
    env = de.DriftVecEnv(world, n_envs=cars, env_config=cfg, reward_config=never_reset, seed=8)
    b = env.base_params[fs.P_DIST_CG_REAR_M]
    obs = env.reset()
    rec = {k: [] for k in ('x', 'z', 'h', 'slip', 'kmh')}
    for _ in range(ticks):
        obs, *_ = env.step(policy(obs))
        s = env.states
        rec['x'].append(s[:, fs.S_X].copy())
        rec['z'].append(s[:, fs.S_Z].copy())
        rec['h'].append(s[:, fs.S_HEADING].copy())
        rec['slip'].append(np.degrees(np.arctan2(s[:, fs.S_LATV] - b * s[:, fs.S_YAW], np.maximum(s[:, fs.S_OWN_SPEED], 0.5))))
        rec['kmh'].append(np.hypot(s[:, fs.S_OWN_SPEED], s[:, fs.S_LATV]) * 3.6)
    return {k: np.array(v).ravel() for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+')
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', required=True)
    ap.add_argument('--layout', type=int, default=0, help='which zone layout (0 = the zones as given)')
    ap.add_argument('--pace', type=float, default=1.0)
    ap.add_argument('--grip', type=float, default=0.55, help='RearSlidingGrip')
    ap.add_argument('--cars', type=int, default=16)
    ap.add_argument('--ticks', type=int, default=3300, help='per car (3300 is about 3 minutes)')
    args = ap.parse_args()
    import drift_env as de
    import fast_sim as fs
    from policy import Policy

    if not 0 <= args.layout < fs.ZONE_VARIANTS:
        sys.exit(f'--layout: 0 - {fs.ZONE_VARIANTS - 1}')
    for course in args.track:
        world = fs.load_world(args.tracks, track=course)
        if not world.has_course:
            print(f"\n{course}: no course edges, no zones")
            continue
        world.pace_ref = de.course_pace_reference(args.tracks, course, world)
        side = world.co[:, fs.ZONE_COL + args.layout].copy()
        world.co[:, fs.ZONE_COL] = side                      # what a server-style run shows and pays
        zones = zones_of(side)
        edges = pd.read_csv(tf.path(args.tracks, 'course_edges', course))
        saved = bz.REAR_AXLE_BEHIND_CENTRE_M
        bz.REAR_AXLE_BEHIND_CENTRE_M = 0.0
        line_point, line_depth = bz.rear_depth(world.sp_pos[:, 0], world.sp_pos[:, 2], np.zeros(len(world.sp_pos)), edges, side)
        bz.REAR_AXLE_BEHIND_CENTRE_M = saved
        print(f"\n{course}: {len(zones)} zones (layout {args.layout}), {args.cars} cars x {args.ticks} ticks at pace {args.pace:g}")
        print("  AI line inside the zone edge, median m: " +
              ' '.join(f"{np.nanmedian(line_depth[np.isin(line_point, pts)]):.1f}" for _, pts in zones))
        for path in args.policies:
            run = drive(world, Policy.load_json(path), args.ticks, args.cars, args.pace, args.grip)
            keep = run['kmh'] > 25.0
            point, depth = bz.rear_depth(run['x'][keep], run['z'][keep], run['h'][keep], edges, side)
            slip = np.abs(run['slip'][keep])
            fills, rows = [], []
            for _, pts in zones:
                here = np.isin(point, pts)
                d = depth[here]
                if here.sum() < 5:
                    fills.append(np.nan)
                    rows.append('   -   ')
                    continue
                fills.append(float(((d >= 0) & (d <= bz.ZONE_DEPTH_M) & (slip[here] >= 20.0)).mean()))
                rows.append(f"{fills[-1]:4.0%} {np.median(d):4.1f}")
            print(f"  {os.path.relpath(path)[-40:]:40s} line {40 * np.nanmean(fills):4.1f}/40  slide {np.mean(slip >= 20):.0%}  "
                  f"km/h {np.median(run['kmh'][keep]):.0f}")
            print("    per zone (filled, rear m inside edge): " + ' | '.join(rows))


if __name__ == '__main__':
    main()
