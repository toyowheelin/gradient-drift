"""
flips.py - how a policy changes direction, measured in simulated server-style runs.

    python flips.py runs/<name>/policy.json [--grip 0.5] [--set MaxSteerRateDegS=150] [--runs 32]

A "flip" is the car going from sliding 20 deg one way to sliding 20 deg the other. For each: how
long it takes, the fastest the car rotates, how far the car's path turns meanwhile (a car that
only spins on the spot turns its path by nothing), how fast the front wheels are steered, and the
hardest the rear tires would have to hold the car back (in g of the whole car) for its forward
speed to do what it did.
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))


def flips(dt, slip, yaw_degs, steer_deg, speed_kmh, path_rad, rear_pull_g):
    """Rows of per-flip numbers for one car's run."""
    sgn = np.sign(slip)
    rows = []
    for k in np.where((sgn[1:] != sgn[:-1]) & (sgn[1:] != 0))[0]:
        lo = k
        while lo > 1 and abs(slip[lo]) < 20 and k - lo < 40:
            lo -= 1
        hi = k + 1
        while hi < len(slip) - 1 and abs(slip[hi]) < 20 and hi - k < 40:
            hi += 1
        if abs(slip[lo]) < 20 or abs(slip[hi]) < 20 or sgn[lo] == sgn[hi] or lo < 2 or (rows and rows[-1][0] == hi):
            continue
        seg = slice(lo, hi + 1)
        steer_rate = np.abs(np.diff(steer_deg[lo - 1:hi + 1])) / dt[lo:hi + 1]
        rows.append((hi, dt[lo + 1:hi + 1].sum(), np.abs(yaw_degs[seg]).max(), np.degrees(abs(path_rad[hi] - path_rad[lo])),
                     steer_rate.max(), rear_pull_g[seg].min(), speed_kmh[lo] - speed_kmh[seg].min(),
                     abs(slip[hi] - slip[lo]) / dt[lo + 1:hi + 1].sum()))
    return [row[1:] for row in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    ap.add_argument('--grip', type=float, default=None)
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='a plugin setting to change')
    ap.add_argument('--runs', type=int, default=32)
    ap.add_argument('--ticks', type=int, default=3255)
    ap.add_argument('--seed', type=int, default=5)
    args = ap.parse_args()
    sys.path.insert(0, _HERE)
    import fast_sim as fs
    import drift_env as de
    from policy import Policy

    pol = Policy.load_json(args.policy)
    low, high = de.sliding_grip_range(pol.meta)
    grip = args.grip if args.grip is not None else (low + high) / 2
    cfg, never_reset = de.server_run_configs(pol.meta, ticks=args.ticks, sliding_grip=grip)
    extra = {n: float(v) for n, v in (x.split('=') for x in args.set)}
    cfg = replace(cfg, plugin_settings={**cfg.plugin_settings, **extra})
    env = de.DriftVecEnv(fs.load_world(args.tracks, track=args.track), args.runs, cfg, never_reset, seed=args.seed)
    p = env.base_params
    mass = (p[fs.P_FRONT_STATIC_LOAD_N] + p[fs.P_REAR_STATIC_LOAD_N]) / 9.80665
    a, b = p[fs.P_DIST_CG_FRONT_M], p[fs.P_DIST_CG_REAR_M]
    front_peak = p[fs.P_LAT_FRONT_PEAK_FRICTION] * p[fs.P_FRONT_STATIC_LOAD_N]
    obs = env.reset()
    rec = {k: [] for k in ('dt', 'x', 'z', 'own', 'latv', 'yaw', 'steer', 'recover', 'speed_action')}
    for _ in range(args.ticks):
        rec['dt'].append(env.mids[:, fs.M_DT].copy())
        action = pol(obs)
        rec['speed_action'].append(np.clip(action[:, 1], -1.0, 1.0))
        obs = env.step(action)[0]
        for k, i in (('x', fs.S_X), ('z', fs.S_Z), ('own', fs.S_OWN_SPEED), ('latv', fs.S_LATV), ('yaw', fs.S_YAW),
                     ('steer', fs.S_LAST_STEER), ('recover', fs.S_SPIN_RECOVERING)):
            rec[k].append(env.states[:, i].copy())
    r = {k: np.array(v)[100:-1] for k, v in rec.items()}       # the last tick is the next run's first
    slip = -np.degrees(np.arctan2(r['latv'], np.maximum(r['own'], 0.5)))
    speed = np.hypot(r['own'], r['latv']) * 3.6
    front_slip = np.arctan2(r['latv'] + a * r['yaw'], np.maximum(r['own'], 1.0)) - r['steer']
    fy_front = -np.sign(front_slip) * front_peak * np.minimum(1.0, np.abs(front_slip) / p[fs.P_FRONT_PEAK_SLIP_RAD])
    rear_pull = np.zeros_like(slip)
    rear_pull[1:] = ((r['own'][1:] - r['own'][:-1]) / r['dt'][1:] - r['latv'][:-1] * r['yaw'][:-1]
                     + fy_front[:-1] * np.sin(r['steer'][:-1]) / mass) / 9.80665
    rows = []
    path = np.zeros_like(slip)
    for i in range(args.runs):
        path[:, i] = np.unwrap(np.arctan2(np.gradient(r['z'][:, i]), np.gradient(r['x'][:, i])))
        rows += flips(r['dt'][:, i], slip[:, i], np.degrees(r['yaw'][:, i]), np.degrees(r['steer'][:, i]), speed[:, i],
                      path[:, i], rear_pull[:, i])
    R = np.array(rows)
    minutes = r['dt'].sum() / 60.0
    aslip = np.abs(slip)
    steer_rate = np.abs(np.diff(np.degrees(r['steer']), axis=0)) / r['dt'][1:]
    print(f"{args.policy}, RearSlidingGrip {grip:g}" + (f", {', '.join(args.set)}" if args.set else '')
          + f": {args.runs} runs, {minutes:.0f} min")
    print(f"  mean speed {speed.mean():.1f} km/h; sliding (20-80 deg) {np.mean((aslip >= 20) & (aslip < 80)):.1%}; "
          f"at 40 deg+ {np.mean((aslip >= 40) & (aslip < 80)):.1%}; mean slip {aslip.mean():.1f} deg; "
          f"recovering {np.mean(r['recover'] > 0.5):.2%}")
    print(f"  steering faster than 150 deg/s {np.mean(steer_rate > 150.01):.1%} of the time, 200 deg/s {np.mean(steer_rate > 200.01):.1%}; "
          f"rear tires holding the car back by more than 0.1 g {np.mean(rear_pull < -0.1):.1%}, 0.5 g {np.mean(rear_pull < -0.5):.1%}")
    print(f"  flips: {len(R) / minutes:.1f} a minute")
    if not len(R):
        return
    for j, name in enumerate(('time, s', 'fastest rotation, deg/s', 'path turns, deg', 'fastest steering, deg/s',
                              'hardest rear hold-back, g', 'speed lost mid-flip, km/h', 'slip angle change, deg/s')):
        q = np.percentile(R[:, j], (10, 50, 90))
        print(f"    {name:28s} 10th {q[0]:7.2f}   median {q[1]:7.2f}   90th {q[2]:7.2f}")

    # The median flip, sample by sample around the moment the slip angle crosses zero. Signs are
    # turned so that every flip ends at a positive slip angle. Forces are worked out from the
    # car's motion: sideways acceleration a = d(lateral velocity)/dt + forward speed x yaw rate,
    # rear side force = (m a x front arm - yaw inertia x yaw acceleration) / wheelbase. The path
    # direction is from where it points at the crossing, positive the way the car is rotating:
    # flat before and rising after is a car thrown onto its new side and carried by it.
    grid = np.arange(-0.6, 0.81, 0.2)
    side_g = np.zeros_like(slip)
    side_g[1:] = ((r['latv'][1:] - r['latv'][:-1]) / r['dt'][1:]
                  + r['own'][1:] * 0.5 * (r['yaw'][1:] + r['yaw'][:-1])) / 9.80665
    yaw_acc = np.zeros_like(slip)
    yaw_acc[1:] = (r['yaw'][1:] - r['yaw'][:-1]) / r['dt'][1:]
    rear_force = (mass * side_g * 9.80665 * a - p[fs.P_YAW_INERTIA_KGM2] * yaw_acc) / (a + b)
    rear_grip = np.abs(rear_force) / p[fs.P_REAR_STATIC_LOAD_N]
    series = {'slip angle, deg': (slip, True), 'sideways acceleration, g': (np.abs(side_g), False),
              'rear side force / load': (rear_grip, False), 'speed action': (r['speed_action'], False),
              'speed, km/h': (speed, False), 'path direction, deg': (np.degrees(path), None),
              'front slip, deg': (np.degrees(front_slip), True)}
    profile = {name: [] for name in series}
    turn_sign = np.sign(np.sum(np.diff(path, axis=0) * r['yaw'][1:]))      # path angle vs yaw rate handedness
    for i in range(args.runs):
        clock = np.cumsum(r['dt'][:, i])
        sgn = np.sign(slip[:, i])
        for k in np.where((sgn[1:] != sgn[:-1]) & (sgn[1:] != 0))[0]:
            if clock[k] + grid[0] < clock[0] or clock[k] + grid[-1] > clock[-1]:
                continue
            ahead = np.interp(clock[k] + 0.4, clock, slip[:, i])
            behind = np.interp(clock[k] - 0.4, clock, slip[:, i])
            if abs(ahead) < 20 or abs(behind) < 20 or ahead * behind > 0:
                continue                                    # not a flip from one slide into the other
            for name, (values, signed) in series.items():
                row = np.interp(clock[k] + grid, clock, values[:, i])
                if signed is None:
                    row = turn_sign * np.sign(r['yaw'][k, i]) * (row - np.interp(clock[k], clock, values[:, i]))
                profile[name].append(np.sign(ahead) * row if signed else row)
    print(f"  the median flip ({len(profile['speed, km/h'])} of them), slip angle crossing zero at 0:")
    print(f"    {'s from the crossing':28s}" + ''.join(f"{g:8.1f}" for g in grid))
    for name in series:
        print(f"    {name:28s}" + ''.join(f"{v:8.2f}" for v in np.median(np.array(profile[name]), axis=0)))


if __name__ == '__main__':
    main()
