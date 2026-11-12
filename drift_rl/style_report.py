"""
style_report.py - does a policy follow the style dials (drift_env STYLE_*)?

    python style_report.py runs/<name>/policy.json --track driftplayground_2021 [--cars 16 --ticks 2400]

Drives the policy server-like (no resets) with every dial at 0, then each dial alone at -1 and +1,
and prints what the dial is meant to move, measured the way the players' style catalogue measures
it (project notes, 2026-11-08 / 11-12):
  angle   the median drift angle while sliding (deg)
  swing   how much deeper the angle runs in tight parts than on open ones: median angle where the
          line's radius is 25 m or less minus where it is 60 m or more (deg)
  commit  speed where a corner starts (the line tightening ahead, as the reward's _corner_entry)
          over the speed asked there, minus 1
  smooth  steering jitter: the steering's spread around its own 0.5 s average (deg of lock)
plus speed, the share of time sliding and off the course, so a dial that buys its style by
spinning or leaving the course shows. A policy trained without the dials shows the same numbers
on every row except commit's: that dial moves the speed asked of the car, which every pace policy
already sees (the pace_over input).
"""
import argparse

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy, default_policy_path

DIALS = ('angle', 'swing', 'commit', 'smooth')


def run(w, pol, cars, ticks, style, pace, seed):
    cfg, rw = de.server_run_configs(pol.meta, ticks=ticks + 1, pace=pace, style=style)
    env = de.DriftVecEnv(w, n_envs=cars, env_config=cfg, reward_config=rw, seed=seed)
    obs = env.reset()
    _, _, la_min = de.spline_lookahead_tables(w)
    slip, steer, speed, entry, ask, radius = ([] for _ in range(6))
    for _ in range(ticks):
        obs, _, _, _, _ = env.step(np.asarray(pol(obs)))
        s, d = env.states, env.diags
        near = d[:, fs.D_NearestPointId].astype(int).clip(0, len(w.sp_radius) - 1)
        slip.append(np.degrees(np.abs(d[:, fs.D_RealSlipAngleRad])))
        steer.append(np.degrees(s[:, fs.S_LAST_STEER]))
        speed.append(d[:, fs.D_SpeedKmh])
        r = w.sp_radius[near]
        radius.append(r)
        entry.append(np.clip(20.0 / np.maximum(la_min[near], 1.0) - 20.0 / np.maximum(r, 1.0), 0.0, 1.0))
        ref = w.pace_ref[near] if getattr(w, 'pace_ref', None) is not None and len(w.pace_ref) == len(w.sp_radius) else np.zeros(cars)
        ask.append(s[:, fs.S_PACE] * ref)
    slip, steer, speed, entry, ask, radius = map(np.array, (slip, steer, speed, entry, ask, radius))
    sliding = slip > 15.0
    tight, open_ = sliding & (radius <= 25.0), sliding & (radius >= 60.0)
    swing = (np.median(slip[tight]) - np.median(slip[open_])) if tight.any() and open_.any() else np.nan
    e = (entry > 0.3) & (ask > 1.0)
    commit = np.median(speed[e] / ask[e]) - 1.0 if e.any() else np.nan
    k = 9
    kern = np.ones(k) / k
    jit = [np.std(steer[:, c] - np.convolve(steer[:, c], kern, 'same')) for c in range(cars)]
    off_share = float(np.mean(env.ep[:, de.E_OFFCOURSE_TICKS])) / max(ticks, 1)
    return dict(angle=float(np.median(slip[sliding])) if sliding.any() else np.nan, swing=float(swing),
                commit=float(commit), smooth=float(np.median(jit)), kmh=float(np.mean(speed)),
                slide=float(np.mean(sliding)), off=off_share)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy', nargs='?', default=default_policy_path())
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None)
    ap.add_argument('--cars', type=int, default=16)
    ap.add_argument('--ticks', type=int, default=2400)
    ap.add_argument('--pace', type=float, default=0.95)
    ap.add_argument('--values', type=float, nargs='+', default=[-1.0, 1.0])
    ap.add_argument('--seed', type=int, default=2045)
    args = ap.parse_args()
    pol = Policy.load_json(args.policy)
    track = args.track or tf.TRACK
    w = fs.load_world(args.tracks, track=track)
    w.pace_ref = de.course_pace_reference(args.tracks, track, w)
    trained = pol.meta.get('style_range')
    print(f'{args.policy} on {track}: style dials {"trained over " + str(trained) if trained else "NOT trained (rows should match)"}')
    print(f'{"dials":24s} {"angle":>6s} {"swing":>6s} {"commit":>7s} {"smooth":>7s} {"km/h":>6s} {"slide":>6s} {"off":>6s}')
    rows = [(0.0, 0.0, 0.0, 0.0)]
    for j in range(len(DIALS)):
        for v in args.values:
            st = [0.0] * len(DIALS)
            st[j] = v
            rows.append(tuple(st))
    for st in rows:
        r = run(w, pol, args.cars, args.ticks, st, args.pace, args.seed)
        name = 'all 0' if not any(st) else ' '.join(f'{d} {v:+g}' for d, v in zip(DIALS, st) if v)
        print(f'{name:24s} {r["angle"]:6.1f} {r["swing"]:6.1f} {r["commit"]:+7.3f} {r["smooth"]:7.2f} '
              f'{r["kmh"]:6.1f} {r["slide"]:6.0%} {r["off"]:6.1%}', flush=True)


if __name__ == '__main__':
    main()
