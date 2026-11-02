"""
pick_policy.py - the same fixed set of scenarios for every candidate policy, scored in one table,
to choose which file to ship or test on a server.

    python pick_policy.py runs/x/policy.json runs/x/policy_final.json runs/x/snapshots/*.json \
        --track driftplayground_2021 --track spiritpeaksraceway --track adc_klutch_kickers_drifters_paradise \
        [--car-pool cars/car_pool.json] [--cars 32] [--by-lead]

Why: the trainer keeps the policy with the best training eval (policy.json) and the last one
(policy_final.json), and the two often disagree. GT Sophy (Wurman et al., Nature 2022, "Policy
selection") met the same thing: after convergence, later policies "differed in small ways", better
at one skill and worse at another. They saved policies at regular intervals (here: the trainer's
--snapshot-every), ran each through a fixed set of scenarios, put the scores in one sheet and
chose from it. Same seeds for every policy, so the numbers are like for like.

Scenarios, on each --track:
  solo 1.0     a server-like run (nothing resets the car) at pace 1.0: slide, walls, off course, spins,
               markers, weaves and swings a minute
  solo 0.85    the same at pace 0.85 (a spaced-out bot): slide, spins
  low grip     pace 1.0 at RearSlidingGrip --low-grip (0.5: what servers 6 and 7 once had): slide, spins, walls
  chase        chase episodes behind the track's ghosts over the whole gap range, touches do not end
               them: gap, in band, chase drift, hits, crash after a touch
  chase close  the same with asks in the closest third: hits, in band

The summary ranks the policies on each column and marks a policy another one beats or ties on every
key column (dominated): those are not worth a server test.
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy

# (column, higher is better)
KEY = [('slide', True), ('walls', False), ('spin', False), ('slide .85', True), ('lowgrip spin', False),
       ('chase drift', True), ('in band', True), ('hits', False), ('close hits', False)]


def solo(world, pol, pace, cars, ticks, car_pool, sliding_grip=None, seed=11):
    cfg, never_reset = de.server_run_configs(pol.meta, ticks=ticks, pace=pace, sliding_grip=sliding_grip)
    if car_pool:
        cfg = replace(cfg, car_pool=car_pool)
    return de.evaluate(world, pol, cars, seed=seed, env_config=cfg, reward_config=never_reset)


def chase(world, pol, gaps, episodes, car_pool, seed=7):
    cfg = replace(de.env_config_for(pol.meta), chase_gap_low=gaps[0], chase_gap_high=gaps[1], chase_share=1.0)
    if car_pool:
        cfg = replace(cfg, car_pool=car_pool)
    ev = de.evaluate(world, pol, episodes, seed=seed, env_config=cfg,
                     reward_config=de.RewardConfig(chase_contact_ends=0.0))
    had = ev['chase_fraction'] > 0
    return {k: v[had] for k, v in ev.items()}


def chase_numbers(c):
    if len(c['length']) == 0:
        return dict(gap=np.nan, band=np.nan, drift=np.nan, hits=np.nan, crash=np.nan)
    w = c['chase_fraction'] * c['length']
    wm = lambda k: float(np.sum(c[k] * w) / max(np.sum(w), 1e-9))
    touches = max(float(np.sum(c['contacts'])), 1.0)
    return dict(gap=wm('chase_mean_gap_m'), band=wm('chase_in_band_fraction'), drift=wm('chase_drift_fraction'),
                hits=float(np.mean(c['contacts'] > 0)), crash=float(np.sum(c['touch_then_crash'])) / touches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+')
    ap.add_argument('--track', action='append', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--car-pool', default=None, help='cars/car_pool.json: drive the pool (default: the default car)')
    ap.add_argument('--cars', type=int, default=32, help='server-like runs per solo scenario')
    ap.add_argument('--ticks', type=int, default=1500, help='ticks per server-like run (1500 is about 80 s)')
    ap.add_argument('--episodes', type=int, default=256, help='chase episodes per chase scenario')
    ap.add_argument('--low-grip', type=float, default=0.5)
    ap.add_argument('--by-lead', action='store_true', help="chase_report.py's split by lead for the chase scenario")
    args = ap.parse_args()
    car_pool = tuple(v for _, v in de.load_car_pool(args.car_pool)) if args.car_pool else ()
    pols = [(p, Policy.load_json(p)) for p in args.policies]
    names = [p if len(p) <= 34 else '...' + p[-31:] for p, _ in pols]
    table = {n: {k: [] for k, _ in KEY} for n in names}

    for track in args.track:
        w = fs.load_world(args.tracks, track=track)
        w.pace_ref = de.course_pace_reference(args.tracks, track, w)
        w.ghosts = de.load_ghosts(args.tracks, track)
        kinds = de.load_ghost_kinds(args.tracks, track) if (args.by_lead and w.ghosts is not None) else None
        print(f"\n{track}")
        print(f"  {'policy':<34} {'slide':>6} {'walls':>6} {'off':>6} {'spin':>6} {'mark/min':>8} {'weave':>6} {'swing':>6}"
              f" | {'.85 slide':>9} {'spin':>6} | {'low slide':>9} {'spin':>6} {'walls':>6}"
              f" | {'gap':>5} {'band':>5} {'drift':>6} {'hits':>6} {'crash':>6} | {'close hits':>10} {'band':>5}")
        for name, (path, pol) in zip(names, pols):
            if len(pol.obs_mean) > de.PACE_OVER_SLOT and w.pace_ref is None:
                raise SystemExit(f'{track} has no pace reference')
            a = solo(w, pol, 1.0, args.cars, args.ticks, car_pool)
            b = solo(w, pol, 0.85, args.cars, args.ticks, car_pool)
            g = solo(w, pol, 1.0, args.cars, args.ticks, car_pool, sliding_grip=args.low_grip)
            m = lambda r, k: float(np.mean(r[k]))
            if w.ghosts is not None:
                full = chase(w, pol, (0.0, 1.0), args.episodes, car_pool)
                close = chase(w, pol, (0.0, 1.0 / 3.0), args.episodes, car_pool, seed=8)
                cn, cc = chase_numbers(full), chase_numbers(close)
            else:
                full = None
                cn = cc = chase_numbers({'length': np.array([])})
            row = table[name]
            for k, v in (('slide', m(a, 'slide_fraction')), ('walls', m(a, 'pushout_tick_fraction')),
                         ('spin', m(a, 'spin_recovery_fraction')), ('slide .85', m(b, 'slide_fraction')),
                         ('lowgrip spin', m(g, 'spin_recovery_fraction')), ('chase drift', cn['drift']),
                         ('in band', cn['band']), ('hits', cn['hits']), ('close hits', cc['hits'])):
                row[k].append(v)
            print(f"  {name:<34} {m(a, 'slide_fraction'):6.0%} {m(a, 'pushout_tick_fraction'):6.2%} "
                  f"{m(a, 'offcourse_fraction'):6.1%} {m(a, 'spin_recovery_fraction'):6.2%} {m(a, 'marker_hits_per_min'):8.2f} "
                  f"{m(a, 'weaves_per_min'):6.1f} {m(a, 'swings_per_min'):6.1f}"
                  f" | {m(b, 'slide_fraction'):9.0%} {m(b, 'spin_recovery_fraction'):6.2%}"
                  f" | {m(g, 'slide_fraction'):9.0%} {m(g, 'spin_recovery_fraction'):6.2%} {m(g, 'pushout_tick_fraction'):6.2%}"
                  f" | {cn['gap']:5.1f} {cn['band']:5.0%} {cn['drift']:6.0%} {cn['hits']:6.1%} {cn['crash']:6.0%}"
                  f" | {cc['hits']:10.1%} {cc['band']:5.0%}")
            if kinds is not None and full is not None and len(full['length']):
                import chase_report
                chase_report.by_lead(full, kinds, w.ghosts)

    print(f"\nall tracks (mean), rank on each key column (1 = best), and whether another policy is at least as good on all")
    print(f"  {'policy':<34} " + " ".join(f"{k:>12}" for k, _ in KEY) + "  dominated by")
    means = {n: {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in table[n].items()}
             for n in names}
    for n in names:
        cells = []
        for k, higher in KEY:
            vals = [means[o][k] for o in names]
            v = means[n][k]
            rank = 1 + sum(1 for x in vals if np.isfinite(x) and np.isfinite(v) and ((x > v) if higher else (x < v)))
            cells.append(f"{v:8.1%} ({rank})" if np.isfinite(v) else f"{'-':>12}")
        beaten_by = []
        for o in names:
            if o == n:
                continue
            ok = [(means[o][k] >= means[n][k]) if higher else (means[o][k] <= means[n][k])
                  for k, higher in KEY if np.isfinite(means[o][k]) and np.isfinite(means[n][k])]
            strict = [(means[o][k] > means[n][k]) if higher else (means[o][k] < means[n][k])
                      for k, higher in KEY if np.isfinite(means[o][k]) and np.isfinite(means[n][k])]
            if ok and all(ok) and any(strict):
                beaten_by.append(o)
        print(f"  {n:<34} " + " ".join(cells) + ("  " + ", ".join(beaten_by) if beaten_by else ""))


if __name__ == '__main__':
    main()
