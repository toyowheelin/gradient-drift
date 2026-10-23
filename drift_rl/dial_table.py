"""
dial_table.py - what each RearSlidingGrip setting does, for a policy trained over a range of it.

    python dial_table.py runs/<name>/policy.json
    python dial_table.py runs/<name>/policy.json --settings 0.4 0.5 0.6 --runs 256

For each setting: simulated 3-minute runs with no resets (the way the server runs a car), and
the median run's speed, drift time and tidiness. RearSlidingGrip is the plugin setting of the same
name (extra_cfg.yml): the share of its peak sideways grip the rear axle keeps once it is sliding.
Less grip = a slower drift through the same corner.

Settings outside the range the policy was trained over are skipped: the plugin would not run them
(it uses the nearest end of the range instead).
"""
import argparse
import json
import os
import sys

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_TICK_S = 0.0553
INCIDENT_OFFROAD_FRACTION = 0.05      # a run with more than this share of its time off the paving


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    ap.add_argument('--settings', type=float, nargs='+', default=None,
                    help='RearSlidingGrip values to try (default: every 0.1 across the trained range)')
    ap.add_argument('--runs', type=int, default=128)
    ap.add_argument('--minutes', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=2033)
    args = ap.parse_args()
    sys.path.insert(0, _HERE)
    import fast_sim as fs
    import drift_env as de
    from policy import Policy

    policy = Policy.load_json(args.policy)
    low, high = de.sliding_grip_range(policy.meta)
    if args.settings:
        settings = [g for g in args.settings if low - 1e-9 <= g <= high + 1e-9]
        skipped = [g for g in args.settings if g not in settings]
        if skipped:
            print(f"skipped (outside the trained range {low:g} - {high:g}): {', '.join(f'{g:g}' for g in skipped)}")
    else:
        settings = [round(g, 2) for g in np.arange(low, high + 1e-9, 0.1)]
        if not settings or abs(settings[-1] - high) > 1e-6:
            settings.append(high)
    world = fs.load_world(args.tracks, track=args.track)
    ticks = int(round(args.minutes * 60.0 / MEAN_TICK_S))
    print(f"{args.policy}: trained for RearSlidingGrip {low:g} - {high:g}")
    print(f"{args.runs} runs of {args.minutes:g} min ({ticks} ticks) per setting, no resets; median run "
          f"(middle 80% of runs for speed)\n")
    print(f"{'RearSlidingGrip':>15s}{'mean speed, km/h':>22s}{'sliding':>10s}{'at 40°+':>9s}{'straights':>11s}{'mean slip':>11s}"
          f"{'outside':>9s}{'markers/min':>13s}{'wall':>8s}{'recovery':>10s}{'bad runs':>10s}{'zones':>8s}{'steer rev/s':>13s}{'front >8°':>11s}{'wrong side':>12s}")
    rows = []
    for g in settings:
        cfg, never_reset = de.server_run_configs(policy.meta, ticks=ticks, sliding_grip=g)
        r = de.evaluate(world, policy, args.runs, seed=args.seed, env_config=cfg, reward_config=never_reset)
        med = lambda k: float(np.median(r[k]))
        lo, hi = (float(np.percentile(r['mean_speed_kmh'], q)) for q in (10, 90))
        row = {'rear_sliding_grip': g, 'mean_speed_kmh': med('mean_speed_kmh'), 'mean_speed_kmh_p10': lo,
               'mean_speed_kmh_p90': hi, 'drift_fraction': med('drift_fraction'),
               'slide_fraction': med('slide_fraction'), 'high_angle_fraction': med('high_angle_fraction'),
               'straight_drift_fraction': med('straight_drift_fraction'), 'mean_abs_slip_deg': med('mean_abs_slip_deg'),
               'offcourse_fraction': med('offcourse_fraction'), 'marker_hits_per_min': med('marker_hits_per_min'),
               'pushout_tick_fraction': float(r['pushout_tick_fraction'].mean()),
               'spin_recovery_fraction': float(r['spin_recovery_fraction'].mean()),
               'runs_with_incident': float((r['offroad_fraction'] > INCIDENT_OFFROAD_FRACTION).mean()),
               'zone_fill_fraction': med('zone_fill_fraction'), 'steer_reversals_per_s': med('steer_reversals_per_s'),
               'front_sliding_fraction': med('front_sliding_fraction'),
               'wrong_side_fraction': med('wrong_side_fraction')}
        rows.append(row)
        speed = f"{row['mean_speed_kmh']:.1f} ({lo:.1f} - {hi:.1f})"
        print(f"{g:>15.2f}{speed:>22s}{row['slide_fraction']:>10.1%}{row['high_angle_fraction']:>9.1%}"
              f"{row['straight_drift_fraction']:>11.1%}{row['mean_abs_slip_deg']:>10.1f}°{row['offcourse_fraction']:>9.1%}"
              f"{row['marker_hits_per_min']:>13.1f}{row['pushout_tick_fraction']:>8.2%}"
              f"{row['spin_recovery_fraction']:>10.2%}{row['runs_with_incident']:>10.1%}"
              f"{row['zone_fill_fraction']:>8.0%}{row['steer_reversals_per_s']:>13.1f}{row['front_sliding_fraction']:>11.1%}"
              f"{row['wrong_side_fraction']:>12.1%}", flush=True)
    out = os.path.splitext(args.policy)[0] + '_dial_table' + (f'_{args.track}' if args.track else '') + '.json'
    with open(out, 'w') as f:
        json.dump({'runs': args.runs, 'minutes': args.minutes, 'ticks': ticks, 'seed': args.seed, 'rows': rows}, f, indent=2)
    print(f"\n'sliding' = slip angle 20-80 deg; 'at 40°+' = the part of that at 40 deg or more."
          f"\n'zones' = share of the outside zones' length with the rear axle in the zone, sliding; "
          f"'steer rev/s' = times a second the steering action turns back by more than {de.STEER_REVERSAL_MIN:.0%} of lock; "
          f"'front >8°' = time with the front tires more than RewardConfig.front_slip_free_deg off the front axle's path; "
          f"'wrong side' = share of the long corners (past their entry) not sliding with the tail toward the outside."
          f"\n'bad runs' = more than {INCIDENT_OFFROAD_FRACTION:.0%} of the time off the paving. wall and recovery are "
          f"means over all runs.\nwritten: {out}")


if __name__ == '__main__':
    main()
