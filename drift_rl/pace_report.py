"""
pace_report.py - how well a policy with paces drives the pace it is asked for.

    python pace_report.py runs/gen5/policy.json --track spiritpeaksraceway --track RiverSide_Drift
    python pace_report.py runs/gen5/policy.json --generated 4          # the held-out generated courses

For each course and each asked pace (fractions of the course's flat-out speed; --paces), the
speed it actually drove as a share of that flat-out speed, how much it slid, how often it
braked with the throttle, and how often it hit a marker. A policy that follows the pace drives
close to the diagonal (asked 0.80 -> drove ~0.80) without losing its slide.

The reference speed of each course (a pace of 1.0) comes from its <track>_pace_reference.csv in its
track folder, made by pace_model where missing (drift_env.course_pace_reference).
"""
import argparse
import os
import sys

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=[])
    ap.add_argument('--generated', type=int, default=0, help='also the first N held-out generated courses (seed 1001)')
    ap.add_argument('--paces', type=float, nargs='+', default=[0.7, 0.8, 0.9, 1.0, 1.05])
    ap.add_argument('--cars', type=int, default=48)
    ap.add_argument('--grip', type=float, default=None, help='RearSlidingGrip (default: the middle of the policy\'s range)')
    args = ap.parse_args()
    sys.path.insert(0, _HERE)
    import fast_sim as fs
    import drift_env as de
    from policy import Policy

    policy = Policy.load_json(args.policy)
    low, high = de.pace_range(policy.meta)
    if high <= 0.0:
        sys.exit(f"{args.policy} was not trained with paces as fractions (no pace_fraction_range)")
    grip = args.grip if args.grip is not None else 0.5 * sum(de.sliding_grip_range(policy.meta))
    courses = list(args.track)
    if args.generated:
        import generate_track as gt
        courses += [gt.ensure(args.tracks, 1001, k) for k in range(args.generated)]
    if not courses:
        sys.exit("give --track and/or --generated")
    data = args.tracks

    print(f"{args.policy}: trained for paces {low:g} - {high:g}; grip {grip:g}")
    print(f"{'course':24s} {'asked':>6s} {'drove':>6s} {'km/h':>6s} {'slide':>6s} {'slip':>5s} {'braking':>8s} {'markers/min':>12s} {'cut short':>10s}")
    for course in courses:
        gen = course.startswith('gen')
        w = fs.load_world(args.tracks, track=course, **(dict(fine_margin_m=40.0, fine_cell_size=4.0, rm_cell=6.0) if gen else {}))
        try:
            w.pace_ref = de.course_pace_reference(data, course, w)
        except RuntimeError as e:
            sys.exit(str(e))
        flat = float(w.sp_length.sum() / np.sum(w.sp_length / (w.pace_ref / 3.6)) * 3.6)
        for pace in args.paces:
            cfg = de.env_config_for(policy.meta, rear_sliding_grip_low=grip, rear_sliding_grip_high=grip,
                                    pace_low=pace, pace_high=pace)
            st = de.evaluate(w, policy, args.cars, seed=8, env_config=cfg)
            kmh = float((st['progress_m'] / np.maximum(st['sim_seconds'], 1e-9)).mean() * 3.6)
            print(f"{course:24s} {pace:6.2f} {kmh / flat:6.2f} {kmh:6.1f} {st['slide_fraction'].mean():6.2f} "
                  f"{st['mean_abs_slip_deg'].mean():5.1f} {st['brake_fraction'].mean():8.2f} "
                  f"{st['marker_hits_per_min'].mean():12.2f} {st['terminated_early'].mean():10.2f}", flush=True)
        print(f"{'':24s} (flat out: {flat:.1f} km/h)")


if __name__ == '__main__':
    main()
