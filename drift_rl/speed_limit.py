"""
speed_limit.py - what stops a policy going faster when it is asked to: the tyres or the engine.

    python speed_limit.py runs/trialT/policy_final.json --track driftplayground_2021 --paces 1.0 1.2

For each asked pace (a share of the course's flat-out speed), per tick of every car, split by
where on the course it is (turning tighter than --corner-radius m, or not):
  front sat / rear sat  share of ticks the axle's sideways force is at 95% of what it has (the
                        tyres can give no more)
  full gas              share of ticks the pedal action is 0.95 or more
  at top                share of ticks within 3% of the car's top speed
  accel                 mean longitudinal acceleration, m/s^2
  lat g                 mean sideways acceleration (speed x the turn rate of its path), in g
A car held back by its tyres sits at front sat / rear sat in the corners with the pedal
eased; one held back by its engine is at full gas without accelerating.
"""
import argparse
import math

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=[])
    ap.add_argument('--paces', type=float, nargs='+', default=[1.0, 1.2])
    ap.add_argument('--cars', type=int, default=48)
    ap.add_argument('--ticks', type=int, default=1200)
    ap.add_argument('--corner-radius', type=float, default=50.0)
    args = ap.parse_args()

    policy = Policy.load_json(args.policy)
    grip = 0.5 * sum(de.sliding_grip_range(policy.meta))
    print(f"{args.policy}: grip {grip:g}")
    print(f"{'course':24s} {'asked':>5s} {'where':>8s} {'share':>6s} {'km/h':>6s} {'front sat':>9s} {'rear sat':>9s} "
          f"{'full gas':>9s} {'at top':>7s} {'accel':>6s} {'lat g':>6s}")
    for course in args.track:
        w = fs.load_world(args.tracks, track=course)
        w.pace_ref = de.course_pace_reference(args.tracks, course, w)
        for pace in args.paces:
            cfg = de.env_config_for(policy.meta, rear_sliding_grip_low=grip, rear_sliding_grip_high=grip,
                                    pace_low=pace, pace_high=pace, recovery_start_fraction=0.0)
            env = de.DriftVecEnv(w, n_envs=args.cars, env_config=cfg, seed=8)
            obs = env.reset()
            rows = []
            prev_dir = None
            for k in range(args.ticks):
                act = policy(obs)
                obs, r, term, trunc, info = env.step(act)
                d = env.diags
                s = env.states
                vdir = np.arctan2(d[:, fs.D_Vz], d[:, fs.D_Vx])
                dt = np.maximum(env.mids[:, fs.M_DT], 1e-3)
                if prev_dir is not None and k > 40:
                    turn = np.arctan2(np.sin(vdir - prev_dir), np.cos(vdir - prev_dir)) / dt
                    v = d[:, fs.D_SpeedKmh] / 3.6
                    ok = ~(term | trunc)
                    rows.append(np.c_[d[:, fs.D_Radius], v, d[:, fs.D_FrontSaturation], d[:, fs.D_RearSaturation],
                                      act[:, 1], v / np.maximum(s[:, fs.S_MAX_SPEED], 0.1),
                                      d[:, fs.D_RealizedLongitudinalAccelMs2], np.abs(v * turn) / 9.81][ok])
                prev_dir = vdir
            a = np.concatenate(rows)
            for where, m in (('corner', a[:, 0] < args.corner_radius), ('other', a[:, 0] >= args.corner_radius)):
                if not m.any():
                    continue
                b = a[m]
                print(f"{course:24s} {pace:5.2f} {where:>8s} {m.mean():6.2f} {b[:, 1].mean() * 3.6:6.1f} "
                      f"{(b[:, 2] >= 0.95).mean():9.2f} {(b[:, 3] >= 0.95).mean():9.2f} {(b[:, 4] >= 0.95).mean():9.2f} "
                      f"{(b[:, 5] >= 0.97).mean():7.2f} {b[:, 6].mean():6.2f} {b[:, 7].mean():6.2f}", flush=True)


if __name__ == '__main__':
    main()
