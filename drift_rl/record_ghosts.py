"""
record_ghosts.py - drives to chase: a policy's runs on a course, kept as ghost leaders for tandem
training (drift_env chase episodes replay one ahead of the training car).

    python record_ghosts.py --track driftplayground_2021 [--policy policy.json] [--cars 16] [--ticks 2400]

Each car drives one long server-like run (server_run_configs) at a pace from --paces and a skill
from --skills; its ticks after the first --skip become one ghost: per tick the time, position,
heading, rear slip angle, yaw rate, speed, place along the line and offset from it. Written to
the track's folder as <track>_ghosts.npz.
"""
import argparse
import os

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy, default_policy_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--policy', default=default_policy_path())
    ap.add_argument('--cars', type=int, default=16, help='ghosts per pace and skill')
    ap.add_argument('--ticks', type=int, default=2400)
    ap.add_argument('--skip', type=int, default=100, help='ticks dropped at the start')
    ap.add_argument('--paces', type=float, nargs='+', default=[0.85, 0.9, 0.95, 1.0])
    ap.add_argument('--skills', type=float, nargs='+', default=[1.0, 0.5])
    args = ap.parse_args()

    pol = Policy.load_json(args.policy)
    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    ghosts = []
    for pace in args.paces:
        for skill in args.skills:
            cfg, rw = de.server_run_configs(pol.meta, ticks=args.ticks + 1, pace=pace, skill=skill)
            env = de.DriftVecEnv(w, n_envs=args.cars, env_config=cfg, reward_config=rw,
                                 seed=int(1000 * pace + 10 * skill))
            obs = env.reset()
            rows = np.zeros((args.ticks, args.cars, de.GH_COLS))
            t = np.zeros(args.cars)
            for k in range(args.ticks):
                dt = env.mids[:, de.M_DT].copy()
                obs, _, _, _, _ = env.step(np.asarray(pol(obs)))
                s, d = env.states, env.diags
                t += dt
                rows[k, :, de.GH_T] = t
                rows[k, :, de.GH_X] = s[:, fs.S_X]
                rows[k, :, de.GH_Z] = s[:, fs.S_Z]
                rows[k, :, de.GH_HEADING] = s[:, fs.S_HEADING]
                rows[k, :, de.GH_SLIP] = d[:, fs.D_RealSlipAngleRad]
                rows[k, :, de.GH_YAW] = s[:, fs.S_YAW]
                rows[k, :, de.GH_SPEED] = d[:, fs.D_SpeedKmh] / 3.6
                rows[k, :, de.GH_ARC] = s[:, fs.S_ARC]
                rows[k, :, de.GH_CTE] = d[:, fs.D_CrossTrackErrorM]
            for c in range(args.cars):
                g = rows[args.skip:, c].copy()
                g[:, de.GH_T] -= g[0, de.GH_T]
                ghosts.append(g)
            slide = np.mean(np.abs(rows[args.skip:, :, de.GH_SLIP]) > np.radians(20.0))
            print(f'pace {pace:.2f} skill {skill:.1f}: {args.cars} ghosts, '
                  f'{np.mean(rows[args.skip:, :, de.GH_SPEED]) * 3.6:.1f} km/h, sliding {100 * slide:.0f}%')
    out = os.path.join(tf.folder(args.tracks, args.track), f'{args.track}_ghosts.npz')
    np.savez_compressed(out, ghosts=np.stack(ghosts))
    print(f'{len(ghosts)} ghosts of {args.ticks - args.skip} ticks -> {out}')


if __name__ == '__main__':
    main()
