"""
amp_reference.py - a human's driving as the motion prior for train_ppo_gpu.py --amp
(Adversarial Motion Priors: Peng, Ma, Abbeel, Levine, Kanazawa, SIGGRAPH 2021,
https://arxiv.org/abs/2104.02180).

    python amp_reference.py <telemetria>.csv [more.csv ...] [--out amp_reference.npz]
    python amp_reference.py ai_car_monitor_<date>_<time>.csv[:4,8] ...

Each recording (the in-game telemetry app's CSV, or an AI Car Monitor CSV of every car on a public
server; telemetry_ghosts.py reads both the same way) becomes the car's state every AMP_DT seconds,
about the simulator's tick: speed, body slip and yaw rate, scaled as the policy's own inputs 0, 3
and 2 are (own_speed/25, body_slip/(pi/2), yaw_rate/2, with the policy's sign for the slip). Pairs
of consecutive states above MIN_KMH are written, each also mirrored (left and right turns alike).

The trainer's discriminator learns to tell these pairs from the training cars' own; the cars are
paid for transitions it takes for the human's. Only the motion is compared, not the course or the
controls, so a recording from any track teaches how a drift moves: how deep, how fast it swings
from one side to the other, how much speed it keeps.
"""
import argparse
import os

import numpy as np

import telemetry_ghosts as tg

AMP_DT = 0.05          # s between the states of a pair (the simulator's tick is 0.049 s)
MIN_KMH = 25.0
N_FEATURES = 3         # per state; train_ppo_gpu.py reads the same three inputs
OBS_SLOTS = (0, 3, 2)  # own_speed/25, body_slip/(pi/2), yaw_rate/2 (drift_env.OBS_NAMES)


def features(path, cars=None):
    """(n, N_FEATURES) on an even AMP_DT grid, for each stretch of path (an AI Car Monitor CSV: of
    those cars, None = all)."""
    out = []
    parts = tg.crowd_paths(path, 0.13, 250.0, cars) if tg.is_crowd(path) else [(0, '', tg.car_path(path, 2.75, 0.54, 0.13))]
    for _, _, p in parts:
        grid = np.arange(p['t'][0], p['t'][-1], AMP_DT)
        if len(grid) < 3:
            continue
        f = np.stack([np.interp(grid, p['t'], p['speed']) / 25.0,
                      -np.interp(grid, p['t'], p['slip']) / (np.pi / 2),     # the policy's slip sign
                      np.interp(grid, p['t'], p['yaw']) / 2.0], axis=1)
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recordings', nargs='+')
    ap.add_argument('--out', default='amp_reference.npz')
    args = ap.parse_args()
    pairs, names = [], []
    for spec in args.recordings:
        path, cars = spec, None
        if not os.path.exists(spec) and ':' in spec:
            path, _, ids = spec.rpartition(':')
            cars = {int(c) for c in ids.split(',') if c}
        for part in features(path, cars):
            pr = np.concatenate([part[:-1], part[1:]], axis=1)
            pr = pr[pr[:, 0] * 25.0 * 3.6 > MIN_KMH]
            if len(pr) == 0:
                continue
            mirror = pr.copy()
            mirror[:, [1, 2, 4, 5]] *= -1.0
            pairs += [pr, mirror]
        names.append(os.path.basename(path))
    if not pairs:
        raise SystemExit('no driving above %g km/h in those recordings' % MIN_KMH)
    pairs = np.concatenate(pairs).astype(np.float32)
    slide = np.mean(np.abs(pairs[:, 1]) * 90.0 > 20.0)
    np.savez_compressed(args.out, pairs=pairs, dt=AMP_DT, sources=np.array(names))
    print(f"{len(pairs):,} state pairs ({len(pairs) // 2 * AMP_DT / 60:.1f} min of driving, mirrored), "
          f"{slide:.0%} of them sliding over 20 deg, median {np.median(pairs[:, 0]) * 25 * 3.6:.0f} km/h -> {args.out}")


if __name__ == '__main__':
    main()
