"""
seed_pace.py - teach a policy the move for driving slower than flat out before training it:
the throttle held on and the brake applied, which scrubs speed and keeps the drift.

    python seed_pace.py runs/gen4/policy.json \\
        --track spiritpeaksraceway --track kb3k__House_of_RWD --track RiverSide_Drift --generated 20 \\
        --pace 0.7 1.05 --out runs/gen4_seeded
    python train_ppo_gpu.py --init-policy runs/gen4_seeded/policy.json --pace 0.7 1.05 ...

Why: braking with the throttle on holds the slide down to about 80% of a course's flat-out speed
(README, "Pace results"), but trained policies never found it - lifting, the move they try first,
straightens the car - so they barely respond to the pace they are asked for. This shows them.

How (imitation with DAgger): a teacher drives - the starting policy's own steering and throttle,
with a scripted rule on top only while the car is faster than asked: the throttle at least
THROTTLE_HELD and the brake GAIN x how far too fast. The network (starting from the policy, sized
to this simulator's inputs and outputs) is fitted to the teacher's actions from what the network
itself sees; then it drives, the teacher says what it would have done in the states the network
actually reaches, and it is fitted again, for --rounds rounds. The rule is a lesson, not part of
anything that ships: the result is an ordinary policy file, and train_ppo_gpu.py then refines it
with the pace reward. Each course's reference speed (a pace of 1.0) is kept with its track data
(<track>_pace_reference.csv), made by pace_model the first time.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
GAIN = 4.0              # brake per unit of "share too fast" (0.25 too fast = full brake)
THROTTLE_HELD = 0.6     # the least throttle while braking: lifting is what loses the slide


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy', help='the starting policy (any earlier inputs/outputs; widened here)')
    ap.add_argument('--pace-reference', default=None, metavar='POLICY',
                    help='no longer used: a pace of 1.0 is pace_model\'s speed round each course')
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=[])
    ap.add_argument('--generated', type=int, default=0, help='also this many generated courses (seed 1)')
    ap.add_argument('--gen-seed', type=int, default=1)
    ap.add_argument('--pace', type=float, nargs=2, default=(0.7, 1.05), metavar=('LOW', 'HIGH'))
    ap.add_argument('--rounds', type=int, default=4, help='DAgger rounds after the teacher\'s own')
    ap.add_argument('--cars', type=int, default=64, help='cars per course per round')
    ap.add_argument('--ticks', type=int, default=600, help='ticks per car per round')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--keep-every', type=int, default=3, help='keep every Nth tick of each run (memory)')
    ap.add_argument('--out', required=True, help='folder for policy.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, _HERE)
    import torch
    import fast_sim as fs
    import drift_env as de
    from policy import MLP, Policy
    from train_ppo_gpu import widen, BRAKE_INIT_BIAS

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    start = Policy.load_json(args.policy)
    de.require_direct(start.meta)
    grip_low, grip_high = de.sliding_grip_range(start.meta)
    grip = 0.5 * (grip_low + grip_high)

    # ---- the student: the starting policy sized to this simulator, new inputs with no weight ----
    n_old = len(start.obs_mean)
    hidden = start.mlp.W[0].shape[0]
    layers = widen(start.mlp.W, start.mlp.b, de.OBS_SIZE, hidden, rng, n_outputs=de.ACT_SIZE,
                   new_output_bias=BRAKE_INIT_BIAS)
    old_names = list(start.meta.get('obs_names') or [])
    mean = np.r_[start.obs_mean, [de.NEW_INPUT_NORM.get(n, (0.6, 0.5))[0] for n in de.OBS_NAMES[n_old:]]]
    std = np.r_[start.obs_std, [de.NEW_INPUT_NORM.get(n, (0.6, 0.5))[1] for n in de.OBS_NAMES[n_old:]]]
    for j in range(min(n_old, len(old_names))):
        if old_names[j] != de.OBS_NAMES[j] and de.OBS_NAMES[j] in de.NEW_INPUT_NORM:
            layers[0][0][:, j] = 0.0
            mean[j], std[j] = de.NEW_INPUT_NORM[de.OBS_NAMES[j]]
            print(f"input {j} was '{old_names[j]}', now '{de.OBS_NAMES[j]}': no weight to start")
    net = torch.nn.Sequential(torch.nn.Linear(de.OBS_SIZE, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, de.ACT_SIZE)).double()
    with torch.no_grad():
        for lin, (w, b) in zip([m for m in net if isinstance(m, torch.nn.Linear)], layers):
            lin.weight.copy_(torch.as_tensor(w))
            lin.bias.copy_(torch.as_tensor(b))
    mean_t, std_t = torch.as_tensor(mean), torch.as_tensor(std)

    def student(obs):
        with torch.no_grad():
            x = torch.clamp((torch.as_tensor(obs) - mean_t) / std_t, -10, 10)
            return torch.clamp(net(x), -1, 1).numpy()

    def teacher(obs, counters, pace_on):
        """The starting policy's steering and throttle; while too fast for the asked pace,
        throttle held and brake by how much too fast. Returns (actions, too_fast)."""
        a = np.asarray(start(obs), dtype=float)
        act = np.zeros((len(obs), de.ACT_SIZE))
        act[:, :2] = a[:, :2]
        over = np.where(counters[:, 9] > 0.5, counters[:, 8] / np.maximum(counters[:, 9], 1e-6) - 1.0, 0.0)
        over = np.where(pace_on, over, 0.0)
        fast = over > 0.0
        act[:, 1] = np.where(fast, np.maximum(act[:, 1], THROTTLE_HELD), act[:, 1])
        act[:, 2] = np.where(fast, np.clip(GAIN * over, 0.0, 1.0), 0.0)
        return act, over

    # ---- the courses and each one's flat-out speed ----
    data = args.tracks
    names = list(args.track)
    if args.generated:
        import generate_track as gt
        names += [gt.ensure(data, args.gen_seed, k) for k in range(args.generated)]
    if not names:
        sys.exit("give --track and/or --generated")
    worlds = []
    t0 = time.time()
    for n in names:
        kw = dict(fine_margin_m=40.0, fine_cell_size=4.0, rm_cell=6.0) if n.startswith('gen') else {}
        w = fs.load_world(args.tracks, track=n, **kw)
        w.pace_ref = de.course_pace_reference(data, n, w, args.pace_reference, sliding_grip=grip)
        worlds.append((n, w))
    print(f"{len(worlds)} courses, flat-out speeds ready in {time.time() - t0:.0f} s", flush=True)

    # ---- collect: the teacher drives (round 0), then the student with the teacher's labels ----
    data_obs, data_act = [], []

    def collect(driver):
        too_fast, kmh_ratio = [], []
        for n, w in worlds:
            cfg = de.env_config_for(start.meta, rear_sliding_grip_low=grip, rear_sliding_grip_high=grip,
                                    pace_low=args.pace[0], pace_high=args.pace[1], recovery_start_fraction=0.1)
            env = de.DriftVecEnv(w, args.cars, env_config=cfg, seed=int(rng.integers(1 << 30)))
            obs = env.reset()
            for tick in range(args.ticks):
                label, over = teacher(obs, env.counters, env.states[:, fs.S_PACE] > 0.0)
                if tick % args.keep_every == 0:
                    data_obs.append(obs.copy())
                    data_act.append(label)
                too_fast.append(over)
                act = label if driver == 'teacher' else student(obs)
                obs, *_ = env.step(act)
        too_fast = np.concatenate(too_fast)
        return float(np.mean(np.abs(too_fast)))

    def fit():
        X = torch.as_tensor(np.concatenate(data_obs))
        Y = torch.as_tensor(np.concatenate(data_act))
        Xn = torch.clamp((X - mean_t) / std_t, -10, 10)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        n = len(X)
        for epoch in range(args.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for s in range(0, n, 4096):
                idx = perm[s:s + 4096]
                out = net(Xn[idx])
                y = Y[idx]
                loss_sp = ((torch.clamp(out[:, :2], -1, 1) - y[:, :2]) ** 2).mean()
                # the brake: no brake means any output at or below 0, so only push it down if above
                brake_out = out[:, 2]
                loss_b = torch.where(y[:, 2] > 0.0, (torch.clamp(brake_out, -1, 1) - y[:, 2]) ** 2,
                                     torch.relu(brake_out + 0.1) ** 2).mean()
                loss = loss_sp + 2.0 * loss_b
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
        return total / n

    for r in range(args.rounds + 1):
        t0 = time.time()
        miss = collect('teacher' if r == 0 else 'student')
        loss = fit()
        print(f"round {r} ({'teacher' if r == 0 else 'student'} drove): mean miss of the asked pace "
              f"{100 * miss:.1f}%, fit loss {loss:.4f}, {len(data_obs) * args.cars:,} samples, "
              f"{time.time() - t0:.0f} s", flush=True)

    m = MLP([de.OBS_SIZE, hidden, hidden, de.ACT_SIZE])
    lins = [l for l in net if isinstance(l, torch.nn.Linear)]
    m.W = [l.weight.detach().numpy().copy() for l in lins]
    m.b = [l.bias.detach().numpy().copy() for l in lins]
    meta = {k: v for k, v in start.meta.items() if k not in ('eval', 'pace_kmh_range')}
    meta.update({'obs_names': list(de.OBS_NAMES), 'action_names': ['steer_angle', 'pedal', 'brake'],
                 'pace_fraction_range': [float(args.pace[0]), float(args.pace[1])],
                 'pace_reference': 'pace_model 3', 'simulator_revision': fs.SIMULATOR_REVISION,
                 'pace_average_s': de.RewardConfig().pace_average_s,
                 'seeded_by': f"seed_pace.py from {args.policy} (GAIN {GAIN}, THROTTLE_HELD {THROTTLE_HELD})"})
    out = Policy(m, mean, std, meta)
    out.save_json(os.path.join(args.out, 'policy.json'))
    print(f"seeded policy: {os.path.join(args.out, 'policy.json')}")


if __name__ == '__main__':
    main()
