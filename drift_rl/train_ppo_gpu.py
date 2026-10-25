"""
train_ppo_gpu.py - PPO training with PyTorch: the simulator on the CPU's cores (--sim cpu) or an
NVIDIA GPU, the network on the GPU when there is one. On the GPU thousands of cars are simulated
in parallel by gpu_env.DriftGpuEnv, their observations never leave GPU memory, and the network
trains on them in place.

    python check_gpu.py                                  # once: verifies the GPU path, measures speed
    python train_ppo_gpu.py                              # auto: GPU sim + GPU network if available
    python train_ppo_gpu.py --envs 16384 --minutes 60    # an hour, 16k cars
    python train_ppo_gpu.py --init-policy runs/run1/policy.json      # start from an earlier policy
    python train_ppo_gpu.py --resume runs/<name>/checkpoint.pt
    python train_ppo_gpu.py --sim cpu                    # CPU simulator (all cores), network on GPU if there is one

Ctrl-C stops cleanly: the checkpoint and the policy are saved first.

Outputs, in runs/<name>/ (policy.json is what the plugin, flips.py and dial_table.py read):
    policy.json        best policy so far by evaluation return
    policy_final.json  the policy when training stopped
    checkpoint.pt      everything needed to --resume
    progress.csv       one row per update
    config.json        the settings used
"""
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))

EVAL_KEYS = ['return', 'progress_m', 'mean_abs_slip_deg', 'drift_fraction', 'slide_fraction',
             'high_angle_fraction', 'pushout_tick_fraction',
             'offroad_fraction', 'spin_recovery_fraction', 'mean_speed_kmh', 'mean_abs_cte_m',
             'terminated_early', 'offcourse_fraction', 'marker_hits_per_min',
             'straight_drift_fraction', 'straight_transitions_per_100m', 'mean_chain_s',
             'zone_fill_fraction', 'steer_reversals_per_s', 'front_sliding_fraction',
             'wrong_side_fraction', 'pace_error_pct', 'pace_abs_error_pct', 'brake_fraction', 'mean_brake',
             'weaves_per_min', 'swings_per_min']
TRAIN_KEYS = ['return', 'mean_abs_slip_deg', 'drift_fraction', 'pushout_tick_fraction',
              'offroad_fraction', 'terminated_early']


# The brake output a two-output policy is given when it is widened: no weights in, this bias. The
# plugin and the simulator use max(0, output), so the saved policy never brakes until training
# moves it; the exploration noise (BRAKE_INIT_STD) takes it above zero on about one tick in six.
BRAKE_INIT_BIAS = -0.3
BRAKE_INIT_STD = 0.3


def widen(W, b, n_inputs, hidden, rng, n_outputs=None, new_output_bias=0.0):
    """A policy's layers [(W, b)] grown to n_inputs inputs, hidden units per hidden layer and
    n_outputs outputs without changing what it does: the new inputs get no weight; each new unit
    gets small random weights in (so it computes something and training can use it) and none out
    (so it changes no output until training gives it some); a new output reads nothing and sits at
    new_output_bias."""
    out = []
    n_in = n_inputs
    for layer, (w, bias) in enumerate(zip(W, b)):
        last = layer == len(W) - 1
        rows = (n_outputs or w.shape[0]) if last else hidden
        w2 = np.zeros((rows, n_in))
        w2[:w.shape[0], :w.shape[1]] = w
        b2 = np.zeros(rows)
        b2[:len(bias)] = bias
        if rows > w.shape[0]:
            if last:
                b2[w.shape[0]:] = new_output_bias
            else:
                # new units read everything; the old units' weights from them stay zero
                w2[w.shape[0]:, :] = rng.standard_normal((rows - w.shape[0], n_in)) * (0.5 / np.sqrt(n_in))
        out.append((w2, b2))
        n_in = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=None,
                    help='a track to train on, by its folder name (default: track_files.TRACK); repeat for several '
                         'tracks at once (CPU simulator), each keeping a group of cars the whole run; NAME:N gives '
                         'it N groups')
    ap.add_argument('--generated', type=int, default=0,
                    help='also train on this many generated courses (generate_track.py, made in --tracks the '
                         'first time), the other groups of cars dealt new ones every --swap-updates updates')
    ap.add_argument('--gen-seed', type=int, default=1, help='which set of generated courses')
    ap.add_argument('--eval-generated', type=int, default=4,
                    help='with --generated: generated courses (from a different seed, never trained on) the '
                         'evaluation also runs on')
    ap.add_argument('--group', type=int, default=32, help='cars per track group with several tracks')
    ap.add_argument('--groups-per-track', type=int, default=1,
                    help='with --generated: groups of cars each --track keeps (the rest drive generated courses)')
    ap.add_argument('--swap-updates', type=int, default=10, help='updates between new courses for the pool groups')
    ap.add_argument('--sim', choices=['auto', 'gpu', 'cpu'], default='auto', help='where the cars are simulated')
    ap.add_argument('--device', default='auto', help="where the network trains: auto, cuda or cpu")
    ap.add_argument('--envs', type=int, default=None, help='cars simulated at once (default 4096 on GPU, 256 on CPU)')
    ap.add_argument('--rollout', type=int, default=None, help='ticks per car between updates (default 32 on GPU, 64 on CPU)')
    ap.add_argument('--steps', type=int, default=None, help='total simulated ticks (default 200M on GPU, 10M on CPU)')
    ap.add_argument('--minutes', type=float, default=None, help='stop after this much wall-clock time')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--minibatch-size', type=int, default=None,
                    help='samples per gradient step (default 4096 on GPU, 2048 on CPU)')
    ap.add_argument('--lr', type=float, default=3e-4, help='starting learning rate (then adapted to hold --target-kl)')
    ap.add_argument('--target-kl', type=float, default=0.01)
    ap.add_argument('--gamma', type=float, default=0.997,
                    help='discount: 0.997 looks ~17 s ahead. With 0.99 (~5 s) losing the car costs too '
                         'little next to the extra drift reward, and continued training drifts toward '
                         'crashing more (measured: early endings 9%% -> 40%%)')
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent-coef', type=float, default=0.002)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=None,
                    help='units per hidden layer (two layers); default: the checkpoint\'s with --resume, else 128 '
                         '(a narrower --init-policy is widened)')
    ap.add_argument('--residual', type=int, default=0, metavar='UNITS',
                    help='with --init-policy and --chase: keep the starting network frozen and train a second one '
                         '(two layers of UNITS) added to its action only while a lead is present (residual '
                         'policy learning), so solo driving stays exactly the starting policy\'s')
    ap.add_argument('--init-log-std', type=float, default=None,
                    help='log of the starting exploration noise (default -1.2; -2.3 with --init-policy)')
    ap.add_argument('--reward-from', default=None, metavar='CONFIG',
                    help="start from the reward of an earlier run (its runs/<name>/config.json); --reward still overrides")
    ap.add_argument('--reward', action='append', default=[], metavar='NAME=VALUE',
                    help='override one drift_env.RewardConfig field, e.g. --reward chain_bonus_per_m=0.4 '
                         '(repeatable; the values used are saved in config.json)')
    ap.add_argument('--min-std', type=float, default=0.05,
                    help='floor on the exploration noise (training only; the saved policy is noise-free)')
    ap.add_argument('--min-std-brake', type=float, default=0.15,
                    help='the same for the brake output: with the general floor it collapsed within a few million '
                         'ticks and the policy stopped trying harder braking when asked for a slower pace')
    ap.add_argument('--car-pool', default=None, metavar='JSON',
                    help='train on several cars: a cars/car_numbers.py --json file (rear-drive cars only; each '
                         'gets a welded diff and at least a 50-degree angle kit, drift_env.DRIFT_STEER_LOCK_DEG)')
    ap.add_argument('--car-exclude', action='append', default=[], metavar='CAR',
                    help='leave this car out of training (to test on a car the policy never drove)')
    ap.add_argument('--car-mass-randomisation', type=float, default=0.10, help='+- share on each episode\'s car weight')
    ap.add_argument('--car-power-randomisation', type=float, default=0.20, help='+- share on its engine force')
    ap.add_argument('--chase', type=float, default=0.0, metavar='SHARE',
                    help="share of episodes chasing a ghost leader (each track's <track>_ghosts.npz, record_ghosts.py)")
    ap.add_argument('--chase-gap', type=float, nargs=2, default=(0.0, 1.0), metavar=('LOW', 'HIGH'),
                    help='the asked gap drawn per chase episode, 0 - 1 (door to door ... a few lengths)')
    ap.add_argument('--chase-gap-anneal-ticks', type=int, default=0,
                    help='with --chase: start the asked gaps in the top half of --chase-gap and lower the bottom to '
                         'LOW over this many ticks (0 = the whole range from the start)')
    ap.add_argument('--skill', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                    help='train the skill input (drift_env SKILL_*): each episode a skill drawn from LOW - HIGH '
                         '(0 = a driver allowed out with others, 1 = the best it can do)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-minutes', type=float, default=1.0, help='wall-clock time between evaluations')
    ap.add_argument('--eval-episodes', type=int, default=128)
    ap.add_argument('--name', default=None)
    ap.add_argument('--resume', default=None, help='checkpoint.pt from an earlier train_ppo_gpu.py run')
    ap.add_argument('--init-policy', default=None, help='policy.json to start the driving network from')
    ap.add_argument('--value-warmup-updates', type=int, default=60,
                    help='with --init-policy: updates that train only the value network first')
    ap.add_argument('--grip-randomisation', type=float, default=None)
    ap.add_argument('--sliding-grip', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                    help="draw the plugin's RearSlidingGrip (share of peak sideways grip the "
                         "rear axle keeps while sliding; less = a slower drift) per episode from this range and "
                         "show it to the policy, so the finished policy runs at any setting inside the range. "
                         "Taken from --init-policy / --resume when they have one. Default 1 1 (no loss of grip).")
    ap.add_argument('--pace', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                    help="draw each car's pace (the plugin's bot-spacing dial: a fraction of how fast the course "
                         "is driven flat out, e.g. 0.7 1.05) per episode and now and then mid-run, and show it to "
                         "the policy; it is asked for that share of the course's reference speed at every point, "
                         "paid for no more distance than that covers, and earns less the further above it it runs "
                         "(--reward pace_over_full_cut, pace_under_full_cut, pace_min_pay). Taken from --init-policy / --resume when "
                         "they have one. Default: no pace.")
    ap.add_argument('--pace-anneal-ticks', type=int, default=None,
                    help="with --pace: start the paces between HIGH - 0.1 and HIGH and lower the bottom to LOW over "
                         "this many ticks (default: the first half of the run; 0 = the whole range from the start)")
    ap.add_argument('--pace-reference', default=None, metavar='POLICY',
                    help="no longer used: each course's pace of 1.0 is pace_model's speed round it (made once per "
                         "course and kept beside its track data as <track>_pace_reference.csv, which the plugin reads "
                         "too); an older file, measured by a policy, is replaced and kept as .before_pace_model")
    ap.add_argument('--plugin-setting', action='append', default=[], metavar='NAME=VALUE',
                    help="a plugin setting to train with where it differs from the plugin's default, e.g. "
                         "MaxSteerAngleDeg=60 WrongWayEntryThresholdDeg=85 (repeat the flag). Written into the "
                         "policy file, and the plugin applies it. Kept from --init-policy / --resume unless "
                         "given again. Changing MaxSteerAngleDeg re-gears the starting network so that it "
                         "asks for the same steering angles as before.")
    ap.add_argument('--sliding-grip-anneal-ticks', type=int, default=None,
                    help="with a --sliding-grip range wider than the starting policy's: lower the bottom of the "
                         "range gradually over this many ticks instead of at once (default: the first half of "
                         "this run). A policy trained at full sliding grip spins at 0.9, so it has to be walked down.")
    args = ap.parse_args()
    if args.resume:
        # a continued run keeps the cars, skills and number of cars it was trained with (given again
        # on the command line, these win); without this a resumed any-car run trained on the default
        # car only and its policy file lacked the car inputs' metadata (trialM, 2026-10-20)
        saved = os.path.join(os.path.dirname(os.path.abspath(args.resume)), 'config.json')
        if os.path.exists(saved):
            with open(saved) as f:
                ppo = json.load(f).get('ppo', {})
            if args.car_pool is None and ppo.get('car_pool'):
                args.car_pool = ppo['car_pool']
                if not args.car_exclude:
                    args.car_exclude = list(ppo.get('car_exclude') or [])
            if args.skill is None and ppo.get('skill'):
                args.skill = list(ppo['skill'])
            if args.envs is None and ppo.get('envs'):
                args.envs = ppo['envs']
    if args.hidden is None:
        import torch as _torch
        args.hidden = (_torch.load(args.resume, map_location='cpu', weights_only=True)['hidden']
                       if args.resume else 128)

    sys.path.insert(0, _HERE)
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("PyTorch is not installed (pip install torch; see README.md).")
    import fast_sim as fs
    import drift_env as de
    import pace_model
    import gpu_env
    from policy import MLP, Policy
    from track_files import TRACK

    # ---- where things run ----
    has_cuda = torch.cuda.is_available()
    sim = args.sim
    if sim == 'auto':
        sim = 'cpu'
        if has_cuda:
            try:
                from numba import cuda
                sim = 'gpu' if cuda.is_available() else 'cpu'
            except Exception:
                sim = 'cpu'
        if sim == 'cpu' and has_cuda:
            print("note: PyTorch has a GPU but numba-cuda is not usable, so the simulator stays on the "
                  "CPU. Run check_gpu.py to see why.")
    if sim == 'gpu' and not has_cuda:
        sys.exit("--sim gpu needs PyTorch with CUDA; torch.cuda.is_available() is False. Run check_gpu.py.")
    dev = torch.device(('cuda' if has_cuda else 'cpu') if args.device == 'auto' else args.device)
    n_envs = args.envs or (4096 if sim == 'gpu' else 256)
    total_new_steps = args.steps or (200_000_000 if sim == 'gpu' else 10_000_000)
    torch.manual_seed(args.seed)

    env_cfg = de.EnvConfig()
    if args.grip_randomisation is not None:
        env_cfg.grip_randomisation = args.grip_randomisation
    grip_low_start = 1.0        # bottom of the range the starting network already handles
    if args.init_policy and not args.resume:
        init_meta = Policy.load_json(args.init_policy).meta
        de.require_direct(init_meta)
        grip_low_start = de.sliding_grip_range(init_meta)[0]
        env_cfg.plugin_settings = dict(init_meta.get('plugin_settings') or {})
        if args.sliding_grip is None:
            env_cfg.rear_sliding_grip_low, env_cfg.rear_sliding_grip_high = de.sliding_grip_range(init_meta)
        if args.pace is None:
            env_cfg.pace_low, env_cfg.pace_high = de.pace_range(init_meta)
    if args.resume:
        # the checkpoint holds the networks, not the environment they were trained in - take the
        # grip range and plugin settings from the config.json saved next to it
        saved = os.path.join(os.path.dirname(os.path.abspath(args.resume)), 'config.json')
        if os.path.exists(saved):
            with open(saved) as f:
                saved_env = json.load(f).get('env', {})
            if not saved_env.get('direct_control', True):
                sys.exit(f"{args.resume} is a residual-mode run (it corrects the hand-written controller), "
                         "which this trainer no longer has")
            for key, given in (('rear_sliding_grip_low', args.sliding_grip),
                               ('rear_sliding_grip_high', args.sliding_grip),
                               ('pace_low', args.pace), ('pace_high', args.pace)):
                if given is None and key in saved_env:
                    setattr(env_cfg, key, saved_env[key])
            grip_low_start = float(saved_env.get('rear_sliding_grip_low', 1.0))
            env_cfg.plugin_settings = dict(saved_env.get('plugin_settings') or {})
    for item in args.plugin_setting:
        setting, _, value = item.partition('=')
        if setting not in de.PLUGIN_SETTING_PARAMS:
            sys.exit(f"--plugin-setting: '{setting}' is not one of {', '.join(de.PLUGIN_SETTING_PARAMS)}")
        env_cfg.plugin_settings[setting] = float(value)
        if setting == 'WrongWayEntryTicks' and float(value) != round(float(value)):
            sys.exit("--plugin-setting WrongWayEntryTicks must be a whole number")
    if env_cfg.plugin_settings:
        print("plugin settings trained with (and written into the policy file): "
              + ", ".join(f"{k}={v:g}" for k, v in env_cfg.plugin_settings.items()))
    if args.sliding_grip is not None:
        low, high = args.sliding_grip
        if not 0.0 < low <= high <= 1.0:
            sys.exit("--sliding-grip LOW HIGH: needs 0 < LOW <= HIGH <= 1")
        env_cfg.rear_sliding_grip_low, env_cfg.rear_sliding_grip_high = low, high
    grip_range = (env_cfg.rear_sliding_grip_low, env_cfg.rear_sliding_grip_high)
    if args.pace is not None:
        low, high = args.pace
        if not 0.0 < low <= high <= 1.5:
            sys.exit("--pace LOW HIGH: fractions of the course's flat-out speed, 0 < LOW <= HIGH <= 1.5 (e.g. 0.7 1.05)")
        env_cfg.pace_low, env_cfg.pace_high = low, high
    pace_range = (env_cfg.pace_low, env_cfg.pace_high)
    pedal = env_cfg.plugin_settings.get('EnableRearFrictionCircle', 0.0) == 1.0
    if pedal and not grip_range[1] < 1.0:
        sys.exit("EnableRearFrictionCircle needs --sliding-grip below 1 "
                 "(full throttle leaves the rear RearSlidingGrip x its peak sideways; at 1 there is no thrust)")
    print(f"policy outputs: steering angle, {'pedal and brake (with the throttle)' if pedal else 'target speed (and a brake output the simulator ignores without EnableRearFrictionCircle)'}")
    if pace_range != (0.0, 0.0):
        print(f"pace: {pace_range[0]:g} - {pace_range[1]:g} of each course's flat-out speed, per episode and "
              f"about every {env_cfg.pace_change_s:g} s, shown to the policy (input {de.PACE_OBS_SLOT})")
    if grip_range != (1.0, 1.0):
        print(f"RearSlidingGrip: {grip_range[0]:g} - {grip_range[1]:g} per episode, shown to the policy (input 35)")
    rew_cfg = de.RewardConfig()
    saved = None
    if args.reward_from:
        saved = args.reward_from
        if not os.path.exists(saved):
            sys.exit(f"--reward-from: no {saved}")
    elif args.resume:
        # a continued run keeps the reward it was being trained with (--reward still overrides)
        saved = os.path.join(os.path.dirname(os.path.abspath(args.resume)), 'config.json')
    if saved:
        if os.path.exists(saved):
            with open(saved) as f:
                for key, value in json.load(f).get('reward', {}).items():
                    if hasattr(rew_cfg, key):
                        setattr(rew_cfg, key, value)
    for item in args.reward:
        name, _, value = item.partition('=')
        if not hasattr(rew_cfg, name):
            sys.exit(f"--reward: no RewardConfig field called '{name}'. Fields: {', '.join(asdict(rew_cfg))}")
        setattr(rew_cfg, name, float(value))

    if args.resume and args.track is None:
        # a continued run stays on the track(s) it was trained on
        saved = os.path.join(os.path.dirname(os.path.abspath(args.resume)), 'config.json')
        if os.path.exists(saved):
            with open(saved) as f:
                ppo = json.load(f).get('ppo', {})
            args.track = ppo.get('track')
            if not args.generated and ppo.get('generated'):
                args.generated, args.gen_seed = ppo['generated'], ppo.get('gen_seed', args.gen_seed)
        if args.track:
            print(f"track (from {args.resume}'s config.json): {args.track}")
    if isinstance(args.track, str):
        args.track = [args.track]
    track_groups = {t.partition(':')[0]: int(t.partition(':')[2] or 1) for t in (args.track or [TRACK])}
    tracks = list(track_groups)

    name = args.name or time.strftime('gpu_%Y%m%dT%H%M%S')
    out_dir = os.path.join(_HERE, 'runs', name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading {', '.join(tracks)} from {os.path.abspath(args.tracks)} ...", flush=True)
    worlds = []
    for t in tracks:
        if not os.path.exists(tf.path(args.tracks, 'visual', t)):
            import sensor_view
            sensor_view.make(args.tracks, t)
            print(f"  {t}: the sensors now see what the game collides with and the course's markers "
                  f"(the drawn model kept as {tf.name('visual', t)}; sensor_view.py)", flush=True)
        w = fs.load_world(args.tracks, track=t)
        fs.require_road_mesh(w, args.tracks, t)
        fs.require_course(w, args.tracks, t)
        worlds.append(w)
    world = worlds[0]
    pool, pool_names, eval_worlds = [], [], list(zip(tracks, worlds))
    if args.generated:
        import generate_track as gt
        data = args.tracks
        print(f"generated courses: {args.generated} to train on (seed {args.gen_seed}), {args.eval_generated} "
              f"to evaluate on (seed {args.gen_seed + 1000}) - made the first time, then reused", flush=True)
        # a generated course has nothing past its edges but ground, so its grid needs far less
        # margin than a real track's, and its sparse geometry coarser cells (memory per course)
        gen_world = dict(fine_margin_m=40.0, fine_cell_size=4.0, rm_cell=6.0)
        pool_names = [gt.ensure(data, args.gen_seed, k) for k in range(args.generated)]
        pool = [fs.load_world(args.tracks, track=course, **gen_world) for course in pool_names]
        for k in range(args.eval_generated):
            name = gt.ensure(data, args.gen_seed + 1000, k)
            eval_worlds.append((name, fs.load_world(args.tracks, track=name, **gen_world)))
    if pace_range != (0.0, 0.0):
        # what a pace of 1.0 asks for on each course: pace_model's speed round it
        named = list(zip(tracks, worlds)) + list(zip(pool_names, pool)) + eval_worlds[len(tracks):]
        data = args.tracks
        print(f"pace reference: each course's {tf.PACE_REFERENCE} in its folder in {os.path.abspath(data)}, made by "
              f"{pace_model.MODEL} where missing or older ...", flush=True)
        t_ref = time.time()
        for course, w in named:
            try:
                w.pace_ref = de.course_pace_reference(data, course, w)
            except RuntimeError as e:
                sys.exit(str(e))
        flat = [float(w.sp_length.sum() / np.sum(w.sp_length / (w.pace_ref / 3.6)) * 3.6) for _, w in named[:len(tracks)]]
        print(f"pace reference ready in {time.time() - t_ref:.0f} s; flat out: "
              + ", ".join(f"{c} {v:.1f} km/h" for c, v in zip(tracks, flat)), flush=True)
    if args.chase > 0.0:
        if sim == 'gpu':
            sys.exit("chase episodes run on the CPU simulator only for now: add --sim cpu")
        env_cfg.chase_share = args.chase
        env_cfg.chase_gap_low, env_cfg.chase_gap_high = args.chase_gap
        for t, w in zip(tracks, worlds):
            w.ghosts = de.load_ghosts(args.tracks, t)
            if w.ghosts is None:
                sys.exit(f"--chase: no ghosts for {t}; make them with record_ghosts.py --track {t}")
        print(f"chase: {args.chase:.0%} of episodes behind a ghost leader ({', '.join(f'{t} {len(w.ghosts)}' for t, w in zip(tracks, worlds))}), "
              f"asked gap {args.chase_gap[0]:g} - {args.chase_gap[1]:g}", flush=True)
    if args.skill:
        env_cfg.skill_low, env_cfg.skill_high = args.skill
        print(f"skill: {args.skill[0]:g} - {args.skill[1]:g} per episode, shown to the policy (input {de.SKILL_OBS_SLOT})")
    car_names = []
    if args.car_pool:
        if pace_range == (0.0, 0.0):
            sys.exit("--car-pool needs a policy with paces (each car's pace is scaled to its own flat-out speed)")
        cars = de.load_car_pool(args.car_pool, exclude=set(args.car_exclude))
        if not cars:
            sys.exit(f"no rear-drive cars in {args.car_pool} (after --car-exclude)")
        print(f"cars: {len(cars)} from {args.car_pool}; each one's speed against the reference car ({pace_model.MODEL}) ...", flush=True)
        t_cars = time.time()
        real_worlds = [w for w in worlds]
        scaled = []
        for name, car in cars:
            k = de.car_pace_scale(real_worlds, car)
            scaled.append(dict(car, PACE_SCALE=k))
            car_names.append(name)
            print(f"  {name}: flat out x{k:.3f}", flush=True)
        env_cfg.car_pool = tuple(scaled)
        env_cfg.car_mass_randomisation = args.car_mass_randomisation
        env_cfg.car_power_randomisation = args.car_power_randomisation
        print(f"cars ready in {time.time() - t_cars:.0f} s; weight +-{args.car_mass_randomisation:.0%}, "
              f"engine +-{args.car_power_randomisation:.0%}", flush=True)
    multi = len(worlds) > 1 or bool(pool)
    if multi:
        if sim == 'gpu':
            sys.exit("several tracks at once run on the CPU simulator only: add --sim cpu")
        fixed = [w for t, w in zip(tracks, worlds) for _ in range(track_groups[t] * (args.groups_per_track if pool else 1))]
        env = gpu_env.MultiTrackCpuEnv(fixed, pool, n_envs, args.group, env_cfg, rew_cfg, seed=args.seed)
        print(f"{env.n_groups} groups of {args.group} cars: {len(fixed)} on {', '.join(f'{t} x{track_groups[t]}' if track_groups[t] > 1 else t for t in tracks)}, "
              f"{env.n_groups - len(fixed)} on generated courses" if pool else
              f"{env.n_groups} groups of {args.group} cars on {', '.join(tracks)}", flush=True)
    else:
        env = gpu_env.make_env(world, sim, n_envs, env_cfg, rew_cfg, seed=args.seed)
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else 'none'
    print(f"simulator: {sim.upper()} ({n_envs} cars)   network: {dev.type.upper()}   GPU: {gpu_name}", flush=True)

    # ---- networks ----
    def mlp(out, out_gain, hidden=None):
        hidden = hidden or args.hidden
        layers, last = [], de.OBS_SIZE
        for _ in range(2):
            lin = nn.Linear(last, hidden)
            nn.init.orthogonal_(lin.weight, 2 ** 0.5)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            last = hidden
        lin = nn.Linear(last, out)
        nn.init.orthogonal_(lin.weight, out_gain)
        nn.init.zeros_(lin.bias)
        return nn.Sequential(*layers, lin).to(dev)

    pi = mlp(de.ACT_SIZE, 0.01)            # starts ~0 = the existing controller
    vf = mlp(1, 1.0)
    # A policy loaded with --init-policy already drives well: explore gently around it (std 0.1)
    # rather than with the from-scratch noise, which would undo it within a few updates.
    init_log_std = args.init_log_std if args.init_log_std is not None else (-2.3 if args.init_policy else -1.2)
    log_std = nn.Parameter(torch.full((de.ACT_SIZE,), init_log_std, device=dev))
    if args.init_policy and not args.resume and Policy.load_json(args.init_policy).mlp.W[-1].shape[0] < de.ACT_SIZE:
        with torch.no_grad():
            log_std[2] = float(np.log(BRAKE_INIT_STD))      # a new brake output: see BRAKE_INIT_BIAS
    if args.residual and not (args.init_policy and args.chase > 0.0):
        sys.exit('--residual needs --init-policy (the frozen network) and --chase')
    res = mlp(de.ACT_SIZE, 0.01, args.residual) if args.residual else None
    gate_slot = de.CHASE_OBS_SLOT            # lead_present, raw 0 / 1
    # with a residual the exploration noise stays where it starts: solo ticks give the entropy bonus a
    # free hand to widen it (the frozen network cannot answer), and the chase add-on then trains
    # under ever more noise (std of the brake 0.16 -> 0.35 in 10M ticks, chase_debug1)
    trained = (list(res.parameters()) if res is not None else list(pi.parameters()) + [log_std]) + list(vf.parameters())
    opt = torch.optim.Adam(trained, lr=args.lr, eps=1e-5)

    def actor(x, lead):
        """The action mean; lead: the raw lead_present input (the residual's gate)."""
        mu = pi(x)
        if res is not None:
            mu = mu + (lead > 0.5).to(mu.dtype)[:, None] * res(x)
        return mu

    # running statistics, kept in float64 on the network's device
    f64 = torch.float64
    obs_mean = torch.zeros(de.OBS_SIZE, dtype=f64, device=dev)
    obs_var = torch.ones(de.OBS_SIZE, dtype=f64, device=dev)
    obs_count = 1e-4
    ret_var = torch.ones((), dtype=f64, device=dev)
    ret_mean = torch.zeros((), dtype=f64, device=dev)
    ret_count = 1e-4
    steps_done = 0
    best_eval = -float('inf')
    lr = args.lr
    freeze_obs_norm = False

    def rms_update(mean, var, count, x):
        b_mean, b_var, b_n = x.mean(dim=0), x.var(dim=0, unbiased=False), x.shape[0]
        delta = b_mean - mean
        tot = count + b_n
        new_mean = mean + delta * b_n / tot
        m2 = var * count + b_var * b_n + delta ** 2 * count * b_n / tot
        return new_mean, m2 / tot, tot

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=True)
        if ck['hidden'] != args.hidden:
            sys.exit(f"checkpoint was trained with --hidden {ck['hidden']}; pass the same value.")
        if ck['log_std'].shape[0] != de.ACT_SIZE or ck['obs_mean'].shape[0] != de.OBS_SIZE:
            sys.exit(f"{args.resume} is from a run with {ck['obs_mean'].shape[0]} inputs and {ck['log_std'].shape[0]} "
                     f"outputs; this trainer has {de.OBS_SIZE} and {de.ACT_SIZE}. Start a new run from that run's "
                     "policy.json with --init-policy instead (it is widened, driving exactly as before).")
        pi.load_state_dict(ck['pi'])
        if res is not None and ck.get('res') is not None:
            res.load_state_dict(ck['res'])
        vf.load_state_dict(ck['vf'])
        with torch.no_grad():
            log_std.copy_(ck['log_std'])
        if 'grip_low_now' in ck:
            grip_low_start = float(ck['grip_low_now'])      # stopped part-way down
        opt.load_state_dict(ck['opt'])
        obs_mean, obs_var, obs_count = ck['obs_mean'], ck['obs_var'], ck['obs_count']
        ret_mean, ret_var, ret_count = ck['ret_mean'], ck['ret_var'], ck['ret_count']
        steps_done, lr = ck['steps_done'], ck['lr']
        freeze_obs_norm = ck.get('freeze_obs_norm', False)
        # the best evaluation so far, so a run continued in its own folder does not overwrite a
        # better policy.json with its first evaluation; older checkpoints lack it - then the folder's
        # policy.json says. Resumed into another folder it starts afresh: the old run's best was
        # judged on its own cars and courses (trialO took trialM's one-Miata 208.8 and never wrote
        # policy.json, 2026-10-20)
        best_eval = -float('inf')
        best_path = os.path.join(out_dir, 'policy.json')
        if os.path.samefile(os.path.dirname(os.path.abspath(args.resume)), out_dir):
            best_eval = float(ck.get('best_eval', -float('inf')))
            if best_eval == -float('inf') and os.path.exists(best_path):
                best_eval = float((Policy.load_json(best_path).meta.get('eval') or {}).get('return', -float('inf')))
        print(f"resumed from {args.resume} at {steps_done:,} ticks; best evaluation so far {best_eval:.2f}")
    elif args.init_policy:
        p0 = Policy.load_json(args.init_policy)
        old_hidden = p0.mlp.W[0].shape[0]
        if old_hidden > args.hidden:
            sys.exit(f"{args.init_policy} has {old_hidden}-unit layers; pass --hidden {old_hidden} or more.")
        lins = [m for m in pi if isinstance(m, nn.Linear)]
        n_old = p0.mlp.W[0].shape[1]
        out_old = p0.mlp.W[-1].shape[0]
        weights = widen(p0.mlp.W, p0.mlp.b, de.OBS_SIZE, args.hidden, np.random.default_rng(args.seed),
                        n_outputs=de.ACT_SIZE, new_output_bias=BRAKE_INIT_BIAS)
        # an input whose meaning changed since that policy was trained (its name differs): no
        # weight, as if new (gen4's km/h pace -> the fraction of the course's flat-out speed)
        old_names = list(p0.meta.get('obs_names') or [])
        changed = [j for j in range(min(n_old, len(old_names))) if old_names[j] != de.OBS_NAMES[j]
                   and de.OBS_NAMES[j] in de.NEW_INPUT_NORM]
        for j in changed:
            weights[0][0][:, j] = 0.0
        with torch.no_grad():
            for lin, (w, b) in zip(lins, weights):
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        if old_hidden < args.hidden:
            print(f"{args.init_policy} has {old_hidden}-unit layers: widened to {args.hidden}, the new units "
                  "starting with no say in the output (it drives exactly as before)")
        new_norm = [de.NEW_INPUT_NORM.get(name, (0.6, 0.5)) for name in de.OBS_NAMES[n_old:]]
        old_mean, old_std = np.array(p0.obs_mean, dtype=float), np.array(p0.obs_std, dtype=float)
        for j in changed:
            old_mean[j], old_std[j] = de.NEW_INPUT_NORM[de.OBS_NAMES[j]]
            print(f"{args.init_policy}: input {j} was '{old_names[j]}', now '{de.OBS_NAMES[j]}': it starts with no weight")
        obs_mean = torch.as_tensor(np.r_[old_mean, [m for m, _ in new_norm]], dtype=f64, device=dev)
        obs_var = torch.as_tensor(np.r_[old_std, [sd for _, sd in new_norm]], dtype=f64, device=dev) ** 2
        if n_old < de.OBS_SIZE:
            print(f"{args.init_policy} reads {n_old} inputs: inputs {n_old} - {de.OBS_SIZE - 1} "
                  f"({', '.join(de.OBS_NAMES[n_old:])}) start with no weight")
        if out_old < de.ACT_SIZE:
            print(f"{args.init_policy} has {out_old} outputs: the brake output starts at never braking "
                  f"(bias {BRAKE_INIT_BIAS:g}, exploration noise {BRAKE_INIT_STD:g})")
        # The loaded network expects exactly this scaling of its inputs, so it is frozen for the
        # whole run (letting it re-estimate from the new, better-driving data rescales every
        # input under the network and steadily degrades it - measured, not hypothetical).
        freeze_obs_norm = True
        if grip_range != (1.0, 1.0) and de.sliding_grip_range(p0.meta) == (1.0, 1.0):
            # Input 35 was always 0 while this network was trained, so its weights on that input
            # are whatever they were initialised to. Zero them: the network then drives exactly as
            # before at every setting until training gives the input a meaning.
            with torch.no_grad():
                lins[0].weight[:, de.SLIDING_GRIP_OBS_SLOT] = 0.0
            obs_mean[de.SLIDING_GRIP_OBS_SLOT] = 0.0
            obs_var[de.SLIDING_GRIP_OBS_SLOT] = 1.0
            print(f"{args.init_policy} was trained without RearSlidingGrip: its weights on input "
                  f"{de.SLIDING_GRIP_OBS_SLOT} start at zero")
        old_lock = de.max_steer_deg(p0.meta)
        new_lock = float(env_cfg.plugin_settings.get('MaxSteerAngleDeg', de.DEFAULT_MAX_STEER_DEG))
        if abs(new_lock - old_lock) > 1e-9:
            # The steering action is a fraction of full lock. Scale the network's steering output,
            # and the two inputs that carry steering as a fraction of lock, by old / new lock: it
            # then asks for the same steering ANGLES as before - and, where it used to ask for
            # more than the old lock and was cut off there, now gets it.
            k = old_lock / new_lock
            with torch.no_grad():
                lins[-1].weight[0, :] *= k
                lins[-1].bias[0] *= k
            for name in ('prev_action_steer', 'last_steer/max_steer'):
                slot = de.OBS_NAMES.index(name)
                obs_mean[slot] *= k
                obs_var[slot] *= k * k
            print(f"steering lock {old_lock:g} -> {new_lock:g} deg: starting network re-geared (x{k:.4f})")
        print(f"driving network initialised from {args.init_policy} (value network starts fresh)")

    meta = {
        'track': tracks[0],
        'obs_names': list(de.OBS_NAMES),
        'action_names': ['steer_angle', 'pedal' if pedal else 'target_speed', 'brake'],
        'control_mode': 'direct',
        'lookahead_m': list(de.LOOKAHEAD_M),
        'min_radius_window_m': de.MIN_RADIUS_WINDOW_M,
        'simulator_revision': fs.SIMULATOR_REVISION,
    }
    if multi:
        meta['tracks'] = list(tracks)
        if pool:
            meta['generated'] = {'count': args.generated, 'seed': args.gen_seed}
    if args.chase > 0.0:
        # the policy reads the chase inputs: it can follow a car ahead
        meta['chase'] = {'share': args.chase, 'gap_range': [float(g) for g in args.chase_gap]}
    if args.skill:
        # the plugin reads this: the skills it may ask the policy for
        meta['skill_range'] = [float(args.skill[0]), float(args.skill[1])]
    if car_names:
        # the plugin reads this: the policy reads the car inputs (drift_env.car_inputs) and drives any car
        meta['cars'] = {'trained': car_names, 'held_out': list(args.car_exclude),
                        'drift_steer_lock_deg': de.DRIFT_STEER_LOCK_DEG,
                        'mass_randomisation': args.car_mass_randomisation,
                        'power_randomisation': args.car_power_randomisation}
    if env_cfg.plugin_settings:
        # the plugin reads this too: it applies these over its own configuration
        meta['plugin_settings'] = dict(env_cfg.plugin_settings)
    if grip_range != (1.0, 1.0):
        # the plugin reads this: it runs the policy only at a RearSlidingGrip inside the range
        meta['rear_sliding_grip_range'] = [grip_range[0], grip_range[1]]
        meta['obs_names'][de.SLIDING_GRIP_OBS_SLOT] = 'rear_grip_lost_sliding/0.5'
    if pace_range != (0.0, 0.0):
        # the plugin reads this: the paces it may ask the policy for (bot spacing), as fractions
        meta['pace_fraction_range'] = [pace_range[0], pace_range[1]]
        meta['pace_reference'] = pace_model.MODEL
        # the plugin reads this: how long the pace error input averages over
        meta['pace_average_s'] = rew_cfg.pace_average_s
    meta['sensor_view'] = 'solid'      # sensor_view.py: walls the game collides with, and the markers
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'env': asdict(env_cfg), 'reward': asdict(rew_cfg), 'ppo': vars(args),
                   'plugin_config_overrides': dict(fs.PLUGIN_CONFIG_OVERRIDES),
                   'sim': sim, 'network_device': dev.type, 'n_envs': n_envs, 'gpu': gpu_name}, f, indent=2)

    def make_policy():
        lins = [m for m in pi if isinstance(m, nn.Linear)]
        m = MLP([de.OBS_SIZE, args.hidden, args.hidden, de.ACT_SIZE])
        m.W = [l.weight.detach().double().cpu().numpy().copy() for l in lins]
        m.b = [l.bias.detach().double().cpu().numpy().copy() for l in lins]
        r = None
        if res is not None:
            rl = [l for l in res if isinstance(l, nn.Linear)]
            r = MLP([de.OBS_SIZE, args.residual, args.residual, de.ACT_SIZE])
            r.W = [l.weight.detach().double().cpu().numpy().copy() for l in rl]
            r.b = [l.bias.detach().double().cpu().numpy().copy() for l in rl]
        return Policy(m, obs_mean.cpu().numpy().copy(), obs_var.sqrt().cpu().numpy() + 1e-8,
                      dict(meta, trained_ticks=steps_done), r, gate_slot, 0.5)

    def save_checkpoint():
        torch.save({'pi': pi.state_dict(), 'vf': vf.state_dict(), 'log_std': log_std.detach(),
                    'res': res.state_dict() if res is not None else None,
                    'opt': opt.state_dict(), 'obs_mean': obs_mean, 'obs_var': obs_var,
                    'obs_count': obs_count, 'ret_mean': ret_mean, 'ret_var': ret_var,
                    'ret_count': ret_count, 'steps_done': steps_done, 'lr': lr, 'hidden': args.hidden,
                    'freeze_obs_norm': freeze_obs_norm, 'grip_low_now': grip_low_now(), 'best_eval': best_eval},
                   os.path.join(out_dir, 'checkpoint.pt'))

    N, T = n_envs, (args.rollout or (32 if sim == 'gpu' else 64))
    f32 = torch.float32
    obs_buf = torch.zeros((T, N, de.OBS_SIZE), dtype=f32, device=dev)
    act_buf = torch.zeros((T, N, de.ACT_SIZE), dtype=f32, device=dev)
    lead_buf = torch.zeros((T, N), dtype=f32, device=dev)
    logp_buf = torch.zeros((T, N), dtype=f32, device=dev)
    rew_buf = torch.zeros((T, N), dtype=f32, device=dev)
    val_buf = torch.zeros((T + 1, N), dtype=f32, device=dev)
    done_buf = torch.zeros((T, N), dtype=f32, device=dev)
    running_ret = torch.zeros(N, dtype=f64, device=dev)
    LOG_2PI = float(np.log(2.0 * np.pi))

    def norm_obs(o):
        return torch.clamp((o - obs_mean) / (obs_var.sqrt() + 1e-8), -10.0, 10.0).to(f32)

    def logp_of(a, mu, ls):
        return (-0.5 * ((a - mu) / ls.exp()) ** 2 - ls - 0.5 * LOG_2PI).sum(dim=1)

    fields = (['update', 'ticks', 'minutes', 'ticks_per_s', 'lr', 'kl', 'clip_fraction', 'std_steer',
               'std_speed', 'std_brake', 'explained_var'] + ['train_' + k for k in TRAIN_KEYS]
              + ['eval_' + k for k in EVAL_KEYS])
    csv_path = os.path.join(out_dir, 'progress.csv')
    new_csv = not os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore', restval='')
    if new_csv:
        writer.writeheader()

    # The bottom of the RearSlidingGrip range the TRAINING cars draw from, walked down from what
    # the starting network handles to the target. Evaluation always uses the full target range,
    # so "best policy" means best over everything the finished policy will be allowed to run at.
    grip_anneal_from = steps_done
    grip_anneal_ticks = 0
    if grip_low_start > grip_range[0]:
        grip_anneal_ticks = (args.sliding_grip_anneal_ticks if args.sliding_grip_anneal_ticks is not None
                             else total_new_steps // 2)
        print(f"RearSlidingGrip: bottom of the training range goes {grip_low_start:g} -> {grip_range[0]:g} "
              f"over the first {grip_anneal_ticks:,} ticks", flush=True)

    pace_anneal_ticks = 0
    pace_low_start = pace_range[0]
    if pace_range != (0.0, 0.0):
        pace_anneal_ticks = (args.pace_anneal_ticks if args.pace_anneal_ticks is not None else total_new_steps // 2)
        pace_low_start = max(pace_range[0], pace_range[1] - 0.1)
        if pace_anneal_ticks > 0 and pace_low_start > pace_range[0]:
            print(f"pace: the slowest pace asked goes {pace_low_start:g} -> {pace_range[0]:g} over the first "
                  f"{pace_anneal_ticks:,} ticks", flush=True)

    def pace_low_now():
        if pace_anneal_ticks <= 0 or pace_low_start <= pace_range[0]:
            return pace_range[0]
        f = min(1.0, max(0.0, (steps_done - grip_anneal_from) / pace_anneal_ticks))
        return pace_low_start + (pace_range[0] - pace_low_start) * f

    def grip_low_now():
        if grip_anneal_ticks <= 0 or grip_low_start <= grip_range[0]:
            return grip_range[0]
        f = min(1.0, max(0.0, (steps_done - grip_anneal_from) / grip_anneal_ticks))
        return grip_low_start + (grip_range[0] - grip_low_start) * f

    chase_low_start = 0.5 * (args.chase_gap[0] + args.chase_gap[1])
    if args.chase > 0.0 and args.chase_gap_anneal_ticks > 0:
        print(f"chase: the closest gap asked goes {chase_low_start:g} -> {args.chase_gap[0]:g} over the first "
              f"{args.chase_gap_anneal_ticks:,} ticks", flush=True)

    def chase_gap_low_now():
        if args.chase_gap_anneal_ticks <= 0:
            return args.chase_gap[0]
        f = min(1.0, max(0.0, (steps_done - grip_anneal_from) / args.chase_gap_anneal_ticks))
        return chase_low_start + (args.chase_gap[0] - chase_low_start) * f

    env.set_sliding_grip_low(grip_low_now())
    if pace_range != (0.0, 0.0):
        env.set_pace_low(pace_low_now())
    if args.chase > 0.0:
        env.set_chase_gap_low(chase_gap_low_now())
    obs = env.reset().to(dev)
    if not freeze_obs_norm:
        obs_mean, obs_var, obs_count = rms_update(obs_mean, obs_var, obs_count, obs)
    target_steps = steps_done + total_new_steps
    t_start = time.time()
    t_last_eval = t_start
    steps_at_start = steps_done
    last_eval = None
    recent = []
    update = 0
    interrupted = False
    # --init-policy gives a good driver but a blank value network. Advantages computed from a
    # blank value network are noise, so for the first few updates only the value network learns.
    warmup_updates = args.value_warmup_updates if (args.init_policy and not args.resume) else 0

    per_track = {}

    last_chase = {}

    def evaluate_now():
        nonlocal best_eval, last_eval
        policy = make_policy()
        episodes = max(32, args.eval_episodes // len(eval_worlds))
        for name, w in eval_worlds:
            ev = de.evaluate(w, policy, episodes, seed=999, env_config=env_cfg, reward_config=rew_cfg)
            per_track[name] = {k: float(ev[k].mean()) for k in EVAL_KEYS}
        last_eval = {k: float(np.mean([v[k] for v in per_track.values()])) for k in EVAL_KEYS}
        if args.chase > 0.0:
            # chase episodes only, at the whole --chase-gap range (what chase_report.py shows)
            name, w = eval_worlds[0]
            ev = de.evaluate(w, policy, episodes, seed=998, env_config=replace(env_cfg, chase_share=1.0),
                             reward_config=rew_cfg)
            had = ev['chase_fraction'] > 0
            wt = ev['chase_fraction'][had] * ev['length'][had]
            cmean = lambda k: float(np.sum(ev[k][had] * wt) / max(np.sum(wt), 1e-9))
            last_chase.update({'return': float(np.mean(ev['return'][had])) if had.any() else float('nan'),
                               'gap': cmean('chase_mean_gap_m'), 'in_band': cmean('chase_in_band_fraction'),
                               'hits': float(np.mean(ev['end_reason'][had] == 6)) if had.any() else 0.0,
                               'slip': cmean('chase_slip_diff_deg'), 'line': cmean('chase_line_diff_m')})
        if last_eval['return'] > best_eval:
            best_eval = last_eval['return']
            policy.meta['eval'] = last_eval
            policy.save_json(os.path.join(out_dir, 'policy.json'))
        save_checkpoint()

    try:
        while steps_done < target_steps:
            if args.minutes is not None and (time.time() - t_start) / 60.0 >= args.minutes:
                break
            update += 1
            ep_rows = []
            if multi and pool and args.swap_updates and update % args.swap_updates == 0:
                o, restarted = env.rotate()
                obs = o.to(dev)
                running_ret[restarted.to(dev)] = 0.0
            env.set_sliding_grip_low(grip_low_now())
            if pace_range != (0.0, 0.0):
                env.set_pace_low(pace_low_now())
            if args.chase > 0.0:
                env.set_chase_gap_low(chase_gap_low_now())
            with torch.no_grad():
                for t in range(T):
                    x = norm_obs(obs)
                    lead_buf[t] = obs[:, gate_slot]
                    mu = actor(x, lead_buf[t])
                    a = mu + log_std.exp() * torch.randn_like(mu)
                    obs_buf[t], act_buf[t] = x, a
                    logp_buf[t] = logp_of(a, mu, log_std)
                    val_buf[t] = vf(x)[:, 0]
                    o, r, term, trunc = env.step(a.to(env.device, f64))
                    obs = o.to(dev)
                    r = r.to(dev)
                    term, trunc = term.to(dev), trunc.to(dev)
                    done = term | trunc
                    # reward scaling by the running std of the discounted return
                    running_ret = running_ret * args.gamma + r
                    ret_mean, ret_var, ret_count = rms_update(ret_mean, ret_var, ret_count, running_ret)
                    r = (r / (ret_var.sqrt() + 1e-8)).to(f32)
                    if bool(done.any()):
                        if bool(trunc.any()):
                            # a time-limit cut-off is not a real ending: bootstrap from the value
                            # of the observation the episode was cut off at
                            fo = norm_obs(env.final_obs.to(dev)[trunc])
                            r[trunc] += args.gamma * vf(fo)[:, 0]
                        running_ret[done] = 0.0
                        if len(ep_rows) < 64:
                            ep_rows.append(env.ep_out[done.to(env.device)].cpu().numpy())
                    rew_buf[t], done_buf[t] = r, done.to(f32)
                    if not freeze_obs_norm:
                        obs_mean, obs_var, obs_count = rms_update(obs_mean, obs_var, obs_count, obs)
                val_buf[T] = vf(norm_obs(obs))[:, 0]
                steps_done += N * T

                adv = torch.zeros((T, N), dtype=f32, device=dev)
                last = torch.zeros(N, dtype=f32, device=dev)
                for t in range(T - 1, -1, -1):
                    nonterm = 1.0 - done_buf[t]
                    delta = rew_buf[t] + args.gamma * val_buf[t + 1] * nonterm - val_buf[t]
                    last = delta + args.gamma * args.lam * nonterm * last
                    adv[t] = last
                ret = adv + val_buf[:T]

            b_obs = obs_buf.reshape(-1, de.OBS_SIZE)
            b_lead = lead_buf.reshape(-1)
            b_act = act_buf.reshape(-1, de.ACT_SIZE)
            b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            n = b_obs.shape[0]
            # Fixed minibatch SIZE, not a fixed number of minibatches: with thousands of cars one
            # update holds 100k+ samples, and learning speed per simulated tick depends on how
            # many gradient steps those samples get (measured: 8 big minibatches per update
            # learned several times slower per tick than ~32 of this size).
            mb = min(n, args.minibatch_size or (4096 if sim == 'gpu' else 2048))
            kls, cfs = [], []
            for g in opt.param_groups:
                g['lr'] = lr
            stop = False
            for epoch in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n - mb + 1, mb):
                    idx = perm[s:s + mb]
                    a_mb = b_adv[idx]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    mu = actor(b_obs[idx], b_lead[idx])
                    logp = logp_of(b_act[idx], mu, log_std)
                    log_ratio = logp - b_logp[idx]
                    ratio = log_ratio.exp()
                    pi_loss = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_mb).mean()
                    entropy = (log_std + 0.5 * (LOG_2PI + 1.0)).sum()
                    v_loss = 0.5 * ((vf(b_obs[idx])[:, 0] - b_ret[idx]) ** 2).mean()
                    if update <= warmup_updates:
                        loss = args.vf_coef * v_loss
                    else:
                        loss = pi_loss - args.ent_coef * entropy + args.vf_coef * v_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(trained, 0.5)
                    opt.step()
                    with torch.no_grad():
                        log_std.clamp_(float(np.log(args.min_std)), 0.5)
                        log_std[2].clamp_(min=float(np.log(args.min_std_brake)))
                        kl = float(((ratio - 1.0) - log_ratio).mean())
                        cfs.append(float(((ratio - 1.0).abs() > args.clip).float().mean()))
                    kls.append(kl)
                    if kl > 3.0 * args.target_kl and update > warmup_updates:   # batch used up; stop early
                        stop = True
                        break
                if stop:
                    break
            # hold the policy change per update near --target-kl by adapting the learning rate
            mean_kl = float(np.mean(kls))
            if update <= warmup_updates:
                pass
            elif mean_kl > 2.0 * args.target_kl:
                lr = max(lr / 1.5, 1e-5)
            elif mean_kl < 0.5 * args.target_kl:
                lr = min(lr * 1.5, 1e-3)

            if ep_rows:
                recent.append(de.finalize_episode_stats(np.concatenate(ep_rows)))
                recent = recent[-10:]
            now = time.time()
            row = {'update': update, 'ticks': steps_done, 'minutes': (now - t_start) / 60.0,
                   'ticks_per_s': (steps_done - steps_at_start) / (now - t_start), 'lr': lr,
                   'kl': mean_kl, 'clip_fraction': float(np.mean(cfs)),
                   'std_steer': float(log_std.detach()[0].exp()), 'std_speed': float(log_std.detach()[1].exp()),
                   'std_brake': float(log_std.detach()[2].exp()),
                   'explained_var': float(1.0 - (b_ret - val_buf[:T].reshape(-1)).var() / (b_ret.var() + 1e-12))}
            if recent:
                for k in TRAIN_KEYS:
                    row['train_' + k] = float(np.mean(np.concatenate([e[k] for e in recent])))

            if (now - t_last_eval) / 60.0 >= args.eval_minutes or steps_done >= target_steps:
                evaluate_now()
                t_last_eval = time.time()
                print(f"[{steps_done:>13,} ticks {row['minutes']:6.1f} min {row['ticks_per_s']:9,.0f}/s] "
                      f"eval return {last_eval['return']:8.2f}  "
                      f"drift {last_eval['drift_fraction']:.2f}  slide {last_eval['slide_fraction']:.2f}  "
                      f"40deg+ {last_eval['high_angle_fraction']:.2f}  slip {last_eval['mean_abs_slip_deg']:.1f}deg  "
                      f"wall {last_eval['pushout_tick_fraction']:.3f}  off {last_eval['offroad_fraction']:.3f}  "
                      f"off-course {last_eval['offcourse_fraction']:.3f}  markers/min {last_eval['marker_hits_per_min']:.1f}  "
                      f"straight-drift {last_eval['straight_drift_fraction']:.2f}  "
                      f"zones {last_eval['zone_fill_fraction']:.2f}  steer-reversals/s {last_eval['steer_reversals_per_s']:.1f}  "
                      f"weaves/min {last_eval['weaves_per_min']:.1f}  swings/min {last_eval['swings_per_min']:.1f}  "
                      f"front-past-free {last_eval['front_sliding_fraction']:.3f}  wrong-side {last_eval['wrong_side_fraction']:.3f}  "
                      f"early-end {last_eval['terminated_early']:.2f}  "
                      f"braking {last_eval['brake_fraction']:.2f}"
                      + (f"  pace-error {last_eval['pace_abs_error_pct']:.1f}% (signed {last_eval['pace_error_pct']:+.1f}%)"
                         if pace_range != (0.0, 0.0) else "")
                      + (f"  | chase return {last_chase['return']:.1f} gap {last_chase['gap']:.1f}m "
                         f"in-band {last_chase['in_band']:.0%} hits {last_chase['hits']:.1%} "
                         f"slip-diff {last_chase['slip']:.1f} line-diff {last_chase['line']:.2f} |" if last_chase else "")
                      + f"  kl {mean_kl:.4f}  lr {lr:.1e}  "
                      f"std {row['std_steer']:.2f}/{row['std_speed']:.2f}/{row['std_brake']:.2f}"
                      + (f"  | training cars: grip {grip_low_now():.2f}-{grip_range[1]:.2f}"
                         + (f" pace {pace_low_now():.2f}-{pace_range[1]:.2f}" if pace_range != (0.0, 0.0) else "")
                         + f"  drift {row.get('train_drift_fraction', float('nan')):.2f}"
                         f"  early-end {row.get('train_terminated_early', float('nan')):.2f}"
                         if grip_range != (1.0, 1.0) else ""), flush=True)
                if len(per_track) > 1:
                    print("    per track: " + "   ".join(
                        f"{name} slide {v['slide_fraction']:.2f} slip {v['mean_abs_slip_deg']:.0f} "
                        f"wall {v['pushout_tick_fraction']:.3f} early-end {v['terminated_early']:.2f}"
                        for name, v in per_track.items()), flush=True)
            if last_eval:
                row.update({'eval_' + k: v for k, v in last_eval.items()})
            writer.writerow(row)
            csv_file.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\nstopping (Ctrl-C) - saving ...", flush=True)

    csv_file.close()
    evaluate_now()
    final = make_policy()
    final.meta['eval'] = last_eval
    final.save_json(os.path.join(out_dir, 'policy_final.json'))
    mins = (time.time() - t_start) / 60.0
    print(f"\n{'stopped' if interrupted else 'done'}: {steps_done:,} ticks total, {mins:.1f} min this run "
          f"({(steps_done - steps_at_start) / max(mins * 60, 1e-9):,.0f} ticks/s). Best evaluation return "
          f"{best_eval:.2f}.\nPolicy: {os.path.join(out_dir, 'policy.json')}")


if __name__ == '__main__':
    main()
