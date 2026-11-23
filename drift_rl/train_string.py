"""
train_string.py - do gap errors grow down a chase train? (string stability, vehicle platooning)

    python train_string.py --track driftplayground_2021 [--policy policy.json] [--cars 48]
                           [--head-share 0 0.5] [--pace 0.95] [--out string_dp.txt]

ChaseBots forms trains: a bot chases a bot that chases a bot (up to ChaseTrainMax). The plugin
moves bots by their own positions (they do not collide unless CarContactPush), so a train can be
simulated one car at a time, each chasing the recorded run of the car ahead:

  1. heads    every car drives alone at --pace (record_ghosts.record): car 1's runs
  2. car 2    every car chases one head run (the gap law of the plugin: ChaseGapMeters 2,
              ChaseTimeGapSeconds 0.25, ChaseResponseSeconds 0.6, ChasePaceMaxPercent 110, the
              lead's plan unknown); its run, on the head's clock, becomes car 3's lead
  3. car 3    every car chases one car-2 run; with --head-share c its gap law takes the train
              head's acceleration as well as its lead's (EnvConfig.chase_head_share, the
              plugin's ChaseHeadFeedforward): (1 - c) x lead + c x head

On the shared clock car 2's and car 3's gap errors (outline gap less the gap law's own aim,
ChaseGapMeters + speed x time gap) and speed swings are compared over the same seconds. A train
is string stable when errors shrink going back: car 3's below car 2's (Ploeg et al. 2014;
Naus et al. 2010). Ratios over 1 = the error grows down the train.
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import record_ghosts as rg
import track_files as tf
from policy import Policy, default_policy_path

# the plugin's defaults (GradientDriftConfiguration.cs)
GAP_M = 2.0
TIME_GAP_S = 0.25
RESPONSE_S = 0.6
PACE_MAX = 1.1
WARMUP_S = 5.0          # after the lead pulls in: the catch-up is not the steady state
EXTRA = ('gap', 'v', 'aim', 'g_src', 'pinned')


def chase_runs(w, pol, ghosts, cars, ticks, pace, seed, head_share=0.0):
    """Every car chases one of ghosts (on their clock); returns (runs as ghosts on that clock,
    per-run extra columns gap / v / aim / source ghost), runs cut where the chase ended."""
    cfg, rw = de.server_run_configs(pol.meta, ticks=ticks + 1, pace=pace)
    ask = (GAP_M - de.CHASE_GAP_M[0]) / (de.CHASE_GAP_M[1] - de.CHASE_GAP_M[0])
    cfg = replace(cfg, chase_share=1.0, chase_scenarios=False, privileged=True,
                  chase_gap_low=ask, chase_gap_high=ask, chase_gap_crowd=(),
                  chase_time_gap_s=TIME_GAP_S, chase_time_gap_blind_s=TIME_GAP_S,
                  chase_response_s=RESPONSE_S, chase_pace_max=PACE_MAX, chase_head_share=head_share)
    w.ghosts = ghosts
    env = de.DriftVecEnv(w, n_envs=cars, env_config=cfg, reward_config=rw, seed=seed)
    obs = env.reset()
    rows = np.full((ticks, cars, de.GH_COLS), np.nan)
    ext = np.full((ticks, cars, len(EXTRA)), np.nan)
    for k in range(ticks):
        dt = env.mids[:, de.M_DT].copy()
        obs, _, _, _, _ = env.step(np.asarray(pol(obs)))
        s, d, c = env.states, env.diags, env.counters
        on = c[:, de.C_GHOST] > 0
        v = d[:, fs.D_ProgressM] / np.maximum(dt, 1e-3)
        r = rows[k]
        r[:, de.GH_T] = c[:, de.C_GHOST_T]
        r[:, de.GH_X] = s[:, fs.S_X]
        r[:, de.GH_Z] = s[:, fs.S_Z]
        r[:, de.GH_HEADING] = s[:, fs.S_HEADING]
        r[:, de.GH_SLIP] = d[:, fs.D_RealSlipAngleRad]
        r[:, de.GH_YAW] = s[:, fs.S_YAW]
        r[:, de.GH_SPEED] = d[:, fs.D_SpeedKmh] / 3.6
        r[:, de.GH_ARC] = s[:, fs.S_ARC]
        r[:, de.GH_CTE] = d[:, fs.D_CrossTrackErrorM]
        r[:, de.GH_STEER] = s[:, fs.S_LAST_STEER]
        r[:, de.GH_PEDAL] = env.prev_act[:, 1]
        r[:, de.GH_BRAKE] = env.prev_act[:, 2]
        ext[k, :, 0] = env.priv[:, 1] * 20.0
        ext[k, :, 1] = v
        ext[k, :, 2] = GAP_M + np.maximum(v, 0.0) * TIME_GAP_S
        ext[k, :, 3] = c[:, de.C_GHOST] - 1
        ext[k, :, 4] = s[:, fs.S_PACE] >= 0.999 * pace * PACE_MAX
        r[~on] = np.nan
        ext[k, ~on] = np.nan
    runs, extra = [], []
    for i in range(cars):
        ok = np.nonzero(~np.isnan(rows[:, i, de.GH_T]))[0]
        if len(ok) < 40:
            continue
        # the first stretch of chasing only (a lost lead ends it)
        stop = ok[0] + np.argmax(np.diff(np.r_[ok, -9]) != 1)
        seg = slice(ok[0], stop + 1)
        runs.append(rows[seg, i])
        extra.append(ext[seg, i])
    return runs, extra


def as_ghosts(runs, extra, keep=0.75):
    """The car-2 runs at least keep x the longest, cut to one length (a ghost set has one length),
    and their extra columns."""
    n = max(len(r) for r in runs)
    idx = [i for i, r in enumerate(runs) if len(r) >= keep * n]
    m = min(len(runs[i]) for i in idx)
    return np.stack([runs[i][:m] for i in idx]), [extra[i][:m] for i in idx]


def add_head(ghosts2, src, heads):
    """Marks each car-2 run (ghosts2[i], on head src[i]'s clock) with its head's acceleration at
    each of its samples (GH_HEAD_ON / GH_HEAD_ACCEL), measured as the plugin and drift_env measure a
    lead's (LEAD_ACCEL_TICKS samples back)."""
    for i in range(len(ghosts2)):
        h = heads[int(src[i])]
        k0 = np.maximum(0, np.arange(len(h)) - de.LEAD_ACCEL_TICKS)
        a = (h[:, de.GH_SPEED] - h[k0, de.GH_SPEED]) / np.maximum(h[:, de.GH_T] - h[k0, de.GH_T], 1e-3)
        ghosts2[i, :, de.GH_HEAD_ON] = 1.0
        ghosts2[i, :, de.GH_HEAD_ACCEL] = np.interp(ghosts2[i, :, de.GH_T], h[:, de.GH_T], a)


def rms(xs):
    a = np.concatenate(xs) if xs else np.zeros(1)
    return float(np.sqrt(np.mean((a - a.mean()) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--policy', default=default_policy_path())
    ap.add_argument('--cars', type=int, default=48)
    ap.add_argument('--pace', type=float, default=0.95, help='the chasers\' pace (their top is this x 1.1)')
    ap.add_argument('--head-pace', type=float, default=None, help='the head\'s pace (default --pace)')
    ap.add_argument('--head-share', type=float, nargs='+', default=[0.0, 0.5])
    ap.add_argument('--seconds', type=float, default=60.0, help='each chasing car\'s run')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    pol = Policy.load_json(args.policy)
    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    n2 = int(args.seconds / 0.05)
    heads = rg.record(w, pol, args.cars, 2 * n2 + 600, 100, args.head_pace or args.pace, 1.0, seed=args.seed)
    runs2, extra2 = chase_runs(w, pol, heads, args.cars, int(1.5 * n2) + 200, args.pace, args.seed + 1)
    g2, extra2 = as_ghosts(runs2, extra2)
    runs2 = list(g2.copy())
    src = np.array([x[0, 3] for x in extra2])
    add_head(g2, src, heads)
    print(f'{args.track}: {len(runs2)} car-2 runs of {g2.shape[1] * 0.05:.0f} s chasing')
    for c in args.head_share:
        runs3, extra3 = chase_runs(w, pol, g2, args.cars, n2, args.pace, args.seed + 2, head_share=c)
        report(c, heads, runs2, extra2, src, runs3, extra3)


def report(c, heads, runs2, extra2, src, runs3, extra3):
    e2, e3, dv12, dv23, gap2, gap3, pin2, pin3 = [], [], [], [], [], [], [], []
    for r3, x3 in zip(runs3, extra3):
        j = int(x3[0, 3])
        r2, x2 = runs2[j], extra2[j]
        h = heads[int(src[j])]
        t3 = r3[:, de.GH_T]
        keep = (t3 >= t3[0] + WARMUP_S) & (t3 >= r2[0, de.GH_T] + WARMUP_S) & (t3 <= r2[-1, de.GH_T])
        if keep.sum() < 40:
            continue
        t = t3[keep]
        e3.append(x3[keep, 0] - x3[keep, 2])
        e2.append(np.interp(t, r2[:, de.GH_T], x2[:, 0] - x2[:, 2]))
        gap3.append(x3[keep, 0])
        gap2.append(np.interp(t, r2[:, de.GH_T], x2[:, 0]))
        v1 = np.interp(t, h[:, de.GH_T], h[:, de.GH_SPEED])
        v2 = np.interp(t, r2[:, de.GH_T], x2[:, 1])
        v3 = x3[keep, 1]
        pin3.append(x3[keep, 4])
        pin2.append(np.interp(t, r2[:, de.GH_T], x2[:, 4]))
        dv12.append(v2 - v1)
        dv23.append(v3 - v2)
    if not e3:
        print(f'head share {c:g}: no overlapping chases')
        return
    cat = np.concatenate
    g2a, g3a = cat(gap2), cat(gap3)
    print(f'head share {c:g}: {len(e3)} trains, {len(cat(e3)) * 0.05:.0f} s')
    print(f'  gap error RMS      car 2 {rms(e2):5.2f} m   car 3 {rms(e3):5.2f} m   ratio {rms(e3) / max(rms(e2), 1e-6):.2f}')
    print(f'  gap mean / p10     car 2 {g2a.mean():5.1f} / {np.percentile(g2a, 10):4.1f} m   '
          f'car 3 {g3a.mean():5.1f} / {np.percentile(g3a, 10):4.1f} m')
    print(f'  speed vs car ahead car 2 {rms(dv12):5.2f}   car 3 {rms(dv23):5.2f} m/s RMS   ratio {rms(dv23) / max(rms(dv12), 1e-6):.2f}')
    e2a, e3a = cat(e2), cat(e3)
    print(f'  over 3 m behind    car 2 {100 * np.mean(e2a > 3):4.1f}%   car 3 {100 * np.mean(e3a > 3):4.1f}%   '
          f'(pace pinned at top: car 2 {100 * np.mean(cat(pin2) > 0.5):.0f}%, car 3 {100 * np.mean(cat(pin3) > 0.5):.0f}%)')
    near = (e2a < 3) & (e3a < 3)
    print(f'  both within 3 m    {100 * np.mean(near):4.1f}% of the time: gap error RMS car 2 {np.std(e2a[near]):4.2f}   '
          f'car 3 {np.std(e3a[near]):4.2f} m   ratio {np.std(e3a[near]) / max(np.std(e2a[near]), 1e-6):.2f}')
    print(f'  touching           car 2 {100 * np.mean(g2a <= 0):4.1f}%   car 3 {100 * np.mean(g3a <= 0):4.1f}%')


if __name__ == '__main__':
    main()
