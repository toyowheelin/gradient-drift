"""
gap_traces.py - why a chaser sits where it does behind the lead: per-tick traces of chase episodes,
each stretch sorted into what the chaser was doing, and a plot of a few runs.

    python gap_traces.py policy.json --track driftplayground_2021 [--cars 256] [--seconds 120]
                         [--lead-speed 0.95 1 1.05] [--out traces_dp.npz] [--plot traces_dp.png]

Every car chases a ghost (asked gaps over --gaps). Ticks count from 3 s after the lead pulls in (the
catch-up is not the steady state). Each such tick is put in one bucket:

  touching       the cars overlap
  close          gap under the gap paid for less 1 m (inside the ask: too close)
  in band        within 1 m of the gap paid for (the ask, opened through the lead's transitions)
  near           1 - 3 m behind the gap paid for
  far, pinned    over 3 m behind with the chase pace at its top: it is asked to go as fast as it may
                 and still is not closing (the car, the pace range or the lead's speed: "can't")
  far, held      over 3 m behind with the chase pace below its top: the gap law itself asks it to
                 wait (its time gap and closing-speed terms), and the policy obeys ("won't")

Per episode (lead present 10 s or more): the gap's spread (std), how often it crosses the gap paid
for (crossings a minute; an episode oscillates with a spread of 2.5 m+ and 4+ crossings a minute), whether it ever got within 1 m and then fell 4 m+ back (a
"fall-back"), and whether it closed from 3 m+ behind to 1.5 m+ inside within 2 s (an overshoot).
Also: gap minus the IDM's own aim (aim + speed x time gap), to see which of the two the car tracks;
and steering / pedal / brake spread by gap bin (does the policy get erratic close up?).
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy

WARMUP_S = 3.0


def record(world, pol, cfg, cars, seconds, seed):
    env = de.DriftVecEnv(world, cars, replace(cfg, privileged=True), seed=seed)
    obs = env.reset()
    ticks = int(seconds / 0.05)
    k0 = de.CHASE_OBS_SLOT
    names = ('chasing', 'gap', 'target', 'ask', 'speed', 'lead_speed', 'pace', 'pace_top', 'asked_speed',
             'rel_heading', 'lat', 'lon', 'slip', 'lead_slip', 'steer', 'pedal', 'brake', 'contact', 'plan',
             'ghost', 'dt', 'ep_id')
    tr = {k: np.zeros((ticks, cars), dtype=np.float32) for k in names}
    ep_id = np.arange(cars, dtype=np.float64)
    next_id = cars
    top = cfg.pace_high * cfg.chase_pace_max
    for t in range(ticks):
        o = obs.copy()
        act = np.asarray(pol(obs), dtype=np.float64)
        obs, _, term, trunc, _ = env.step(act)
        done = term | trunc
        tr['chasing'][t] = env.priv[:, 0] > 0.5
        tr['gap'][t] = env.priv[:, 1] * 20.0
        tr['target'][t] = tr['gap'][t] - env.priv[:, 2] * 5.0
        tr['ask'][t] = de.CHASE_GAP_M[0] + env.counters[:, de.C_CHASE_ASK] * (de.CHASE_GAP_M[1] - de.CHASE_GAP_M[0])
        tr['speed'][t] = env.diags[:, de.D_ProgressM] / np.maximum(env.mids[:, fs.M_DT], 1e-3)
        tr['lead_speed'][t] = env.priv[:, 4] * 25.0
        tr['pace'][t] = env.states[:, fs.S_PACE]
        tr['pace_top'][t] = top
        tr['rel_heading'][t] = np.degrees(o[:, k0 + 6] * np.pi)
        tr['lon'][t] = o[:, k0 + 3] * 20.0
        tr['lat'][t] = o[:, k0 + 4] * 20.0
        tr['slip'][t] = np.degrees(env.diags[:, fs.D_RealSlipAngleRad])
        tr['lead_slip'][t] = env.priv[:, 5] * 90.0
        tr['steer'][t] = act[:, 0]
        tr['pedal'][t] = act[:, 1]
        tr['brake'][t] = np.clip(act[:, 2], 0.0, 1.0)
        tr['contact'][t] = env.counters[:, de.C_CONTACT] > 0
        tr['plan'][t] = env.counters[:, de.C_CHASE_PLAN] > 0
        tr['ghost'][t] = env.chase_ghost
        tr['dt'][t] = env.mids[:, fs.M_DT]
        tr['ep_id'][t] = ep_id
        k = np.nonzero(done)[0]
        ep_id[k] = next_id + np.arange(len(k))
        next_id += len(k)
        tr['ep_id'][t, k] = ep_id[k]
    return tr


def segments(tr):
    """(car, start, end) of each unbroken chase stretch, start moved WARMUP_S past the lead pulling in."""
    out = []
    T, n = tr['gap'].shape
    for c in range(n):
        ch = tr['chasing'][:, c] > 0.5
        ep = tr['ep_id'][:, c]
        t = 0
        while t < T:
            if not ch[t]:
                t += 1
                continue
            u = t
            while u < T and ch[u] and ep[u] == ep[t]:
                u += 1
            dt = float(np.median(tr['dt'][t:u, c]))
            s = t + int(WARMUP_S / dt)
            if u - s > 5:
                out.append((c, s, u))
            t = u
    return out


def buckets(tr, segs, aim_time_gap):
    names = ('touching', 'close', 'in band', 'near', 'far, pinned', 'far, held')
    counts = dict.fromkeys(names, 0)
    per_ep = []
    rows = []
    for c, s, u in segs:
        g = tr['gap'][s:u, c]
        tg = tr['target'][s:u, c]
        off = g - tg
        touch = tr['contact'][s:u, c] > 0.5
        pinned = tr['pace'][s:u, c] >= tr['pace_top'][s:u, c] - 1e-3
        b = np.full(len(g), 'in band', dtype=object)
        b[off < -1.0] = 'close'
        b[(off > 1.0) & (off <= 3.0)] = 'near'
        far = off > 3.0
        b[far & pinned] = 'far, pinned'
        b[far & ~pinned] = 'far, held'
        b[touch] = 'touching'
        for k in names:
            counts[k] += int(np.sum(b == k))
        dt = float(np.median(tr['dt'][s:u, c]))
        dur = (u - s) * dt
        idm_aim = tr['ask'][s:u, c] + tr['speed'][s:u, c] * np.where(tr['plan'][s:u, c] > 0.5, aim_time_gap[0], aim_time_gap[1])
        rows.append(np.stack([off, g - idm_aim, tr['steer'][s:u, c], tr['pedal'][s:u, c], tr['brake'][s:u, c],
                              pinned.astype(np.float32), tr['speed'][s:u, c] - tr['lead_speed'][s:u, c]], 1))
        if dur >= 10.0:
            cross = int(np.sum(np.diff(np.sign(off)) != 0))
            near_then_back = False
            been_close = np.nonzero(np.abs(off) < 1.0)[0]
            if len(been_close):
                near_then_back = bool(np.any(off[been_close[0]:] > 4.0))
            w = int(2.0 / dt)
            over = False
            for i in np.nonzero(off > 3.0)[0][::4]:
                if np.any(off[i:i + w] < -1.5):
                    over = True
                    break
            per_ep.append((np.std(off), cross / (dur / 60.0), near_then_back, over, bool(touch.any()),
                           float(np.median(off)), float(np.mean(pinned)), int(tr['ghost'][s, c])))
    return counts, np.array(per_ep, dtype=np.float64).reshape(-1, 8), np.concatenate(rows) if rows else np.zeros((0, 7))


def report(name, tr, segs, aim_time_gap, kinds):
    counts, E, R = buckets(tr, segs, aim_time_gap)
    tot = max(sum(counts.values()), 1)
    print(f'\n== {name}: {len(segs)} chase stretches, {tot} ticks after the first {WARMUP_S:g} s')
    print('  ' + '  '.join(f'{k} {v / tot:5.1%}' for k, v in counts.items()))
    off, idm = R[:, 0], R[:, 1]
    q = lambda x: ' / '.join(f'{v:+5.1f}' for v in np.percentile(x, (10, 50, 90)))
    print(f'  gap minus gap paid for, m (p10 / p50 / p90): {q(off)}')
    print(f'  gap minus the IDM aim (ask + speed x time gap): {q(idm)}')
    print(f'  chase pace at its top {np.mean(R[:, 5]):.0%} of ticks; speed minus lead speed, m/s: {q(R[:, 6])}')
    if len(E):
        print(f'  episodes >= 10 s: {len(E)}; gap std p50 {np.median(E[:, 0]):.1f} m, crossings of the paid gap '
              f'{np.median(E[:, 1]):.1f}/min (p90 {np.percentile(E[:, 1], 90):.1f}); within 1 m then 4 m+ back: '
              f'{np.mean(E[:, 2]):.0%}; overshoot (3 m behind -> 1.5 m inside in 2 s): {np.mean(E[:, 3]):.0%}; '
              f'touched {np.mean(E[:, 4]):.0%}')
        # sort the episodes: Copilot's A (never tries), B (overshoots), C (oscillates), or holds
        cat = np.where(E[:, 3] > 0.5, 'B overshoots',
              np.where((E[:, 0] >= 2.5) & (E[:, 1] >= 4.0), 'C oscillates',
              np.where(E[:, 5] > 2.0, np.where(E[:, 6] > 0.5, 'A far, pinned (cannot)', 'A far, held back'), 'holds the gap')))
        print('  episodes by kind: ' + ', '.join(f'{k} {np.mean(cat == k):.0%}' for k in sorted(set(cat))))
        if kinds is not None:
            grp = np.array([('human' if any(w in kinds[int(g)] for w in ('telemetria', 'public', 'human', 'monitor', 'owner'))
                             else 'bot') if 0 <= g < len(kinds) else '?' for g in E[:, 7]])
            for gname in sorted(set(grp)):
                m = grp == gname
                print(f'    {gname:<6} {int(m.sum()):4d} eps: ' + ', '.join(f'{k} {np.mean(cat[m] == k):.0%}' for k in sorted(set(cat)))
                      + f'; median gap minus paid {np.median(E[m, 5]):+.1f} m')
    print('  by gap minus paid (m):   ticks   steer std  pedal mean/std  brake mean  pace top')
    for lo, hi in ((-9, -1), (-1, 1), (1, 3), (3, 6), (6, 99)):
        m = (off >= lo) & (off < hi)
        if m.sum() < 50:
            continue
        print(f'    {lo:+3d} .. {hi:+3d}        {m.mean():5.0%}   {np.std(R[m, 2]):.2f}       {np.mean(R[m, 3]):+.2f} / {np.std(R[m, 3]):.2f}'
              f'      {np.mean(R[m, 4]):.2f}       {np.mean(R[m, 5]):.0%}')


def plot(tr, segs, path, title, n=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    long = sorted(segs, key=lambda x: -(x[2] - x[1]))
    pick = long[::max(1, len(long) // (n * 3))][:n]
    fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 9), sharex='col', squeeze=False)
    for j, (c, s, u) in enumerate(pick):
        s0 = max(0, s - int(WARMUP_S / 0.05))
        t = np.cumsum(tr['dt'][s0:u, c]); t -= t[0]
        ax = axes[0, j]
        ax.plot(t, tr['gap'][s0:u, c], color='#2b6cb0', lw=1.2, label='gap')
        ax.plot(t, tr['target'][s0:u, c], color='#718096', lw=1, ls='--', label='gap paid for')
        ct = tr['contact'][s0:u, c] > 0.5
        if ct.any():
            ax.scatter(t[ct], tr['gap'][s0:u, c][ct], s=6, color='#c53030', label='touch', zorder=3)
        ax.set_ylim(-1, 20); ax.set_title(f'car {c}, lead #{int(tr["ghost"][s, c])}', fontsize=9)
        axes[1, j].plot(t, tr['rel_heading'][s0:u, c], color='#2f855a', lw=1, label='heading vs lead, deg')
        axes[1, j].plot(t, tr['slip'][s0:u, c], color='#d69e2e', lw=1, label='own slip, deg')
        axes[1, j].plot(t, tr['lead_slip'][s0:u, c], color='#805ad5', lw=1, ls=':', label='lead slip, deg')
        axes[2, j].plot(t, tr['steer'][s0:u, c], color='#2d3748', lw=1, label='steer')
        axes[2, j].plot(t, tr['pedal'][s0:u, c], color='#dd6b20', lw=1, label='pedal')
        axes[2, j].plot(t, tr['brake'][s0:u, c], color='#c53030', lw=1, label='front brake')
        axes[2, j].set_ylim(-1.05, 1.05)
        axes[3, j].plot(t, tr['pace'][s0:u, c], color='#2b6cb0', lw=1, label='chase pace asked')
        axes[3, j].axhline(tr['pace_top'][s, c], color='#718096', lw=0.8, ls='--', label='pace top')
        axes[3, j].set_ylim(0.5, 1.5)
        axes[3, j].set_xlabel('s')
    for i, lab in enumerate(('gap, m', 'angles, deg', 'actions', 'pace')):
        axes[i, 0].set_ylabel(lab)
        axes[i, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--cars', type=int, default=256)
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--gaps', type=float, nargs=2, default=(0.0, 1.0), metavar=('LOW', 'HIGH'))
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out')
    ap.add_argument('--plot')
    de.add_chase_controller_args(ap)
    args = ap.parse_args()
    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    w.ghosts = de.load_ghosts(args.tracks, args.track, args.lead_speed)
    kinds = de.load_ghost_kinds(args.tracks, args.track, args.lead_speed)
    pol = Policy.load_json(args.policy)
    cfg = replace(de.env_config_for(pol.meta), chase_gap_low=args.gaps[0], chase_gap_high=args.gaps[1],
                  chase_scenarios=False, chase_share=1.0, **de.chase_controller_overrides(args))
    tr = record(w, pol, cfg, args.cars, args.seconds, args.seed)
    segs = segments(tr)
    report(f'{args.policy} on {args.track}', tr, segs, (cfg.chase_time_gap_s, cfg.chase_time_gap_blind_s), kinds)
    if args.out:
        np.savez_compressed(args.out, **tr)
    if args.plot:
        plot(tr, segs, args.plot, f'{args.track}: gap, angles, actions and pace through chase runs')


if __name__ == '__main__':
    main()
