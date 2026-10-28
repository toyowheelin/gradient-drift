"""
chase_drops.py - why a chaser falls out of its drift: throttle lifts, front brake, or the lead.

    python chase_drops.py policy.json [more.json ...] --track driftplayground_2021 [--cars 256] [--seconds 120]

Each policy drives --cars cars for --seconds behind the track's ghosts (every episode a chase, asked
gaps over --gaps), and again alone. A drop is a car that was sliding 20 deg or more for at least
DROP_HELD_S and then, within DROP_FADE_S, falls under 10 deg and stays there for at least DROP_OUT_S (a
transition passes through zero faster than that, so it is not counted). For the DROP_LOOK_S before each drop the report gives:

  drops/min      drops per minute of sliding (20 deg+)
  lift           share of drops with the throttle under LIFT_BELOW somewhere in the look-back
  brake          share with the brake action over BRAKE_ABOVE somewhere in the look-back
  lead held      share where the lead was still sliding 20 deg+ when the chaser dropped
  P(drop|lift)   of all lifts while sliding, the share followed by a drop within DROP_LOOK_S
                 (P(drop|brake) the same for braking): which one actually throws the angle away

and the share of the time sliding 20 deg+, 10 - 20 and under 10, with the mean throttle and how often it
is lifted or braking (a chaser can lose the drift statistic by sliding shallower without dropping out).
For chase drops, the median gap minus the gap paid for (m, - = closer than asked) and the median
closing speed (m/s, + = gaining on the lead) at the start of the look-back, against the same over all
sliding ticks. Then drops/min and the lift share by gap-to-target bins.
"""
import argparse
from dataclasses import replace

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy

SLIDE_DEG = 20.0
OUT_DEG = 10.0
DROP_HELD_S = 0.5
DROP_OUT_S = 0.5
DROP_FADE_S = 0.5
DROP_LOOK_S = 1.0
LIFT_BELOW = 0.2
BRAKE_ABOVE = 0.2


def record(world, pol, cfg, cars, seconds, seed):
    """Per-tick traces (ticks x cars) of one policy's driving; a new episode starts where ep_id changes."""
    env = de.DriftVecEnv(world, cars, replace(cfg, privileged=True), seed=seed)
    obs = env.reset()
    ticks = int(seconds / 0.05)
    names = ('slip', 'throttle', 'brake', 'chasing', 'gap_off', 'gap', 'lead_slip', 'dt', 'ep_id')
    tr = {k: np.zeros((ticks, cars)) for k in names}
    ep_id = np.arange(cars, dtype=np.float64)
    next_id = cars
    for t in range(ticks):
        act = np.asarray(pol(obs), dtype=np.float64)
        obs, _, term, trunc, _ = env.step(act)
        done = term | trunc
        tr['slip'][t] = np.degrees(env.diags[:, fs.D_RealSlipAngleRad])
        tr['throttle'][t] = np.clip(act[:, 1], 0.0, 1.0)
        tr['brake'][t] = np.clip(act[:, 2], 0.0, 1.0) if act.shape[1] > 2 else 0.0
        tr['chasing'][t] = env.priv[:, 0] > 0.5
        tr['gap'][t] = env.priv[:, 1] * 20.0
        tr['gap_off'][t] = env.priv[:, 2] * 5.0
        tr['lead_slip'][t] = env.priv[:, 5] * 90.0
        tr['dt'][t] = env.mids[:, fs.M_DT]
        tr['ep_id'][t] = ep_id
        # the tick a car finishes holds the new episode's first state: give it the new id
        k = np.nonzero(done)[0]
        ep_id[k] = next_id + np.arange(len(k))
        next_id += len(k)
        tr['ep_id'][t, k] = ep_id[k]
    return tr


def drops(tr, chase_only):
    """Drop events and per-tick lift/brake markers, from the traces."""
    T, n = tr['slip'].shape
    dt = float(np.median(tr['dt']))
    held, out, look = int(DROP_HELD_S / dt), int(DROP_OUT_S / dt), int(DROP_LOOK_S / dt)
    fade = int(DROP_FADE_S / dt)
    a = np.abs(tr['slip'])
    ev = []
    slide_ticks = 0
    lifts = brakes = lift_drops = brake_drops = 0
    slide_rows = []
    for c in range(n):
        s, ep = a[:, c], tr['ep_id'][:, c]
        use = tr['chasing'][:, c] > 0.5 if chase_only else np.ones(T, dtype=bool)
        sliding = (s >= SLIDE_DEG) & use
        slide_ticks += int(sliding.sum())
        drop_at = np.zeros(T, dtype=bool)
        t = held
        while t < T - out:
            seg = slice(t - held, t)
            # the angle may fade through 10 - 20 deg for up to DROP_FADE_S before it is out
            u = t
            while u < min(t + fade, T - out) and s[u] >= OUT_DEG and s[u] < SLIDE_DEG:
                u += 1
            if (sliding[t - 1] and s[t] < SLIDE_DEG and np.all(s[seg] >= SLIDE_DEG) and np.all(ep[seg] == ep[t])
                    and np.all(s[u:u + out] < OUT_DEG) and np.all(ep[t:u + out] == ep[t])):
                drop_at[t] = True
                lb = slice(max(0, t - look), t)
                same = ep[lb] == ep[t]
                th, br = tr['throttle'][lb, c][same], tr['brake'][lb, c][same]
                t0 = max(0, t - look)
                closing = 0.0
                if t0 + 1 < t and ep[t0] == ep[t]:
                    closing = -(tr['gap'][t0 + 1, c] - tr['gap'][t0, c]) / dt
                ev.append((th.min() < LIFT_BELOW, br.max() > BRAKE_ABOVE,
                           abs(tr['lead_slip'][t - 1, c]) >= SLIDE_DEG, tr['gap_off'][t0, c], closing))
                t = u + out
            else:
                t += 1
        # every lift / brake tick while sliding: is a drop within the look-ahead?
        drop_idx = np.nonzero(drop_at)[0]
        for name in ('lift', 'brake'):
            mark = (tr['throttle'][:, c] < LIFT_BELOW) if name == 'lift' else (tr['brake'][:, c] > BRAKE_ABOVE)
            idx = np.nonzero(mark & sliding)[0]
            if len(idx) == 0:
                continue
            if len(drop_idx):
                j = np.searchsorted(drop_idx, idx)
                hit = (j < len(drop_idx)) & (drop_idx[np.minimum(j, len(drop_idx) - 1)] - idx <= look)
            else:
                hit = np.zeros(len(idx), dtype=bool)
            if name == 'lift':
                lifts += len(idx); lift_drops += int(hit.sum())
            else:
                brakes += len(idx); brake_drops += int(hit.sum())
        ok = sliding.copy()
        ok[0] = False
        idx = np.nonzero(ok)[0]
        same = ep[idx] == ep[idx - 1]
        idx = idx[same]
        slide_rows.append(np.stack([tr['gap_off'][idx, c], -(tr['gap'][idx, c] - tr['gap'][idx - 1, c]) / dt], 1))
    E = np.array(ev, dtype=np.float64).reshape(-1, 5)
    use = tr['chasing'] > 0.5 if chase_only else np.ones_like(a, dtype=bool)
    inside = use.sum()
    share = lambda m: float((m & use).sum()) / max(inside, 1)
    th, br = tr['throttle'], tr['brake']
    time = dict(slide=share(a >= SLIDE_DEG), shallow=share((a >= OUT_DEG) & (a < SLIDE_DEG)), out=share(a < OUT_DEG),
                throttle=float(th[use].mean()) if inside else 0.0, lifted=share(th < LIFT_BELOW),
                braking=share(br > BRAKE_ABOVE))
    return dict(E=E, time=time, slide_min=slide_ticks * dt / 60.0, lifts=lifts, lift_drops=lift_drops,
                brakes=brakes, brake_drops=brake_drops, slide=np.concatenate(slide_rows) if slide_rows else np.zeros((0, 2)))


def line(name, d, chase):
    E = d['E']
    n = len(E)
    rate = n / max(d['slide_min'], 1e-9)
    pl = d['lift_drops'] / max(d['lifts'], 1)
    pb = d['brake_drops'] / max(d['brakes'], 1)
    sh = lambda k: np.mean(E[:, k]) if n else 0.0
    out = (f"{name[-30:]:<30} {n:5d} {rate:8.2f} {sh(0):6.0%} {sh(1):6.0%} {sh(2):6.0%} "
           f"{pl:9.1%} {pb:9.1%}")
    if chase:
        S = d['slide']
        md = lambda x: float(np.median(x)) if len(x) else float('nan')
        out += (f"  {md(E[:, 3]):+5.1f}/{md(S[:, 0]):+5.1f}  {md(E[:, 4]):+5.2f}/{md(S[:, 1]):+5.2f}")
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='+')
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--cars', type=int, default=256)
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--gaps', type=float, nargs=2, default=(0.0, 1.0), metavar=('LOW', 'HIGH'))
    de.add_chase_controller_args(ap)
    args = ap.parse_args()

    w = fs.load_world(args.tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(args.tracks, args.track, w)
    w.ghosts = de.load_ghosts(args.tracks, args.track)
    if w.ghosts is None:
        raise SystemExit(f'no ghosts for {args.track}: record_ghosts.py --track {args.track}')
    head = f"{'policy':<30} {'drops':>5} {'drops/min':>8} {'lift':>6} {'brake':>6} {'lead held':>6} {'P(drop|lift)':>9} {'P(drop|brake)':>9}"
    results = {}
    for mode in ('solo', 'chase'):
        print(f'\n{mode}')
        print(head + ('  gap-target drop/all  closing drop/all' if mode == 'chase' else ''))
        for path in args.policies:
            pol = Policy.load_json(path)
            cfg = replace(de.env_config_for(pol.meta), chase_gap_low=args.gaps[0], chase_gap_high=args.gaps[1],
                          chase_scenarios=False, chase_share=1.0 if mode == 'chase' else 0.0,
                          **de.chase_controller_overrides(args))
            d = drops(record(w, pol, cfg, args.cars, args.seconds, 11), mode == 'chase')
            results[(mode, path)] = d
            line(path, d, mode == 'chase')
    print('\nshare of the time: sliding 20 deg+ / 10 - 20 / under 10 | mean throttle, lifted (under '
          f'{LIFT_BELOW:g}), braking (over {BRAKE_ABOVE:g})')
    for mode in ('solo', 'chase'):
        for path in args.policies:
            m = results[(mode, path)]['time']
            print(f"{mode:<6}{path[-30:]:<30} {m['slide']:4.0%} / {m['shallow']:4.0%} / {m['out']:4.0%} | "
                  f"{m['throttle']:.2f}  {m['lifted']:4.0%}  {m['braking']:4.0%}")
    print('\nchase drops by gap minus the gap paid for (m): drops/min of sliding there | lift share')
    bins = (-np.inf, -2.0, -0.5, 0.5, 2.0, 5.0, np.inf)
    print(f"{'policy':<30} " + ' '.join(f'{lo:>+5.1f}..{hi:<+5.1f}' for lo, hi in zip(bins[:-1], bins[1:])))
    for path in args.policies:
        d = results[('chase', path)]
        E, S = d['E'], d['slide']
        dt_min = d['slide_min'] / max(len(S), 1)
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (E[:, 3] >= lo) & (E[:, 3] < hi) if len(E) else np.zeros(0, dtype=bool)
            ms = (S[:, 0] >= lo) & (S[:, 0] < hi)
            rate = m.sum() / max(ms.sum() * dt_min, 1e-9)
            lift = np.mean(E[m, 0]) if m.any() else float('nan')
            cells.append(f'{rate:5.2f} | {lift:4.0%}')
        print(f'{path[-30:]:<30} ' + ' '.join(f'{c:>12}' for c in cells))


if __name__ == '__main__':
    main()
