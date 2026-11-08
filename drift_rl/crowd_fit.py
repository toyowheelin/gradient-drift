"""
crowd_fit.py - how public drivers drift a course and follow each other, from AI Car Monitor CSVs.

    python crowd_fit.py --track driftplayground_2021 ai_car_monitor_<date>_<time>.csv[.gz][:4,8] ...
    python crowd_fit.py --track <track> <recordings> --out tracks/<track>/<track>_crowd.json

AI Car Monitor (the in-game app that logs every car this client sees) records the other drivers
on a public server too. This measures the two things the bots are asked for, the same way the
simulator measures them, so the bots can be asked to drive like the crowd and not only like one
driver:

  * pace: each driver's speed along the AI line against the track's pace reference
    (<track>_pace_reference.csv, what a pace of 1.0 asks), as the share of the reference's time
    over the line they covered (a lap-time share), and split into corners (the slower half of the
    reference) and straights;
  * the gap in a tandem: for every pair of cars one behind the other on the line, the gap between
    their outlines (drift_env._outline_gap, the simulator's own measure and the plugin's), the
    chaser's time gap (gap / speed), how far off the lead's line it runs, and the gap law that fits
    how it follows: the gap it settles at, s = s0 + T v (Intelligent Driver Model's desired gap,
    Treiber, Hennecke, Helbing 2000, fitted by least squares on the steady stretches as IDM is
    calibrated on trajectory data, Treiber and Kesting, "Traffic Flow Dynamics", 2013, ch. 16), and how
    fast it closes a gap that is too big (closing speed per metre over it: 1 / that is a response
    time).

A pair counts while both cars are within --max-offset of the AI line, faster than --min-kmh and the
chaser is 0 - --max-gap behind; a train is such a pair held at least --min-train-s. Give it
recordings of people only: a server whose "AI" plays back one recorded lap on every car follows
at a fixed fraction of a second on the same line and is no crowd (a pair whose lines are under
0.2 m apart is left out, but copies a little apart are not caught). Drivers are told apart by the
app's CarId only; names are read nowhere.

--out writes the numbers as JSON (with the steady samples, so recordings of several tracks can be
pooled: --combine a.json b.json [--out all.json] refits the gap law on all of them and pools the
gaps, weighted by time). train_ppo_gpu.py --chase-gap-crowd draws the asked gaps from such a file;
pace_model.CROWD_SCALE holds each track's pace share against the one the model was fitted on.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import drift_env as de
import fast_sim as fs
import telemetry_ghosts as tg
import track_files as tf

TICK_S = 0.1          # common clock the cars are compared on


def car_tracks(path, world, steer_ratio, max_kmh, cars=None):
    """Per stretch of each car: CarId, model and samples on a TICK_S clock from the recording's
    start - time, x, z, heading, slip, speed, place along the line (unwrapped), line offset."""
    t0 = tg.read_monitor(path, usecols=['ClientTimeMs']).ClientTimeMs.min() / 1000.0
    y = float(np.median(world.sp_pos[:, 1]))
    lap = float(np.sum(world.sp_length))
    out = []
    for car_id, model, c in tg.crowd_paths(path, steer_ratio, max_kmh, cars):
        start = c['start'] - t0
        t = np.arange(np.ceil(start / TICK_S) * TICK_S, start + c['t'][-1], TICK_S)
        if len(t) < 20:
            continue
        s = {k: np.interp(t - start, c['t'], c[k]) for k in ('x', 'z', 'slip', 'speed')}
        s['heading'] = np.interp(t - start, c['t'], np.unwrap(c['heading']))
        arc, cte = tg.spline_place(world, s['x'], s['z'], y)
        s['arc'] = np.unwrap(arc / lap * 2 * np.pi) * lap / (2 * np.pi)
        s['cte'] = cte
        s['t'] = np.round(t / TICK_S).astype(np.int64)
        s['progress'] = np.gradient(s['arc'], TICK_S)
        out.append((car_id, model, s))
    return out


def reference_kmh(world, tracks_dir, track):
    ref = pd.read_csv(tf.pace_reference_path(tracks_dir, track))
    arc = np.r_[0.0, np.cumsum(np.asarray(world.sp_length, dtype=np.float64)[:-1])]
    return arc, ref.Kmh.values[:len(arc)], ref.Source.iloc[0]


def pace_table(stretches, world, ref_arc, ref_kmh, max_offset, min_kmh):
    """Per car: seconds on the line, mean km/h, lap-time share of the reference (all, corners,
    straights), sliding share."""
    lap = float(np.sum(world.sp_length))
    split = np.median(ref_kmh)
    rows = []
    by_car = {}
    for car_id, model, s in stretches:
        by_car.setdefault((car_id, model), []).append(s)
    for (car_id, model), ss in by_car.items():
        dist = {'all': 0.0, 'corner': 0.0, 'straight': 0.0}
        time = dict(dist)
        ref_t = dict(dist)
        slide = []
        for s in ss:
            ok = (np.abs(s['cte']) < max_offset) & (s['progress'] * 3.6 > min_kmh)
            if ok.sum() < 10:
                continue
            ds = s['progress'][ok] * TICK_S
            v_ref = np.interp(s['arc'][ok] % lap, np.r_[ref_arc, lap], np.r_[ref_kmh, ref_kmh[0]]) / 3.6
            corner = v_ref * 3.6 < split
            for part, m in (('all', np.ones_like(corner)), ('corner', corner), ('straight', ~corner)):
                dist[part] += ds[m].sum()
                time[part] += TICK_S * m.sum()
                ref_t[part] += (ds[m] / v_ref[m]).sum()
            slide.append(np.abs(s['slip'][ok]) > np.radians(15.0))
        if time['all'] < 20.0:
            continue
        rows.append(dict(car=car_id, model=model, seconds=round(time['all']),
                         kmh=round(3.6 * dist['all'] / time['all'], 1),
                         pace=ref_t['all'] / time['all'],
                         corner=ref_t['corner'] / max(time['corner'], 1e-9),
                         straight=ref_t['straight'] / max(time['straight'], 1e-9),
                         slide=float(np.mean(np.concatenate(slide)))))
    return pd.DataFrame(rows)


def pair_samples(stretches, world, max_offset, min_kmh, max_gap):
    """Every moment one car is behind another on the line: chaser, lead, the outline gap, speeds,
    closing speed, offset from the lead's line, slips."""
    lap = float(np.sum(world.sp_length))
    rows = []
    for i, (ci, _, a) in enumerate(stretches):
        for j, (cj, _, b) in enumerate(stretches):
            if ci == cj:
                continue
            common, ia, ib = np.intersect1d(a['t'], b['t'], return_indices=True)
            if len(common) < 10:
                continue
            for k in range(len(common)):
                p, q = ia[k], ib[k]           # a chases b
                if abs(a['cte'][p]) > max_offset or abs(b['cte'][q]) > max_offset:
                    continue
                if a['progress'][p] * 3.6 < min_kmh or b['progress'][q] * 3.6 < min_kmh:
                    continue
                ahead = de._wrap_arc(b['arc'][q] - a['arc'][p], lap)
                if ahead <= 0.0 or ahead > max_gap + 6.0:
                    continue
                gap = de._outline_gap(a['x'][p], a['z'][p], a['heading'][p], a['arc'][p] % lap,
                                      b['x'][q], b['z'][q], b['heading'][q], b['arc'][q] % lap, lap)
                if gap < 0.0 or gap > max_gap:
                    continue
                rows.append((common[k], ci, cj, gap, a['progress'][p], b['progress'][q],
                             a['cte'][p] - b['cte'][q], a['slip'][p], b['slip'][q]))
    p = pd.DataFrame(rows, columns='tick chase lead gap v v_lead off slip lead_slip'.split())
    return p.sort_values(['chase', 'lead', 'tick']).reset_index(drop=True)


def trains(p, min_train_s):
    """The pair samples held as a train for at least min_train_s (a pair's run with no break over
    0.5 s), each with its train number; replay copies (lines under 0.2 m apart) left out."""
    out = []
    n = 0
    for (c, l), g in p.groupby(['chase', 'lead']):
        brk = np.r_[True, np.diff(g.tick.values) > round(0.5 / TICK_S)]
        for _, run in g.groupby(np.cumsum(brk)):
            if len(run) * TICK_S < min_train_s or np.median(np.abs(run.off)) < 0.2:
                continue
            run = run.copy()
            run['train'] = n
            run['closing'] = run.v - run.v_lead
            # the gap's own rate is the closing speed measured another way; use it to smooth
            run['gap_rate'] = np.gradient(run.gap.values, TICK_S)
            n += 1
            out.append(run)
    return pd.concat(out) if out else p.iloc[:0]


def fit_law(v, gap, gap_rate):
    """Least squares: the settled gap s = s0 + T v on steady samples (gap_rate None: all given are
    steady), and how fast a gap over it closes (m/s per metre too far; 1 / that a response time)."""
    A = np.c_[np.ones(len(v)), v]
    (s0, T), *_ = np.linalg.lstsq(A, gap, rcond=None)
    return float(s0), float(T)


def gap_law(t):
    """fit_law on the trains' steady samples (|closing| under 1 m/s), plus the closing rate."""
    steady = t[np.abs(t.closing) < 1.0]
    s0, T = fit_law(steady.v.values, steady.gap.values, None)
    err = t.gap.values - (s0 + T * t.v.values)
    far = err > 1.0
    k = float(np.sum(-t.gap_rate.values[far] * err[far]) / max(np.sum(err[far] ** 2), 1e-9))
    return dict(s0_m=s0, time_gap_s=T, closing_per_m=k,
                response_s=float(1.0 / k) if k > 1e-6 else None, steady_samples=int(len(steady)),
                steady_v=[round(float(x), 2) for x in steady.v.values],
                steady_gap=[round(float(x), 2) for x in steady.gap.values])


def combine(paths, out_path):
    """Pools several --out files (one per track): gap quantiles weighted by each file's train
    time, the gap law refitted on all steady samples, pace per track."""
    files = [json.load(open(p)) for p in paths]
    with_gap = [f for f in files if 'gap' in f]
    qs = np.linspace(0, 1, 21)
    # pooled quantiles: sample each file's quantile function in proportion to its seconds
    draws = np.concatenate([np.interp(np.random.default_rng(0).random(int(10 * f['gap']['seconds'])), qs,
                                      [f['gap']['quantiles'][str(round(q, 2))] for q in qs]) for f in with_gap])
    v = np.concatenate([f['gap']['law']['steady_v'] for f in with_gap])
    g = np.concatenate([f['gap']['law']['steady_gap'] for f in with_gap])
    s0, T = fit_law(v, g, None)
    secs = sum(f['gap']['seconds'] for f in with_gap)
    out = dict(tracks=[f['track'] for f in files],
               pace={f['track']: f['pace'] for f in files if 'pace' in f},
               gap=dict(quantiles={str(round(q, 2)): float(np.quantile(draws, q)) for q in qs},
                        median=float(np.median(draws)), p10=float(np.quantile(draws, .1)),
                        p90=float(np.quantile(draws, .9)), seconds=float(secs),
                        trains=int(sum(f['gap']['trains'] for f in with_gap)),
                        time_gap_median_s=float(np.average([f['gap']['time_gap_median_s'] for f in with_gap],
                                                           weights=[f['gap']['seconds'] for f in with_gap])),
                        law=dict(s0_m=s0, time_gap_s=T, steady_samples=int(len(v)))))
    print(f"pooled {', '.join(out['tracks'])}: {secs:.0f} s of trains; gap p10 {out['gap']['p10']:.1f}, "
          f"median {out['gap']['median']:.1f}, p90 {out['gap']['p90']:.1f} m; settled gap {s0:.2f} m + "
          f"{T:.2f} s x speed ({len(v)} steady samples; at 50 km/h {s0 + T * 50 / 3.6:.1f} m)")
    for track, pc in out['pace'].items():
        print(f"  {track}: pace {pc['mean']:.3f} of its reference (corners {pc['corners']:.3f}, "
              f"straights {pc['straights']:.3f}, {pc['drivers']} drivers)")
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'-> {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recordings', nargs='+')
    ap.add_argument('--track')
    ap.add_argument('--combine', action='store_true', help='the arguments are --out files to pool')
    tf.tracks_arg(ap)
    ap.add_argument('--max-offset', type=float, default=6.0, help='m from the AI line that counts as on it')
    ap.add_argument('--min-kmh', type=float, default=20.0)
    ap.add_argument('--max-gap', type=float, default=20.0, help='m, outline gap that still counts as a pair')
    ap.add_argument('--min-train-s', type=float, default=3.0)
    ap.add_argument('--steer-ratio', type=float, default=0.13)
    ap.add_argument('--max-kmh', type=float, default=180.0)
    ap.add_argument('--out', help='write the numbers as JSON here')
    args = ap.parse_args()
    if args.combine:
        return combine(args.recordings, args.out)
    if not args.track:
        ap.error('--track is needed (or --combine)')

    w = fs.load_world(args.tracks, track=args.track)
    ref_arc, ref_kmh, source = reference_kmh(w, args.tracks, args.track)
    paces, gaps = [], []
    for path in args.recordings:
        cars = None
        head, _, tail = path.rpartition(':')
        if head and tail.replace(',', '').isdigit():
            path, cars = head, {int(n) for n in tail.split(',')}
        st = car_tracks(path, w, args.steer_ratio, args.max_kmh, cars)
        name = os.path.basename(path)
        pt = pace_table(st, w, ref_arc, ref_kmh, args.max_offset, args.min_kmh)
        if len(pt):
            pt.insert(0, 'recording', name)
            paces.append(pt)
        tr = trains(pair_samples(st, w, args.max_offset, args.min_kmh, args.max_gap), args.min_train_s)
        if len(tr):
            tr.insert(0, 'recording', name)
            tr['train'] = name + ':' + tr.train.astype(str)
            gaps.append(tr)
    pace = pd.concat(paces) if paces else pd.DataFrame()
    print(f'{args.track}: reference {source}, {np.mean(ref_kmh):.1f} km/h mean over the points')
    if len(pace):
        print('\nPace (share of the reference time over the same line; 1.0 = what pace 1.0 asks):')
        print(pace.round(3).to_string(index=False))
        # time-weighted over drivers: what a crowd lap looks like
        wsum = pace.seconds.sum()
        print(f'crowd: pace {np.average(pace.pace, weights=pace.seconds):.3f} '
              f'(drivers p10 {pace.pace.quantile(.1):.3f}, p90 {pace.pace.quantile(.9):.3f}), '
              f'corners {np.average(pace.corner, weights=pace.seconds):.3f}, '
              f'straights {np.average(pace.straight, weights=pace.seconds):.3f}, {wsum:.0f} s')
    tr = pd.concat(gaps) if gaps else pd.DataFrame()
    out = dict(track=args.track, reference=source, recordings=[os.path.basename(r) for r in args.recordings])
    if len(pace):
        out['pace'] = dict(mean=float(np.average(pace.pace, weights=pace.seconds)),
                           p10=float(pace.pace.quantile(.1)), p90=float(pace.pace.quantile(.9)),
                           corners=float(np.average(pace.corner, weights=pace.seconds)),
                           straights=float(np.average(pace.straight, weights=pace.seconds)),
                           drivers=int(len(pace)), seconds=float(pace.seconds.sum()))
    if len(tr):
        drift = (np.abs(tr.slip) > np.radians(15)) & (np.abs(tr.lead_slip) > np.radians(15))
        per = tr.groupby('train').agg(s=('gap', 'size'), gap=('gap', 'median'), p10=('gap', lambda x: x.quantile(.1)),
                                      kmh=('v', lambda x: 3.6 * x.median()), off=('off', lambda x: x.abs().median()),
                                      closing_sd=('closing', 'std'))
        per['s'] *= TICK_S
        per['both_drift'] = drift.groupby(tr.train).mean()
        print(f'\nTrains (pairs held {args.min_train_s:.0f} s or more, gap between outlines):')
        print(per.sort_values('s', ascending=False).round(2).to_string())
        q = tr.gap.quantile([.1, .25, .5, .75, .9])
        hw = (tr.gap / tr.v).quantile([.1, .5, .9])
        law = gap_law(tr)
        bins = [0, 2, 4, 6, 8, 10, 15, 20]
        share = pd.cut(tr.gap, bins).value_counts(normalize=True).sort_index()
        print(f'\nall trains: {len(tr) * TICK_S:.0f} s in {tr.train.nunique()} trains; gap p10 {q[.1]:.1f}, '
              f'p25 {q[.25]:.1f}, median {q[.5]:.1f}, p75 {q[.75]:.1f}, p90 {q[.9]:.1f} m; time gap median '
              f'{hw[.5]:.2f} s (p10 {hw[.1]:.2f}, p90 {hw[.9]:.2f}); off the lead\'s line median '
              f'{tr.off.abs().median():.1f} m; both drifting {100 * drift.mean():.0f}%')
        print('gap shares: ' + ', '.join(f'{a}-{b} m {100 * s:.0f}%' for (a, b), s in
                                          zip(zip(bins[:-1], bins[1:]), share.values)))
        print(f"gap law: settled gap {law['s0_m']:.2f} m + {law['time_gap_s']:.2f} s x speed "
              f"({law['steady_samples']} steady samples); closes {law['closing_per_m']:.2f} m/s per m too far"
              + (f" (response {law['response_s']:.1f} s)" if law['response_s'] else ''))
        out['gap'] = dict(quantiles={str(round(k, 2)): float(v) for k, v in tr.gap.quantile(np.linspace(0, 1, 21)).items()},
                          median=float(q[.5]), p10=float(q[.1]), p90=float(q[.9]),
                          time_gap_median_s=float(hw[.5]), offset_median_m=float(tr.off.abs().median()),
                          both_drifting=float(drift.mean()), seconds=float(len(tr) * TICK_S),
                          trains=int(tr.train.nunique()), law=law)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
