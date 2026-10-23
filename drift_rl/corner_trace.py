"""
corner_trace.py - how a policy drives one stretch of a course, point by point, against the owner's
own laps through it.

    python corner_trace.py runs/gen18/policy.json --track driftplayground_2021 --from 125 --to 165
    python corner_trace.py runs/gen18/policy.json --track driftplayground_2021 --from 125 --to 165 \
        --human path/to/<your telemetry recording>.csv

Per 5-point bin of the AI line: progress speed (km/h along the line, what the pace ask and reward
use), the pace ask, the car's own speed (forward speed, which drops at big slip angles - not the
number to compare with the ask), rear slip, pedal and the brake actually applied (none with
DisableBrakes), and with --human the owner's progress speed per pass through the stretch from a
telemetry recorder CSV (X, Z columns; the median of each pass, then the range over the passes).
"""
import argparse
import os
import sys

import numpy as np

import drift_env as de
import fast_sim as fs
import track_files as tf
from policy import Policy

RECORDER_DT_S = 0.006      # the telemetry recorder app writes 167 rows a second


def project(A, AB, L2, arc_cum, seg_len, q):
    """Arc length along the AI line of the point nearest q (x, z), and that point's index."""
    t = np.clip(((q - A) * AB).sum(1) / L2, 0.0, 1.0)
    c = A + AB * t[:, None]
    i = int(((c - q) ** 2).sum(1).argmin())
    return arc_cum[i] + t[i] * seg_len[i], i


def human_passes(csv, w, lo, hi, lap_m):
    """The owner's median progress speed per bin for each pass through points lo..hi."""
    import pandas as pd
    d = pd.read_csv(csv, usecols=['X', 'Z'])
    X, Z = d.X.values, d.Z.values
    A = w.sp_pos[:, [0, 2]]
    AB = np.roll(A, -1, axis=0) - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    arc_cum = np.r_[0.0, np.cumsum(w.sp_length[:-1])]
    box_lo = A[max(lo - 10, 0):hi + 10].min(0) - 15.0
    box_hi = A[max(lo - 10, 0):hi + 10].max(0) + 15.0
    near = np.flatnonzero((X > box_lo[0]) & (X < box_hi[0]) & (Z > box_lo[1]) & (Z < box_hi[1]))
    pt = np.full(len(X), -1)
    arc = np.full(len(X), np.nan)
    for k in near:
        arc[k], pt[k] = project(A, AB, L2, arc_cum, w.sp_length, np.array([X[k], Z[k]]))
    da = np.r_[(np.diff(arc) + lap_m / 2) % lap_m - lap_m / 2, np.nan]
    da[~np.isfinite(da) | (np.abs(da) > 3.0)] = np.nan
    prog = da / RECORDER_DT_S * 3.6
    inside = (pt >= lo) & (pt <= hi)
    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
    passes = []
    for s, e in edges.reshape(-1, 2):
        if pt[s] > lo + 5 or pt[e - 1] < hi - 5:
            continue                # only whole passes through the stretch
        row = []
        for b in range(lo, hi + 1, 5):
            m = (pt[s:e] >= b) & (pt[s:e] < b + 5) & np.isfinite(prog[s:e])
            row.append(float(np.median(prog[s:e][m])) if m.any() else np.nan)
        passes.append(row)
    return np.array(passes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policy')
    tf.tracks_arg(ap)
    ap.add_argument('--track', required=True)
    ap.add_argument('--from', dest='lo', type=int, required=True, help='first AI-line point')
    ap.add_argument('--to', dest='hi', type=int, required=True, help='last AI-line point')
    ap.add_argument('--pace', type=float, default=1.0)
    ap.add_argument('--cars', type=int, default=64)
    ap.add_argument('--ticks', type=int, default=3300)
    ap.add_argument('--human', default=None, metavar='CSV', help="the owner's telemetry recording of this track")
    args = ap.parse_args()
    tracks = args.tracks

    pol = Policy.load_json(args.policy)
    w = fs.load_world(tracks, track=args.track)
    w.pace_ref = de.course_pace_reference(tracks, args.track, w)
    lap_m = float(w.sp_length.sum())
    cfg, rw = de.server_run_configs(pol.meta, ticks=args.ticks, pace=args.pace)
    env = de.DriftVecEnv(w, n_envs=args.cars, env_config=cfg, reward_config=rw, seed=9)
    no_brakes = float((pol.meta.get('plugin_settings') or {}).get('DisableBrakes', 0.0)) == 1.0

    lo, hi = args.lo, args.hi
    n_bins = (hi - lo) // 5 + 1
    acc = np.zeros((n_bins, 6))
    cnt = np.zeros(n_bins)
    obs = env.reset()
    prev_arc = env.states[:, fs.S_ARC].copy()
    for t in range(args.ticks):
        a = np.asarray(pol(obs))
        obs, _, _, _, _ = env.step(a)
        s, d = env.states, env.diags
        arc = s[:, fs.S_ARC]
        ds = (arc - prev_arc + lap_m / 2) % lap_m - lap_m / 2
        prev_arc = arc.copy()
        if t < 200:
            continue                # past the start
        dt = np.maximum(env.mids[:, de.M_DT], 1e-3)
        pt = env.mids[:, de.M_NEAREST].astype(int)
        pedal = np.clip(a[:, 1], -1.0, 1.0)
        brake = np.clip(a[:, 2], 0.0, 1.0) if a.shape[1] > 2 else np.zeros(len(a))
        if no_brakes:
            pedal, brake = np.maximum(pedal, 0.0), np.zeros(len(a))
        m = (pt >= lo) & (pt <= hi) & (np.abs(ds) < 5.0)
        if not m.any():
            continue
        row = np.c_[ds / dt * 3.6, args.pace * w.pace_ref[pt],
                    s[:, fs.S_OWN_SPEED] * 3.6, np.abs(np.degrees(d[:, fs.D_RealSlipAngleRad])), pedal, brake][m]
        k = (pt[m] - lo) // 5
        np.add.at(acc, k, row)
        np.add.at(cnt, k, 1)

    owner = human_passes(args.human, w, lo, hi, lap_m) if args.human else None
    print(f"{os.path.basename(args.policy)} on {args.track}, pace {args.pace}"
          f"{' (DisableBrakes)' if no_brakes else ''}; speeds km/h, progress = along the AI line")
    head = f"{'point':>5s} {'progress':>8s} {'ask':>6s} {'own':>6s} {'slip':>5s} {'pedal':>6s} {'brake':>6s}"
    if owner is not None:
        head += f"   owner ({len(owner)} passes) min - median - max"
    print(head)
    for b in range(n_bins):
        v = acc[b] / max(cnt[b], 1)
        line = f"{lo + 5 * b:5d} {v[0]:8.1f} {v[1]:6.1f} {v[2]:6.1f} {v[3]:5.0f} {v[4]:+6.2f} {v[5]:6.2f}"
        if owner is not None and len(owner):
            col = owner[:, b]
            col = col[np.isfinite(col)]
            if len(col):
                line += f"   {col.min():5.1f} - {np.median(col):5.1f} - {col.max():5.1f}"
        print(line)


if __name__ == '__main__':
    main()
