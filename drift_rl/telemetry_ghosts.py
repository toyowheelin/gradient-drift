"""
telemetry_ghosts.py - a human's laps as ghost leaders for tandem training, beside record_ghosts.py's.

    python telemetry_ghosts.py --track driftplayground_2021 <telemetria>.csv [--seconds-apart 25]

The in-game telemetry app's CSV (human_recording.py) of someone driving the course in Assetto Corsa
becomes ghosts like record_ghosts.py's: the same columns, sampled every --tick seconds, cut into runs
as long as the track's ghosts already are (one starting every --seconds-apart), and added to
<track>_ghosts.npz (the file before is kept as <track>_ghosts_before_telemetry.npz, and a
second run adds to that again rather than twice; after record_ghosts.py, delete it). A run with
the car near stopped (under --min-kmh for over a second) is left out.

A lead driven in the game's own physics is what a chaser meets behind a player: no bot habits, and
a car the simulator did not drive. GT Sophy trained against a mix of opponents for that reason
(Wurman et al., Nature 2022: against itself alone it rear-ended humans).

The car's state comes from its path: speed and course angle from the smoothed positions; the body
slip angle from the rear tires' slip angle and the yaw balance tan(rear slip) = (v_y - b r) / v_x,
b the CG to rear axle distance (--wheelbase, --front-share; the defaults are the E36 Touring's);
heading = course angle - body slip; yaw rate its derivative. Steering: the road wheel turns
--steer-ratio x the steering wheel's degrees (0.13 for the E36, fitted to the front tires' slip).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import drift_env as de
import fast_sim as fs
import track_files as tf

RAW_S = 0.006       # the app's sample spacing, s (its clock is in whole milliseconds)


def car_path(path, wheelbase, front_share, steer_ratio):
    """Per sample on an even RAW_S grid: time, x, z, heading, slip (fast_sim's sign), yaw rate,
    speed, road-wheel steer (rad), throttle, brake."""
    d = pd.read_csv(path)
    dt = np.diff(d.LapTime_ms.values) / 1000.0
    dt = np.where((dt > 0) & (dt < 0.05), dt, RAW_S)       # laps restart the clock
    t = np.r_[0.0, np.cumsum(dt)]
    keep = np.r_[True, np.diff(t) > 0]
    d, t = d[keep].reset_index(drop=True), t[keep]
    grid = np.arange(t[0], t[-1], RAW_S)
    col = {c: np.interp(grid, t, d[c].values) for c in ('X', 'Z', 'Steer', 'Throttle', 'Brake',
                                                        'SlipAngle_RL', 'SlipAngle_RR')}
    win = 15                                               # 90 ms
    x = savgol_filter(col['X'], win, 3)
    z = savgol_filter(col['Z'], win, 3)
    vx_w = savgol_filter(col['X'], win, 3, deriv=1, delta=RAW_S)
    vz_w = savgol_filter(col['Z'], win, 3, deriv=1, delta=RAW_S)
    v = np.hypot(vx_w, vz_w)
    course = np.unwrap(np.arctan2(vz_w, vx_w))
    course_rate = savgol_filter(course, win, 3, deriv=1, delta=RAW_S)
    b = front_share * wheelbase
    rear = np.radians(0.5 * (col['SlipAngle_RL'] + col['SlipAngle_RR']))
    # body slip beta = atan(v_y / v_x), integrated from the yaw balance above (r = course rate - beta')
    beta = np.zeros(len(grid))
    beta[0] = rear[0]
    for k in range(1, len(grid)):
        v_long = max(v[k - 1] * np.cos(beta[k - 1]), 2.0)
        rate = course_rate[k - 1] - (v_long / b) * (np.tan(beta[k - 1]) - np.tan(rear[k - 1]))
        beta[k] = np.clip(beta[k - 1] + rate * RAW_S, -1.45, 1.45)
    beta_s = savgol_filter(beta, win, 3)
    yaw = course_rate - savgol_filter(beta, win, 3, deriv=1, delta=RAW_S)
    heading = course - beta_s
    return dict(t=grid - grid[0], x=x, z=z, heading=heading, slip=-beta_s, yaw=yaw, speed=v,
                steer=np.radians(steer_ratio * col['Steer']), pedal=col['Throttle'], brake=col['Brake'])


def spline_place(world, x, z, y):
    """Place along the AI line (m, 0 - lap) and signed offset from it (m), as fast_sim measures them."""
    arc = np.zeros(len(x))
    cte = np.zeros(len(x))
    for k in range(len(x)):
        n = fs.world_to_spline(x[k], y, z[k], world.sp_pos)
        a, b, t = fs.find_interpolation_segment(x[k], y, z[k], n, world.sp_pos, world.sp_next, world.sp_prev)
        arc[k] = world.sp_cum[a] + t * world.sp_length[a]
        fx = world.sp_fwd[a, 0] + (world.sp_fwd[b, 0] - world.sp_fwd[a, 0]) * t
        fz = world.sp_fwd[a, 2] + (world.sp_fwd[b, 2] - world.sp_fwd[a, 2]) * t
        ang = np.arctan2(fz, fx)
        ix = world.sp_pos[a, 0] + (world.sp_pos[b, 0] - world.sp_pos[a, 0]) * t
        iz = world.sp_pos[a, 2] + (world.sp_pos[b, 2] - world.sp_pos[a, 2]) * t
        cte[k] = (x[k] - ix) * (-np.sin(ang)) + (z[k] - iz) * np.cos(ang)
    return arc, cte


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recordings', nargs='+')
    ap.add_argument('--track', required=True)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--tick', type=float, default=0.05, help='seconds between ghost samples')
    ap.add_argument('--seconds-apart', type=float, default=25.0, help='one run starting every this many seconds')
    ap.add_argument('--min-kmh', type=float, default=15.0)
    ap.add_argument('--wheelbase', type=float, default=2.75)
    ap.add_argument('--front-share', type=float, default=0.54, help='static weight on the front axle')
    ap.add_argument('--steer-ratio', type=float, default=0.13, help='road wheel deg per steering wheel deg')
    args = ap.parse_args()

    w = fs.load_world(args.tracks, track=args.track)
    folder = tf.folder(args.tracks, args.track)
    out = os.path.join(folder, f'{args.track}_ghosts.npz')
    backup = os.path.join(folder, f'{args.track}_ghosts_before_telemetry.npz')
    if not os.path.exists(out):
        raise SystemExit(f'no ghosts for {args.track} yet: record_ghosts.py --track {args.track} first')
    if not os.path.exists(backup):
        shutil.copyfile(out, backup)
    # always added to the ghosts from before the first run of this, so running it again replaces
    # the telemetry runs rather than adding them twice (record_ghosts.py again: delete the backup)
    old = np.load(backup)['ghosts']
    if old.shape[2] < de.GH_COLS:
        old = np.concatenate([old, np.zeros(old.shape[:2] + (de.GH_COLS - old.shape[2],))], axis=2)
    length = old.shape[1]
    y = float(np.median(w.sp_pos[:, 1]))
    new = []
    for path in args.recordings:
        c = car_path(path, args.wheelbase, args.front_share, args.steer_ratio)
        t = np.arange(0.0, c['t'][-1], args.tick)
        g = np.zeros((len(t), de.GH_COLS))
        g[:, de.GH_T] = t
        for col, key in ((de.GH_X, 'x'), (de.GH_Z, 'z'), (de.GH_HEADING, 'heading'), (de.GH_SLIP, 'slip'),
                         (de.GH_YAW, 'yaw'), (de.GH_SPEED, 'speed'), (de.GH_STEER, 'steer'),
                         (de.GH_PEDAL, 'pedal'), (de.GH_BRAKE, 'brake')):
            g[:, col] = np.interp(t, c['t'], c[key])
        g[:, de.GH_HEADING] = np.arctan2(np.sin(g[:, de.GH_HEADING]), np.cos(g[:, de.GH_HEADING]))
        g[:, de.GH_ARC], g[:, de.GH_CTE] = spline_place(w, g[:, de.GH_X], g[:, de.GH_Z], y)
        step = max(1, int(round(args.seconds_apart / args.tick)))
        slow = g[:, de.GH_SPEED] * 3.6 < args.min_kmh
        kept = 0
        for s0 in range(0, len(g) - length + 1, step):
            run = g[s0:s0 + length].copy()
            # near stopped for over a second: not a lead to chase
            longest = held = 0
            for v in slow[s0:s0 + length]:
                held = held + 1 if v else 0
                longest = max(longest, held)
            if longest * args.tick > 1.0:
                continue
            run[:, de.GH_T] -= run[0, de.GH_T]
            new.append(run)
            kept += 1
        slide = np.mean(np.abs(g[:, de.GH_SLIP]) > np.radians(20.0))
        print(f'{os.path.basename(path)}: {c["t"][-1]:.0f} s, {np.mean(g[:, de.GH_SPEED]) * 3.6:.1f} km/h, '
              f'sliding {100 * slide:.0f}%, {kept} runs of {length * args.tick:.0f} s')
    if not new:
        raise SystemExit('no runs long enough: the recordings are shorter than the ghosts')
    np.savez_compressed(out, ghosts=np.concatenate([old, np.stack(new)]))
    print(f'{len(old)} + {len(new)} ghosts -> {out} (before: {backup})')


if __name__ == '__main__':
    main()
