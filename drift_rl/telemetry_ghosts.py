"""
telemetry_ghosts.py - a human's laps as ghost leaders for tandem training, beside record_ghosts.py's.

    python telemetry_ghosts.py --track driftplayground_2021 <telemetria>.csv [--seconds-apart 25]
    python telemetry_ghosts.py --track <track> ai_car_monitor_<date>_<time>.csv [--cars 1 4 16]

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

An AI Car Monitor CSV (the in-game app that logs every car this client sees, one row per car per
frame) gives a lead per car, other drivers on a public server included. A remote car's tire
values are not sent, so its heading comes from its four tire contact points instead (front axle
middle minus rear axle middle), and the body slip is the course angle minus that heading. Each car
is cut into stretches where it stays on the track: a sample with a value out of range, a move
faster than --max-kmh between frames (a reset to the pits), the body turning over 0.6 rad in one
frame (a network correction) or a gap of over a second ends one. Only stretches as long as a ghost
give runs. A public server can run another version of the track: the median distance from the
AI line printed for each lead shows it (a few metres is driving; tens is another layout). Driver names are read nowhere: cars are told apart by the app's
CarId and printed with their model.
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
CROWD_S = 0.02      # AI Car Monitor rows (one per frame, ~27 a second) resampled to this spacing


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


def is_crowd(path):
    """True for an AI Car Monitor CSV (every car, a CarId column), False for the telemetria app's."""
    return 'CarId' in pd.read_csv(path, nrows=0).columns


def crowd_paths(path, steer_ratio, max_kmh, cars=None):
    """car_path's dict for each stretch of each car in an AI Car Monitor CSV, with its CarId and
    model: [(car_id, model, path dict)]."""
    cols = ['ClientTimeMs', 'CarId', 'CarModel', 'X', 'Z', 'SpeedKMH', 'Gas', 'Brake', 'Steer'] + \
        [f'Wheel{w}_{a}' for w in ('FL', 'FR', 'RL', 'RR') for a in 'XZ']
    d = pd.read_csv(path, usecols=cols)
    out = []
    for car_id, c in d.groupby('CarId'):
        if cars and car_id not in cars:
            continue
        c = c.sort_values('ClientTimeMs').drop_duplicates('ClientTimeMs').reset_index(drop=True)
        t = (c.ClientTimeMs.values - c.ClientTimeMs.values[0]) / 1000.0
        fx = 0.5 * (c.WheelFL_X + c.WheelFR_X - c.WheelRL_X - c.WheelRR_X).values
        fz = 0.5 * (c.WheelFL_Z + c.WheelFR_Z - c.WheelRL_Z - c.WheelRR_Z).values
        wheelbase = np.hypot(fx, fz)
        good = (np.abs(c.X.values) < 5000) & (np.abs(c.Z.values) < 5000) & (c.SpeedKMH.values < 300) \
            & (wheelbase > 1.5) & (wheelbase < 4.0)
        steer = np.nan_to_num(c.Steer.values)                 # some cars never send it
        step = np.hypot(np.diff(c.X.values), np.diff(c.Z.values))
        turn = np.abs(np.angle(np.exp(1j * np.diff(np.arctan2(fz, fx)))))
        # a body turning over 0.6 rad in a frame (~17 rad/s) is a network correction, not driving
        fast = step / np.maximum(np.diff(t), 1e-3) * 3.6 > max_kmh
        breaks = np.r_[True, fast | (turn > 0.6) | (np.diff(t) > 1.0)] | ~good
        # stretches: good samples between breaks (a bad sample starts one, and is left out)
        stretch = np.cumsum(breaks)
        for n in np.unique(stretch[good]):
            k = np.flatnonzero((stretch == n) & good)
            if t[k[-1]] - t[k[0]] < 2.0:
                continue
            out.append((int(car_id), c.CarModel.iloc[k[0]],
                        stretch_path(t[k], c.X.values[k], c.Z.values[k], np.arctan2(fz[k], fx[k]),
                                     steer[k], c.Gas.values[k], c.Brake.values[k], steer_ratio)))
    return out


def stretch_path(t, x_raw, z_raw, heading_raw, steer, gas, brake, steer_ratio):
    """car_path's dict from one car's positions and body heading on an uneven clock."""
    grid = np.arange(t[0], t[-1], CROWD_S)
    win = 15                                               # 0.3 s
    if len(grid) <= win:
        win = len(grid) - 1 - (len(grid) % 2 == 0)
    x_g, z_g, h_g = (np.interp(grid, t, a) for a in (x_raw, z_raw, np.unwrap(heading_raw)))
    x = savgol_filter(x_g, win, 3)
    z = savgol_filter(z_g, win, 3)
    vx_w = savgol_filter(x_g, win, 3, deriv=1, delta=CROWD_S)
    vz_w = savgol_filter(z_g, win, 3, deriv=1, delta=CROWD_S)
    v = np.hypot(vx_w, vz_w)
    course = np.unwrap(np.arctan2(vz_w, vx_w))
    heading = savgol_filter(h_g, win, 3)
    yaw = savgol_filter(h_g, win, 3, deriv=1, delta=CROWD_S)
    # course is meaningless when barely moving: keep the body's own heading there
    beta = np.where(v > 1.0, np.arctan2(np.sin(course - heading), np.cos(course - heading)), 0.0)
    return dict(t=grid - grid[0], x=x, z=z, heading=heading, slip=-beta, yaw=yaw, speed=v,
                steer=np.radians(steer_ratio * np.interp(grid, t, steer)),
                pedal=np.clip(np.interp(grid, t, gas), 0, 1), brake=np.clip(np.interp(grid, t, brake), 0, 1))


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
    ap.add_argument('--cars', type=int, nargs='+', help='AI Car Monitor CSVs: only these CarIds')
    ap.add_argument('--max-kmh', type=float, default=180.0,
                    help='AI Car Monitor CSVs: a move faster than this between frames ends a stretch')
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
    leads = []
    for path in args.recordings:
        if is_crowd(path):
            leads += [(f'{os.path.basename(path)} car {car_id} ({model})', c) for car_id, model, c in
                      crowd_paths(path, args.steer_ratio, args.max_kmh, args.cars)]
        else:
            leads.append((os.path.basename(path), car_path(path, args.wheelbase, args.front_share,
                                                           args.steer_ratio)))
    for name, c in leads:
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
        print(f'{name}: {c["t"][-1]:.0f} s, {np.mean(g[:, de.GH_SPEED]) * 3.6:.1f} km/h, '
              f'sliding {100 * slide:.0f}%, {np.median(np.abs(g[:, de.GH_CTE])):.1f} m from the AI line, '
              f'{kept} runs of {length * args.tick:.0f} s')
    if not new:
        raise SystemExit('no runs long enough: the recordings are shorter than the ghosts')
    np.savez_compressed(out, ghosts=np.concatenate([old, np.stack(new)]))
    print(f'{len(old)} + {len(new)} ghosts -> {out} (before: {backup})')


if __name__ == '__main__':
    main()
