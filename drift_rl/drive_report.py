"""
drive_report.py - policies side by side, and against a human, on how they drive: pace, sliding,
rotation, and how much they leave the course or lean on the walls.

    python drive_report.py runs/gen6/policy.json runs/gen7/policy.json \\
        --track spiritpeaksraceway --track driftplayground_2021 \\
        --human spiritpeaksraceway_telemetria_<date>_<time>.csv --pace-reference runs/gen3/policy.json

Each policy drives what the server would have it drive: long runs nothing resets (server_run_configs),
in the car physics its file carries (the plugin turns on only what a policy was trained with;
--set changes a plugin setting for every policy, e.g. --set EnableTireLoadSensitivity=1). A pace is a
fraction of the course's flat-out speed: with --pace-reference that is measured here, by the given
policy in each policy's own physics (what that policy's server would use); without it, from the
course's <track>_pace_reference.csv.

Rotation is measured the same way for a bot and for a human recording (the in-game telemetry
app's CSV, 167 samples a second): resampled to 20 Hz, the yaw rate is the rate the path turns
minus the rate the rear slip angle changes, so neither side gets a better-behaved signal.

  of flat out  distance per second over the course's flat-out speed (the asked pace, if it holds)
  km/h         mean speed while moving (over 25 km/h)
  slide        share of moving time with the rear sliding 20 - 80 deg
  yaw p90      deg/s; yaw accel p90 / p99 in deg/s^2 - how violently the car is rotated
  reversals    a minute: the yaw rate going from 60 deg/s one way to 60 the other within 2 s
  lat g p90    sideways acceleration (the recording's G_Lat for a human)
  weaves       a minute (drift_env weave_penalty's count); swings the same for swing_penalty; off course / walls / spinning: share of time

--telemetry adds what the bots did on a server (the plugin's telemetry CSV), by asked pace, in the
same terms (walls: ticks with a wall push-out; no off course or weaving, which need the course
model), so a server test reads straight against the simulator's prediction.
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
RATE_HZ = 20.0
MOVING_KMH = 25.0
REVERSAL_DEGS = 60.0
REVERSAL_WINDOW_S = 2.0


def _smooth(a, n=3):
    return np.convolve(a, np.ones(n) / n, 'same')


def rotation(t, x, z, rear_slip_deg, kmh, lat_g=None):
    """The rotation numbers for one driver's run, from positions, rear slip and speed against time."""
    tu = np.arange(t[0], t[-1], 1.0 / RATE_HZ)
    X, Z, S, V = (np.interp(tu, t, a) for a in (x, z, rear_slip_deg, kmh))
    path = np.unwrap(np.arctan2(np.gradient(Z), np.gradient(X)))
    yaw = np.degrees(np.gradient(_smooth(path) - np.radians(_smooth(S)), 1.0 / RATE_HZ))
    ok = V > MOVING_KMH
    acc = np.abs(np.gradient(yaw, 1.0 / RATE_HZ))
    reversals, i, n, win = 0, 0, len(yaw), int(REVERSAL_WINDOW_S * RATE_HZ)
    while i < n:
        if ok[i] and abs(yaw[i]) > REVERSAL_DEGS:
            sg = np.sign(yaw[i])
            ahead = np.where((np.sign(yaw[i:i + win]) == -sg) & (np.abs(yaw[i:i + win]) > REVERSAL_DEGS))[0]
            if len(ahead):
                reversals += 1
                i += int(ahead[0])
                continue
        i += 1
    out = dict(minutes=ok.sum() / RATE_HZ / 60.0, kmh=V[ok].mean(),
               slide=np.mean((np.abs(S[ok]) >= 20) & (np.abs(S[ok]) < 80)),
               yaw_p90=np.percentile(np.abs(yaw[ok]), 90), acc_p90=np.percentile(acc[ok], 90),
               acc_p99=np.percentile(acc[ok], 99), reversals=reversals)
    if lat_g is not None:
        out['lat_p90'] = np.percentile(np.abs(np.interp(tu, t, lat_g))[ok], 90)
    return out


def pooled(runs):
    """One row from several runs: time-weighted means, percentiles weighted the same way."""
    m = np.array([r['minutes'] for r in runs])
    w = m / max(m.sum(), 1e-9)
    row = {k: float(np.sum(w * np.array([r[k] for r in runs]))) for k in runs[0] if k not in ('minutes', 'reversals')}
    row['minutes'] = float(m.sum())
    row['reversals'] = sum(r['reversals'] for r in runs) / max(m.sum(), 1e-9)
    return row


def recorder_slip_sign(x, z, slip):
    """+1 if the recording's rear slip angle has the simulator's sign, -1 if not, 0 if it cannot
    tell. While sliding, the slip angle and the way the path turns have opposite signs in the
    simulator ~93% of the time (the tail hangs out of the turn); the recording has to agree."""
    turn = np.gradient(np.unwrap(np.arctan2(np.gradient(z), np.gradient(x))))
    m = np.abs(slip) > 20
    if m.sum() < 100:
        return 0
    same = np.mean(np.sign(slip[m]) == np.sign(turn[m]))
    return 1 if same < 0.3 else -1 if same > 0.7 else 0


def human(path):
    """Rotation numbers from an in-game telemetry recording."""
    import pandas as pd
    d = pd.read_csv(path, usecols=['X', 'Z', 'SpeedKMH', 'SlipAngle_RL', 'SlipAngle_RR', 'G_Lat'])
    t = np.arange(len(d)) * 0.006
    slip = ((d.SlipAngle_RL + d.SlipAngle_RR) / 2).values
    sign = recorder_slip_sign(d.X.values[::8], d.Z.values[::8], slip[::8])
    if sign == 0:
        print(f"  {os.path.basename(path)}: can't tell the slip angle's sign (too little sliding?); skipped")
        return None
    row = rotation(t, d.X.values, d.Z.values, sign * slip, d.SpeedKMH.values, d.G_Lat.values)
    row['reversals'] /= max(row['minutes'], 1e-9)
    return row


def telemetry(path):
    """Rotation numbers from a server's plugin telemetry (EnableTelemetryLogging): one row per
    pace the bots were asked for (to the nearest 0.05; the spacing moves it around), lead cars only
    (a tandem follower drives someone else's line). Off course and weaving need the simulator's
    course model, so only the walls (a push-out) and spin recovery are counted."""
    import pandas as pd
    cols = ['SessionId', 'SpawnCounter', 'Tick', 'RealDtMs', 'PosX', 'PosZ', 'SpeedKmh', 'OwnSpeedKmh', 'YawRateDegS',
            'LateralVelocityMs', 'RearSlipAngleDeg', 'PolicyActive', 'CollisionPushOutM', 'WrongWayRecovering']
    head = pd.read_csv(path, nrows=0).columns
    d = pd.read_csv(path, usecols=[c for c in cols + ['FollowLeadId', 'Pace'] if c in head], low_memory=False)
    d = d[d.PolicyActive.astype(str).str.lower().isin(('true', '1', '1.0'))]
    if 'FollowLeadId' in d:
        d = d[d.FollowLeadId < 0]
    d = d.assign(bucket=(d.Pace * 20).round() / 20 if 'Pace' in d else 0.0)
    rows = {}
    for bucket, part in d.groupby('bucket'):
        runs, walls, spin, sign = [], [], [], None
        for _, g in part.groupby(['SessionId', 'SpawnCounter']):
            g = g.sort_values('Tick')
            if len(g) < 200:
                continue
            t = np.cumsum(g.RealDtMs.values) / 1000.0
            if sign is None:
                sign = recorder_slip_sign(g.PosX.values, g.PosZ.values, g.RearSlipAngleDeg.values) or None
            dt = np.maximum(np.gradient(t), 1e-3)
            lat = (np.gradient(g.LateralVelocityMs.values) / dt
                   + g.OwnSpeedKmh.values / 3.6 * np.radians(g.YawRateDegS.values)) / 9.80665
            runs.append((t, g.PosX.values, g.PosZ.values, g.RearSlipAngleDeg.values, g.SpeedKmh.values, lat))
            walls.append(g.CollisionPushOutM.values > 0)
            spin.append(g.WrongWayRecovering.astype(float).values > 0.5)
        if not runs or sign is None:
            continue
        row = pooled([rotation(t, x, z, sign * s, v, lat) for t, x, z, s, v, lat in runs])
        if row['minutes'] < 1.0:
            continue
        row['walls'] = float(np.concatenate(walls).mean())
        row['spin'] = float(np.concatenate(spin).mean())
        rows[float(bucket)] = row
    return rows


def bot(world, policy, pace, grip, ticks, cars, settings, seed=8, skill=None):
    """Server-style runs of one policy at one pace: (rotation row, episode statistics)."""
    import fast_sim as fs
    import drift_env as de
    cfg, never_reset = de.server_run_configs(policy.meta, ticks=ticks, sliding_grip=grip, pace=pace, skill=skill)
    cfg = replace(cfg, plugin_settings={**cfg.plugin_settings, **settings}, recovery_start_fraction=0.0)
    env = de.DriftVecEnv(world, n_envs=cars, env_config=cfg, reward_config=never_reset, seed=seed)
    p = env.base_params
    b = p[fs.P_DIST_CG_REAR_M]
    obs = env.reset()
    rec = {k: [] for k in ('dt', 'x', 'z', 'own', 'latv', 'yaw', 'lat', 'spin')}
    stats = np.zeros((cars, de.EP_SIZE))
    prev_latv = env.states[:, fs.S_LATV].copy()
    for _ in range(ticks):
        obs, _, term, trunc, _ = env.step(policy(obs))
        dt = np.maximum(env.mids[:, de.M_DT], 1e-3)
        s = env.states
        rec['dt'].append(dt.copy())
        for k, i in (('x', fs.S_X), ('z', fs.S_Z), ('own', fs.S_OWN_SPEED), ('latv', fs.S_LATV), ('yaw', fs.S_YAW)):
            rec[k].append(s[:, i].copy())
        rec['lat'].append(((s[:, fs.S_LATV] - prev_latv) / dt + s[:, fs.S_OWN_SPEED] * s[:, fs.S_YAW]) / 9.80665)
        rec['spin'].append(s[:, fs.S_SPIN_RECOVERING] > 0.5)
        prev_latv = s[:, fs.S_LATV].copy()
        done = term | trunc
        stats[done] = env.ep_out[done]
    r = {k: np.array(v)[:-1] for k, v in rec.items()}        # the last tick is the next run's first
    rear_slip = np.degrees(np.arctan2(r['latv'] - b * r['yaw'], np.maximum(r['own'], 0.5)))
    kmh = np.hypot(r['own'], r['latv']) * 3.6
    runs = []
    for i in range(cars):
        t = np.cumsum(r['dt'][:, i])
        runs.append(rotation(t, r['x'][:, i], r['z'][:, i], rear_slip[:, i], kmh[:, i], r['lat'][:, i]))
    row = pooled(runs)
    row['spin'] = float(r['spin'].mean())
    return row, de.finalize_episode_stats(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', nargs='*')
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=[])
    ap.add_argument('--human', action='append', default=[], help="an in-game telemetry recording (CSV)")
    ap.add_argument('--telemetry', action='append', default=[],
                    help='a server\'s plugin telemetry CSV (drift_physics_telemetry_*.csv): the bots it logged')
    ap.add_argument('--paces', type=float, nargs='+', default=[0.85, 1.0])
    ap.add_argument('--pace-reference', default=None, help='measure each course\'s flat-out speed with this policy')
    ap.add_argument('--skill', type=float, default=None,
                    help="the skill input (the plugin's BotSkill) for a policy trained with one; default full skill")
    ap.add_argument('--grip', type=float, default=0.55, help='RearSlidingGrip')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='a plugin setting for every policy')
    ap.add_argument('--cars', type=int, default=16)
    ap.add_argument('--ticks', type=int, default=3300, help='per run (3300 is about 3 minutes)')
    args = ap.parse_args()
    sys.path.insert(0, _HERE)
    import fast_sim as fs
    import drift_env as de
    from policy import Policy

    settings = {n: float(v) for n, v in (x.split('=', 1) for x in args.set)}
    head = (f"{'':28s} {'pace':>5s} {'of flat out':>11s} {'km/h':>5s} {'slide':>6s} {'yaw p90':>7s} {'yaw accel p90/p99':>17s} "
            f"{'reversals':>9s} {'lat g p90':>9s} {'weaves':>6s} {'swings':>6s} {'off course':>10s} {'walls':>6s} {'spinning':>8s}")

    def line(name, pace, flat, row, st=None):
        frac = f"{st['progress_m'].sum() / st['sim_seconds'].sum() * 3.6 / flat:11.2f}" if st is not None else f"{'':11s}"
        tail = (f" {st['weaves_per_min'].mean():6.1f} {st['swings_per_min'].mean():6.1f} {st['offcourse_fraction'].mean():10.1%} "
                f"{st['pushout_tick_fraction'].mean():6.1%} {row['spin']:8.1%}") if st is not None else ''
        print(f"{name[:28]:28s} {pace:>5s} {frac} {row['kmh']:5.1f} {row['slide']:6.0%} {row['yaw_p90']:7.0f} "
              f"{row['acc_p90']:8.0f} / {row['acc_p99']:<6.0f} {row['reversals']:9.1f} {row.get('lat_p90', np.nan):9.2f}" + tail,
              flush=True)

    if args.human:
        print(head)
    for path in args.human:
        row = human(path)
        if row:
            line('human ' + os.path.basename(path)[-19:-4], '', 1.0, row)
    for path in args.telemetry:
        rows = telemetry(path)
        print(f"\nserver: {os.path.basename(path)} (off course and weaving are simulator-only)")
        print(head)
        for pace, row in sorted(rows.items()):
            print(f"{'server':28s} {pace:5.2f} {'':11s} {row['kmh']:5.1f} {row['slide']:6.0%} {row['yaw_p90']:7.0f} "
                  f"{row['acc_p90']:8.0f} / {row['acc_p99']:<6.0f} {row['reversals']:9.1f} {row['lat_p90']:9.2f} "
                  f"{'':6s} {'':6s} {'':10s} {row['walls']:6.1%} {row['spin']:8.1%}   ({row['minutes']:.0f} min)")
        if not rows:
            print('  no lead-car runs with the policy driving')
    policies = [(p, Policy.load_json(p)) for p in args.policies]
    for course in args.track:
        world = fs.load_world(args.tracks, track=course)
        print(f"\n{course}")
        print(head)
        refs = {}
        for name, pol in policies:
            physics = {**(pol.meta.get('plugin_settings') or {}), **settings}
            key = tuple(sorted((k, v) for k, v in physics.items() if k in de.PHYSICS_SETTINGS))
            if key not in refs:
                if args.pace_reference:
                    ref = Policy.load_json(args.pace_reference)
                    ref.meta = dict(ref.meta, plugin_settings={**{k: v for k, v in (ref.meta.get('plugin_settings') or {}).items()
                                                                 if k not in de.PHYSICS_SETTINGS}, **dict(key)})
                    refs[key] = de.pace_reference(world, ref, sliding_grip=args.grip)
                else:
                    refs[key] = de.course_pace_reference(args.tracks, course, world)
            world.pace_ref = refs[key]
            flat = float(world.sp_length.sum() / np.sum(world.sp_length / (world.pace_ref / 3.6)) * 3.6)
            label = os.path.basename(os.path.dirname(name)) or name
            if dict(key).get('EnableTireLoadSensitivity'):
                label += ' (tire load)'
            for pace in args.paces:
                row, st = bot(world, pol, pace, args.grip, args.ticks, args.cars, settings, skill=args.skill)
                line(label, f'{pace:g}', flat, row, st)
            print(f"{'':28s} flat out {flat:.1f} km/h")


if __name__ == '__main__':
    main()
