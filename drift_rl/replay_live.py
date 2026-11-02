"""
replay_live.py - one-tick-at-a-time replay of a real server run through fast_sim.

For every telemetry row t: rebuild the simulator's state from row t-1 (plus the PolicyObs the
plugin logged at t, which carries last tick's own-speed, yaw rate, lateral velocity, rays, pins
exactly), step ONE tick with the real tick length and the real policy actions, and compare with
row t. Errors cannot compound, so any field that disagrees points at a specific mechanism where
the simulator and the plugin differ.

    python replay_live.py drift_physics_telemetry_<date>_<time>.csv --track <track> --cars-dir <plugin>/cars

The plugin logs every bot into one file (EnableTelemetry), rows interleaved; each bot is followed by
its SessionId, replayed with the car file whose car inputs match the ones it logged, and with the
state the telemetry does not log carried from its own previous tick. A policy with a history block
logs a longer PolicyObs row; its first inputs are the ones read here. The table "along the lap"
shows where on the line the two part, e.g. a surface the simulator gives the road's grip.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import fast_sim as fs
import drift_env as de
import check_plugin_policy as cp


def car_files(folder):
    """{name: fast_sim parameter values} of the car files in a plugin's cars/ folder (setup writes
    <car model>.json there; the plugin drives each bot with its model's)."""
    import json
    out = {}
    for name in sorted(os.listdir(folder)) if folder else ():
        if name.endswith('.json'):
            with open(os.path.join(folder, name)) as f:
                v = json.load(f)
            out[name[:-5]] = {k: float(x) for k, x in v.items() if k in fs.P and isinstance(x, (int, float))}
    return out


def car_for(logged, cars, base, plugin_settings):
    """(name, parameter vector, worst car-input error) of the car file whose car inputs (OBS 55-65,
    drift_env.car_inputs) best match a bot's logged ones; ('default', base, error) without car files."""
    best = ('default', base, np.inf)
    for name, values in [('default', {})] + list(cars.items()):
        p = base.copy()
        for k, v in values.items():
            p[fs.P[k]] = v
        de.apply_plugin_settings(p, plugin_settings)
        mine = np.zeros(de.N_CAR_OBS)
        de.car_inputs(p, mine, 0)
        err = float(np.nanmax(np.abs(mine - logged))) if np.isfinite(logged).any() else np.nan
        if not np.isfinite(best[2]) or err < best[2]:
            best = (name, p, err)
    return best


def replay(df, world, params_override=None, plugin_settings=None, cars=None, session=None):
    """One-tick predictions for every row whose car's previous tick (same SessionId, same
    SpawnCounter, Tick - 1) is in the file. The plugin logs every bot in one file, rows interleaved,
    so each car is followed by its SessionId. State the telemetry does not log (load shift, line
    counters, surface grip, pin surfaces ...) is carried over from the same car's previous replayed
    tick; the rest is rebuilt from the log each tick, so errors cannot compound."""
    W = world.arrays()
    base = fs.make_params()
    de.apply_plugin_settings(base, plugin_settings)      # the steering lock etc. the policy file carries
    # the height model of the plugin build that wrote this file (ground ray or the older blend);
    # an explicit params_override still wins, for "what would the other one have done" replays
    for k, v in cp.plugin_overrides_for(df).items():
        base[fs.P[k]] = v
    if params_override:
        for k, v in params_override.items():
            base[fs.P[k]] = v
    obs = cp.parse_vectors(df['PolicyObs'], de.OBS_SIZE)
    rays = cp.parse_vectors(df['RayDistances'], fs.N_RAYS)
    pins = cp.parse_vectors(df['PinDistances'], fs.N_PINS)
    n = len(df)
    col = {c: df[c].to_numpy() for c in df.columns if df[c].dtype != object}
    sid = col['SessionId'] if 'SessionId' in col else np.zeros(n)
    prev_row = np.full(n, -1)
    last = {}
    for t in range(n):
        key = (sid[t], col['SpawnCounter'][t])
        q = last.get(key)
        if q is not None and col['Tick'][t] - col['Tick'][q] == 1:
            prev_row[t] = q
        last[key] = t
    next_row = np.full(n, -1)
    next_row[prev_row[prev_row >= 0]] = np.nonzero(prev_row >= 0)[0]

    out = {k: np.full(n, np.nan) for k in (
        'heading_deg', 'yaw_degs', 'latv', 'x', 'z', 'speed_kmh', 'steer_deg', 'cte', 'spin', 'pushout',
        'accel', 'front_slip', 'rear_slip', 'nearest', 'np_glitch', 'glitch', 'y', 'ray_err', 'pin_err', 'own_kmh',
        'own_real_kmh')}
    out['session'] = sid
    params = {}
    car_names = {}
    carry = {}
    diag = np.zeros(fs.DIAG_SIZE)
    mid = np.zeros(fs.MID_SIZE)
    first_obs = de.CAR_OBS_SLOT + de.N_CAR_OBS <= obs.shape[1]
    for t in range(1, n):
        if session is not None and sid[t] != session:
            continue
        q = prev_row[t]
        if q < 0 or not np.isfinite(obs[t, :de.OBS_SIZE_V1]).all() or obs[t, 45] > 0.5:
            continue
        if sid[t] not in params:
            logged = (np.nanmedian(obs[sid == sid[t], de.CAR_OBS_SLOT:de.CAR_OBS_SLOT + de.N_CAR_OBS], axis=0)
                      if first_obs else np.full(de.N_CAR_OBS, np.nan))
            name, params[sid[t]], err = car_for(logged, cars or {}, base, plugin_settings)
            car_names[sid[t]] = (name, err)
        p = params[sid[t]]
        c = carry.get(sid[t])
        if c is not None and c[0] == q:
            s = c[1].copy()
            s[fs.S_X], s[fs.S_Y], s[fs.S_Z] = col['PosX'][q], col['PosY'][q], col['PosZ'][q]
            s[fs.S_HEADING] = math.radians(col['RealHeadingDeg'][q])
            s[fs.S_MAX_SPEED] = col['CeilingKmh'][t] / 3.6
        else:
            s = fs.initial_state((col['PosX'][q], col['PosY'][q], col['PosZ'][q]),
                                 math.radians(col['RealHeadingDeg'][q]), col['CeilingKmh'][t] / 3.6)
        s[fs.S_YAW] = obs[t, 2] * 2.0
        s[fs.S_LATV] = obs[t, 1] * 10.0
        s[fs.S_OWN_SPEED] = obs[t, 0] * 25.0
        s[fs.S_CORR_ANGLE] = col['CorrectedSplineAngle'][q]
        s[fs.S_PIN0:fs.S_PIN0 + fs.N_PINS] = obs[t, 29:35] * 12.0
        s[fs.S_RAY0:fs.S_RAY0 + fs.N_RAYS] = obs[t, 13:28] * 60.0
        s[fs.S_SPIN_RECOVERING] = obs[t, 40]
        s[fs.S_SMOOTH_H] = col['PosY'][q]       # _smoothedHeight is what the plugin wrote into position.Y
        s[fs.S_REAR_SLIDE_GRIP] = 1.0 - 0.5 * obs[t, 35]    # the plugin's RearSlidingGrip, as the policy was shown it
        if 'Pace' in col and np.isfinite(col['Pace'][t]):
            s[fs.S_PACE] = col['Pace'][t]
        # last tick's steering command is model state once the rear thrust limit is on (it sets
        # the front tires' pull along the car); the policy was shown it as a fraction of lock
        s[fs.S_LAST_STEER] = obs[t, 41] * p[fs.P_MAX_STEER_RAD]
        brake = col['PolicyBrakeAction'][t] if 'PolicyBrakeAction' in col else 0.0   # older builds: none
        fs.step(s, col['PolicySteerAction'][t], col['PolicySpeedAction'][t], brake, col['RealDtMs'][t] / 1000.0,
                p, diag, mid, *W)
        carry[sid[t]] = (t, s.copy())
        out['y'][t] = s[fs.S_Y]
        out['own_kmh'][t] = s[fs.S_OWN_SPEED] * 3.6
        if 'OwnSpeedKmh' in col:
            out['own_real_kmh'][t] = col['OwnSpeedKmh'][t]          # the direct-only plugin logs it
        else:
            nxt = next_row[t]
            if nxt >= 0 and np.isfinite(obs[nxt, 0]):
                out['own_real_kmh'][t] = obs[nxt, 0] * 25.0 * 3.6     # the NEXT row's observation carries it
        out['ray_err'][t] = np.abs(s[fs.S_RAY0:fs.S_RAY0 + fs.N_RAYS] - rays[t]).max()
        out['pin_err'][t] = np.abs(s[fs.S_PIN0:fs.S_PIN0 + fs.N_PINS] - pins[t]).max()
        out['nearest'][t] = diag[fs.D_NearestPointId]
        out['np_glitch'][t] = diag[fs.D_NearestPointGlitchCorrected]
        out['glitch'][t] = diag[fs.D_GlitchCorrected]
        out['heading_deg'][t] = math.degrees(s[fs.S_HEADING])
        out['yaw_degs'][t] = math.degrees(s[fs.S_YAW])
        out['latv'][t] = s[fs.S_LATV]
        out['x'][t], out['z'][t] = s[fs.S_X], s[fs.S_Z]
        out['speed_kmh'][t] = math.hypot(diag[fs.D_BodyForwardSpeedMs], s[fs.S_LATV]) * 3.6
        out['steer_deg'][t] = diag[fs.D_SteerCommandDeg]
        out['cte'][t] = diag[fs.D_CrossTrackErrorM]
        out['spin'][t] = diag[fs.D_UncontrolledSpinRecovering]
        out['pushout'][t] = diag[fs.D_CollisionPushOutM]
        out['accel'][t] = diag[fs.D_RealizedLongitudinalAccelMs2]
        out['front_slip'][t] = diag[fs.D_FrontSlipAngleDeg]
        out['rear_slip'][t] = diag[fs.D_RearSlipAngleDeg]
    out['cars'] = car_names
    return out


def report(df, out):
    def wrap(x):
        return (x + 180.0) % 360.0 - 180.0
    def col(name):
        return df[name] if name in df.columns else None
    rows = [
        ('cross-track error, m', out['cte'], col('CrossTrackErrorM'), None),
        ('longitudinal accel, m/s2', out['accel'], col('RealizedLongitudinalAccelMs2'), None),
        ('steer command, deg', out['steer_deg'], col('SteerCommandDeg'), None),
        ('front slip angle, deg', out['front_slip'], col('FrontSlipAngleDeg'), None),
        ('rear slip angle, deg', out['rear_slip'], col('RearSlipAngleDeg'), None),
        ('yaw rate, deg/s', out['yaw_degs'], col('YawRateDegS'), None),
        ('lateral velocity, m/s', out['latv'], col('LateralVelocityMs'), None),
        ('heading, deg', out['heading_deg'], col('RealHeadingDeg'), wrap),
        ('position X, m', out['x'], col('PosX'), None),
        ('position Z, m', out['z'], col('PosZ'), None),
        ('own (forward) speed, km/h', out['own_kmh'], out['own_real_kmh'], None),
        ('logged SpeedKmh *, km/h', out['speed_kmh'], col('SpeedKmh'), None),
        ('wall push-out, m', out['pushout'], col('CollisionPushOutM'), None),
        ('height Y, m', out['y'], col('PosY'), None),
        ('worst fan ray, m', out['ray_err'], np.zeros(len(df)), None),
        ('worst proximity pin, m', out['pin_err'], np.zeros(len(df)), None),
    ]
    rows = [r for r in rows if r[2] is not None]
    if 'NearestPointId' in df.columns:
        real_nearest = np.asarray(df['NearestPointId'], dtype=float)
        ok = np.isfinite(out['nearest'])
        print(f"nearest spline point: disagrees on {int((out['nearest'][ok] != real_nearest[ok]).sum())} of {ok.sum()} ticks")
    print(f"{'one-tick prediction vs real':28s}{'n':>6s}{'median':>10s}{'p90':>10s}{'p99':>10s}{'max':>10s}")
    for name, pred, real, f in rows:
        real = np.asarray(real, dtype=float)
        ok = np.isfinite(pred) & np.isfinite(real)
        e = pred[ok] - real[ok]
        e = np.abs(f(e) if f else e)
        print(f"{name:28s}{ok.sum():6d}{np.median(e):10.4f}{np.quantile(e, 0.9):10.4f}{np.quantile(e, 0.99):10.4f}{e.max():10.3f}")
    ok = np.isfinite(out['ray_err'])
    print(f"ticks with a fan ray off by more than 0.6 m: {np.mean(out['ray_err'][ok] > 0.6):.1%}; "
          f"with a pin off by more than 0.3 m: {np.mean(out['pin_err'][ok] > 0.3):.1%}")
    ok = np.isfinite(out['spin'])
    for name, pred, real in (('angle-glitch flag', out['glitch'], col('AngleGlitchCorrected')),
                             ('nearest-point-glitch flag', out['np_glitch'], col('NearestPointGlitchCorrected'))):
        if real is None:
            continue
        real = np.asarray(real, dtype=float)
        print(f"{name}: real {int(real[ok].sum())}, simulator {int(pred[ok].sum())}, disagree on {int((pred[ok] != real[ok]).sum())} ticks")
    real_spin = np.asarray(df['UncontrolledSpinRecovering'], dtype=float)
    print(f"spin-recovery flag mismatches: {int((out['spin'][ok] != real_spin[ok]).sum())} of {ok.sum()}")
    sessions = out.get('session')
    if sessions is not None and len(out.get('cars') or {}) > 1:
        print(f"\nper car (SessionId, car file matched by its car inputs, worst car-input error; "
              f"median one-tick error of yaw rate deg/s, lateral velocity m/s, position m)")
        for s, (name, err) in sorted(out['cars'].items()):
            m = (sessions == s) & np.isfinite(out['yaw_degs'])
            def med(pred, real):
                e = np.abs(pred[m] - np.asarray(real, dtype=float)[m])
                return float(np.median(e)) if len(e) else np.nan
            pos = np.hypot(out['x'][m] - df['PosX'].to_numpy()[m], out['z'][m] - df['PosZ'].to_numpy()[m])
            print(f"  {int(s):4d} {name:28s} {err:7.4f}  ticks {int(m.sum()):6d}  yaw {med(out['yaw_degs'], df['YawRateDegS']):7.3f}"
                  f"  latv {med(out['latv'], df['LateralVelocityMs']):6.3f}  pos {float(np.median(pos)) if len(pos) else np.nan:6.3f}")
    if 'NearestPointId' in df.columns:
        # where along the line the model and the server part: a stretch whose errors stand out is a
        # place (a surface, a kerb, a wall) the simulator does not see as the server does
        point = np.asarray(df['NearestPointId'], dtype=float)
        ok = np.isfinite(out['yaw_degs']) & np.isfinite(point)
        if ok.sum() > 200:
            n_points = max(int(np.nanmax(point)) + 1, 1)
            share = point[ok] / n_points
            e_yaw = np.abs(out['yaw_degs'][ok] - np.asarray(df['YawRateDegS'], dtype=float)[ok])
            e_lat = np.abs(out['latv'][ok] - np.asarray(df['LateralVelocityMs'], dtype=float)[ok])
            print("\nalong the lap (share of the AI line; median one-tick error, then the worst tenth's)")
            print(f"  {'lap':>11s}{'ticks':>7s}{'yaw deg/s':>11s}{'p90':>8s}{'latv m/s':>10s}{'p90':>8s}")
            for b in range(20):
                m = (share >= b / 20) & (share < (b + 1) / 20)
                if m.sum() < 20:
                    continue
                print(f"  {b / 20:5.2f}-{(b + 1) / 20:4.2f}{int(m.sum()):7d}{np.median(e_yaw[m]):11.3f}{np.quantile(e_yaw[m], 0.9):8.3f}"
                      f"{np.median(e_lat[m]):10.4f}{np.quantile(e_lat[m], 0.9):8.4f}")
    print("* SpeedKmh is the velocity the plugin reports to clients, not state: on wall-contact ticks the plugin\n"
          "  removes the into-the-wall part of it first (GradientDriftPlugin.cs, collision response), so p99/max are\n"
          "  large here by construction. The state behind it (own speed, lateral velocity, position, heading)\n"
          "  is compared in the rows above.")


def policy_settings(policy_path=None):
    """(path, plugin settings) of the policy the server ran: the steering lock and the other
    settings it carries are in the policy file, not in the telemetry. Default: the plugin source
    folder's GradientDriftPlugin.policy.json, as check_plugin_policy.py does."""
    from policy import Policy, default_policy_path
    if policy_path is None:
        policy_path = default_policy_path()
    meta = Policy.load_json(policy_path).meta
    de.require_direct(meta)
    return os.path.abspath(policy_path), dict(meta.get('plugin_settings') or {})


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('telemetry')
    ap.add_argument('--policy', default=None,
                    help='the policy file the server ran (for the plugin settings it carries). Default: the one '
                         'in the plugin source folder')
    import track_files as tf
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None, help="the track the server ran (default: the policy's own)")
    ap.add_argument('--cars-dir', default=None,
                    help="the plugin's cars/ folder (setup's car files): each bot is replayed with the file whose "
                         "car inputs match its logged ones; without it, the default car")
    ap.add_argument('--session', type=int, default=None, help='replay only this SessionId')
    args = ap.parse_args()
    used, plugin_settings = policy_settings(args.policy)
    print(f"policy: {used}")
    if plugin_settings:
        print("plugin settings from the policy file: " + ", ".join(f"{k}={v:g}" for k, v in plugin_settings.items()))
    df = cp.load_telemetry(args.telemetry)
    from policy import Policy
    track = args.track or Policy.load_json(used).meta.get('track') or tf.TRACK
    print(f"track: {track}")
    world = fs.load_world(args.tracks, track=track)
    print(f"car height in this run: {'ground ray' if cp.plugin_overrides_for(df)['EXT_GROUND_RAY_HEIGHT'] else 'spline / road-mesh blend'}")
    out = replay(df, world, plugin_settings=plugin_settings, cars=car_files(args.cars_dir), session=args.session)
    for s, (name, err) in sorted(out['cars'].items()):
        if np.isfinite(err) and err > 0.01:
            print(f"WARNING: SessionId {int(s)}: no car file matches its logged car inputs (closest {name}, off by "
                  f"{err:.3f}); its replay uses the wrong car. Give --cars-dir the server's plugin cars/ folder.")
    report(df, out)
