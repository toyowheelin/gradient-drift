"""Writes the cases the plugin checker (tests/plugin/Program.cs) runs against the C# plugin:

  policies/<name>.json   the shipped policy and small made-up ones covering every network shape the
                         plugin runs (two outputs, a residual chase network, a history block, deeper
                         layers), plus broken or unrunnable files the plugin must refuse
  cases.json             for each policy: whether the plugin must load it or refuse it, and for the
                         ones it loads, observations with the actions the Python policy (drift_rl/
                         policy.py, what the trainer and every tool here run) gives for them

    python tests/plugin/make_cases.py <out folder>

The C# side must give the same actions to within ACTION_TOLERANCE (it works in float32), or, for
a policy whose network float32 itself moves further than that on these inputs, twice what the same
network gives in float32 numpy (a case's own action_tolerance).
"""
import copy
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'drift_rl'))
from policy import MLP, Policy  # noqa: E402

SHIPPED = os.path.join(ROOT, 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json')
ACTION_TOLERANCE = 2e-4
OBS_PER_POLICY = 64

# the plugin's input counts (DriftPolicy.cs ObservationSize*, ChaseInputs.Slot / Count)
OBS_V1 = 46
OBS_CHASE_SLOT = 67
OBS_V7 = OBS_CHASE_SLOT + 13 + 10 + 4
OBS_FULL = OBS_V7 + 6


def made_up(obs_size, hidden, actions, seed, **extra):
    """A policy file with random weights (orthogonal, as training starts) and normalisation."""
    rng = np.random.default_rng(seed)
    mlp = MLP([obs_size] + list(hidden) + [actions], rng=rng, out_gain=1.5)
    for b in mlp.b:
        b[:] = rng.normal(0.0, 0.2, b.shape)
    meta = {'track': 'testtrack', 'control_mode': 'direct', 'lookahead_m': [5, 10, 15, 20, 30],
            'min_radius_window_m': 40.0, 'trained_ticks': 1000}
    if obs_size > 55:
        meta['cars'] = {'drift_steer_lock_deg': 50.0}
    meta.update(extra)
    return Policy(mlp, rng.normal(0.0, 3.0, obs_size), rng.uniform(0.2, 4.0, obs_size), meta)


def with_residual(pol, seed):
    rng = np.random.default_rng(seed)
    n = len(pol.obs_mean)
    res = MLP([n] + [len(b) for b in pol.mlp.b], rng=rng, out_gain=0.5)
    return Policy(pol.mlp, pol.obs_mean, pol.obs_std, pol.meta, res, OBS_CHASE_SLOT, 0.5)


def observations(pol, seed, gate=None):
    """Inputs around the policy's normalisation, some far enough out to be clipped."""
    rng = np.random.default_rng(seed)
    n = len(pol.obs_mean)
    x = pol.obs_mean + pol.obs_std * rng.normal(0.0, 1.5, (OBS_PER_POLICY, n))
    x[::8] = pol.obs_mean + pol.obs_std * rng.choice([-30.0, 30.0], (len(x[::8]), n))   # past the clip
    if gate is not None:
        x[:, gate] = np.where(np.arange(len(x)) % 2 == 0, 1.0, 0.0)       # the residual on, then off
    return x


def plugin_actions(pol, obs):
    """What the plugin must give: the Python policy's actions, the brake held to 0 - 1 (the plugin's
    range for it), 0 for a two-output policy."""
    meta = pol.meta
    pol.meta = {k: v for k, v in meta.items() if k != 'history'}   # the inputs already hold the history
    a = pol(obs)
    pol.meta = meta
    if a.shape[1] < 3:
        a = np.concatenate([a, np.zeros((len(a), 3 - a.shape[1]))], axis=1)
    a[:, 2] = np.clip(a[:, 2], 0.0, 1.0)
    return a


def float32_spread(pol, obs):
    """How far the policy's actions move when its arithmetic is float32, as the plugin's is."""
    p32 = copy.deepcopy(pol)
    for net in (p32.mlp, p32.residual):
        if net is not None:
            net.W = [w.astype(np.float32) for w in net.W]
            net.b = [b.astype(np.float32) for b in net.b]
    p32.obs_mean = p32.obs_mean.astype(np.float32)
    p32.obs_std = p32.obs_std.astype(np.float32)
    return float(np.abs(plugin_actions(p32, obs.astype(np.float32)) - plugin_actions(pol, obs)).max())


def contact_cases(n=48, seed=5):
    """Car-on-car pushes (CarContact.Collide in the plugin) against the trainer's
    (drift_env._touch_push): a car overlapping a lead at random places, speeds and yaw rates. Each:
    the car, the lead's place and velocity, and what the push leaves the car with plus the yaw-rate
    kick it gives the lead (deg/s)."""
    import drift_env as de
    import fast_sim as fs
    p = fs.make_params()
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = np.zeros(fs.STATE_SIZE)
        s[fs.S_HEADING] = rng.uniform(-math.pi, math.pi)
        s[fs.S_OWN_SPEED] = rng.uniform(5.0, 25.0)
        s[fs.S_LATV] = rng.uniform(-8.0, 8.0)
        s[fs.S_YAW] = rng.uniform(-1.5, 1.5)
        lh = s[fs.S_HEADING] + rng.uniform(-1.0, 1.0)
        ang = s[fs.S_HEADING] + rng.uniform(-math.pi, math.pi)
        r = rng.uniform(1.5, 4.0)
        lx, lz = r * math.cos(ang), r * math.sin(ang)
        lead_v = rng.uniform(5.0, 25.0)
        lvx, lvz = lead_v * math.cos(lh) + rng.normal(0, 2), lead_v * math.sin(lh) + rng.normal(0, 2)
        if not de._cars_overlap(0.0, 0.0, s[fs.S_HEADING], lx, lz, lh):
            continue
        car = [0.0, 0.0, float(s[fs.S_HEADING]), float(s[fs.S_OWN_SPEED]), float(s[fs.S_LATV]), float(s[fs.S_YAW])]
        c = np.zeros(de.N_COUNTERS)
        kick = de._touch_push(s, p, lx, lz, lh, lvx, lvz, c)
        out.append({'car': car, 'lead': [lx, lz, lh, lvx, lvz],
                    'mass': float((p[fs.P_FRONT_STATIC_LOAD_N] + p[fs.P_REAR_STATIC_LOAD_N]) / 9.80665),
                    'inertia': float(p[fs.P_YAW_INERTIA_KGM2]),
                    'after': [float(s[fs.S_OWN_SPEED]), float(s[fs.S_LATV]), float(s[fs.S_YAW]), float(kick)]})
    return out


def zone_case(out, n=200):
    """A course edges file and a zones file for Program.cs's round 200-point line (one zone across the
    line's start), and the zone side per point the trainer reads from them (fast_sim.zone_variants,
    the layout a server's bots see)."""
    import fast_sim as fs
    zones = [(-1, 20, 41), (1, 90, 130), (-1, 185, 12), (1, 150, 150)]
    with open(os.path.join(out, 'case_course_edges.csv'), 'w') as f:
        f.write('PointId,PositiveSideM,NegativeSideM\n')
        for i in range(n):
            f.write(f'{i},6.0,7.0\n')
    with open(os.path.join(out, 'case_course_zones.csv'), 'w') as f:
        f.write('Zone,Kind,Side,FirstPointId,LastPointId,LengthM,Corner,CornerTurnsDeg,CornerRadiusM\n')
        for k, (side, a, b) in enumerate(zones, 1):
            f.write(f'{k},outside zone,{side},{a},{b},10.0,{k},120,20\n')
    side = fs.zone_variants(np.array(zones, dtype=float), n, 1.57)[:, 0]
    return {'edges': 'case_course_edges.csv', 'zones': 'case_course_zones.csv', 'side': side.astype(int).tolist()}


def main():
    out = sys.argv[1]
    folder = os.path.join(out, 'policies')
    os.makedirs(folder, exist_ok=True)
    cases = {'action_tolerance': ACTION_TOLERANCE, 'policies': []}

    def add(name, pol=None, path=None, expect='loads', why='', gate=None, seed=0):
        target = os.path.join(folder, name + '.json')
        if path:
            shutil.copyfile(path, target)
            pol = Policy.load_json(path)
        else:
            pol.save_json(target)
        case = {'file': f'policies/{name}.json', 'expect': expect, 'why': why}
        if expect == 'loads':
            obs = observations(pol, seed, gate)
            case['obs'] = obs.tolist()
            case['actions'] = plugin_actions(pol, obs).tolist()
            spread = float32_spread(pol, obs)
            if 2 * spread > ACTION_TOLERANCE:
                case['action_tolerance'] = 2 * spread
        cases['policies'].append(case)

    def add_doc(name, doc, why):
        with open(os.path.join(folder, name + '.json'), 'w') as f:
            json.dump(doc, f)
        cases['policies'].append({'file': f'policies/{name}.json', 'expect': 'refused', 'why': why})

    # --- what the plugin must run, and run the same as Python ---
    add('shipped', path=SHIPPED, seed=1, why='the policy the plugin ships with')
    add('two_outputs', made_up(OBS_V1, (64, 64), 2, 2), seed=2, why='an early policy: 46 inputs, no brake output')
    add('deep', made_up(OBS_CHASE_SLOT, (256, 256, 128, 64), 3, 3, plugin_settings={
        'RoundTireCurve': 1.0, 'RearSpinGripShare': 0.95, 'SurfaceGrip': 1.0, 'CarOutlineContact': 1.0}),
        seed=3, why='four hidden layers with the newer model settings')
    add('residual', with_residual(made_up(OBS_FULL, (128, 128), 3, 4, chase={}), 5), gate=OBS_CHASE_SLOT, seed=4,
        why='a chase residual network, gated by the car-ahead input')
    hist = made_up(OBS_FULL + 3 * 4, (128, 128), 3, 6, chase={},
                   history={'slots': [9, 10, 11, 12], 'frames': 3, 'every': 2, 'reset_slot': 0})
    add('history', hist, seed=6, why='a history block (3 frames of 4 inputs)')
    hist7 = made_up(OBS_V7 + 3 * 4, (128, 128), 3, 7, chase={},
                    history={'slots': [9, 10, 11, 12], 'frames': 3, 'every': 2, 'reset_slot': 0})
    add('history_before_zones', hist7, seed=7, why='a history block after the 94 inputs from before the zone inputs')

    # --- what the plugin must refuse (it then leaves the cars to the stock AI) ---
    good = json.load(open(os.path.join(folder, 'shipped.json')))

    def broken(name, why, change):
        doc = copy.deepcopy(good)
        change(doc)
        add_doc(name, doc, why)

    broken('bad_format', 'an unknown file format', lambda d: d.update(format='drift-policy-v0'))
    broken('bad_obs_size', 'an input count no build had', lambda d: (
        d.update(obs_size=60, obs_mean=d['obs_mean'][:60], obs_std=d['obs_std'][:60]),
        d['layers'][0].update(W=[row[:60] for row in d['layers'][0]['W']])))
    broken('zero_std', 'a zero normalisation scale', lambda d: d['obs_std'].__setitem__(5, 0.0))
    broken('short_mean', 'obs_mean of the wrong length', lambda d: d['obs_mean'].pop())
    broken('ragged_layer', 'a weight matrix of the wrong size', lambda d: d['layers'][1]['W'].pop())
    broken('wrong_outputs', 'a last layer without the actions', lambda d: (
        d['layers'][-1].update(W=d['layers'][-1]['W'][:1], b=d['layers'][-1]['b'][:1])))
    broken('residual_mode', 'a residual-control policy (the old kind)', lambda d: d.update(control_mode='residual'))
    broken('no_steer_lock', 'reads the car inputs but has no cars.drift_steer_lock_deg', lambda d: d.pop('cars'))
    broken('bad_pace_range', 'a pace range upside down', lambda d: d.update(pace_fraction_range=[1.1, 0.8]))
    broken('bad_skill_range', 'a skill range past 1', lambda d: d.update(skill_range=[0.0, 1.5]))
    broken('unknown_setting', 'a plugin setting this build does not have',
           lambda d: d['plugin_settings'].update(EnableWeightTransfer=1.0))
    broken('setting_out_of_range', 'a known setting at a value the plugin cannot take',
           lambda d: d['plugin_settings'].update(MaxSteerAngleDeg=120.0))
    broken('huge_history', 'a history block past the plugin\'s limit', lambda d: d.update(
        history={'slots': list(range(60)), 'frames': 5, 'every': 1, 'reset_slot': 0}))
    with open(os.path.join(folder, 'not_json.json'), 'w') as f:
        f.write('{"format": "drift-policy-v1", "obs_size": ')
    cases['policies'].append({'file': 'policies/not_json.json', 'expect': 'refused', 'why': 'a cut-off file'})

    cases['contacts'] = contact_cases()
    cases['zones'] = zone_case(out)
    with open(os.path.join(out, 'cases.json'), 'w') as f:
        json.dump(cases, f)
    loads = sum(c['expect'] == 'loads' for c in cases['policies'])
    print(f"{len(cases['policies'])} policy files ({loads} to load and match, the rest to refuse) in {out}")


if __name__ == '__main__':
    main()
