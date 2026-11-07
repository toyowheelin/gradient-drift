"""Golden physics traces: the simulator (drift_rl/fast_sim.py) driven with fixed inputs, no policy,
on a generated course, and the car's path saved. test_sim_golden.py replays them and fails when a
change moves a path further than the tolerances there, which catches a physics change nobody meant
to make (a tyre tweak that quietly changes every car).

When a physics change IS meant, write the traces again and commit them with it:

    python tests/sim_traces.py --write          (from the repository root)

and say in the PR why they moved.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'drift_rl'))

GOLDEN = os.path.join(HERE, 'golden', 'sim_traces.json')
COURSE = (1, 0)             # generated course (seed, number) the traces run on
DT_S = 0.05                 # one plugin tick at its usual rate
MAX_SPEED_KMH = 120.0
EVERY = 5                   # ticks between saved samples

# Physics variants: the plugin's own defaults, the settings the shipped policy's file turns on, and
# every newer model switch at once (the combined-technique policies' settings).
VARIANTS = {
    'defaults': {},
    'shipped': {
        'MaxSteerAngleDeg': 50.0, 'EnableRearThrustLimit': 1.0, 'MaxSteerRateDegS': 150.0,
        'EnableRearBrakeLimit': 1.0, 'EnableRearFrictionCircle': 1.0, 'EnableTireLoadSensitivity': 1.0,
        'ProximityPinHeightOffsetMeters': -0.28, 'FrontBrakeOnly': 1.0,
    },
    'newer': {
        'MaxSteerAngleDeg': 50.0, 'EnableRearThrustLimit': 1.0, 'MaxSteerRateDegS': 150.0,
        'EnableRearBrakeLimit': 1.0, 'EnableRearFrictionCircle': 1.0, 'EnableTireLoadSensitivity': 1.0,
        'FrontBrakeOnly': 1.0, 'CarOutlineContact': 1.0, 'RoundTireCurve': 1.0, 'RearSpinGripShare': 0.95,
        'SurfaceGrip': 1.0,
    },
    'fitted': {
        'MaxSteerAngleDeg': 50.0, 'EnableRearThrustLimit': 1.0, 'MaxSteerRateDegS': 150.0,
        'EnableRearBrakeLimit': 1.0, 'EnableRearFrictionCircle': 1.0, 'EnableTireLoadSensitivity': 1.0,
        'FrontBrakeOnly': 1.0, 'RoundTireCurve': 1.0, 'RearSpinGripShare': 0.95, 'FittedTireCorrection': 1.0,
    },
}


def _inputs(name, tick):
    """(steer, speed, brake) for a scenario at a tick."""
    if name == 'lift':                      # a gentle bend, off the throttle (the brake pedal, where there is one)
        return 0.15, 0.7 if tick == 0 else -1.0, 0.0
    if name == 'full_lock':                 # full lock, a little throttle (72 km/h at the start)
        return 1.0, 0.2, 0.0
    if name == 'flick':                     # feint, turn in, countersteer on the throttle
        if tick < 8:
            return -0.5, 0.6, 0.0
        if tick < 20:
            return 1.0, 1.0, 0.0
        return -0.6, 1.0, 0.0
    if name == 'front_brake':               # turned in on the throttle with the left foot on the brake
        return 0.6, 1.0, 1.0 if 5 <= tick < 45 else 0.0
    raise KeyError(name)


# The plugin's first tick puts a car at the speed its action asks for (a spawn), so each scenario's
# first input sets its starting speed: (speed + 1) / 2 x 120 km/h.
SCENARIOS = {'lift': 80, 'full_lock': 80, 'flick': 80, 'front_brake': 80}     # name: ticks

FIELDS = ('x', 'z', 'heading', 'yaw', 'latv', 'speed')


def run(tracks_dir):
    """{variant: {scenario: {field: [samples]}}}"""
    import fast_sim as fs
    import drift_env as de
    import generate_track as gt
    name = gt.ensure(tracks_dir, *COURSE)
    world = fs.load_world(tracks_dir, track=name)
    arrays = world.arrays()
    start = world.sp_pos[0]
    ahead = world.sp_pos[3] - start
    heading = math.atan2(ahead[2], ahead[0])
    out = {}
    for variant, settings in VARIANTS.items():
        p = de.apply_plugin_settings(fs.make_params(), settings)
        out[variant] = {}
        for scenario, ticks in SCENARIOS.items():
            s = fs.initial_state(tuple(start), heading, MAX_SPEED_KMH / 3.6)
            mid = np.zeros(fs.MID_SIZE)
            diag = np.zeros(fs.DIAG_SIZE)
            trace = {f: [] for f in FIELDS}
            for tick in range(ticks):
                steer, speed, brake = _inputs(scenario, tick)
                fs.step(s, steer, speed, brake, DT_S, p, diag, mid, *arrays)
                if (tick + 1) % EVERY == 0:
                    for f, v in zip(FIELDS, (s[fs.S_X], s[fs.S_Z], s[fs.S_HEADING], s[fs.S_YAW], s[fs.S_LATV],
                                             s[fs.S_OWN_SPEED])):
                        trace[f].append(round(float(v), 5))
            out[variant][scenario] = trace
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='write the traces to tests/golden/sim_traces.json')
    ap.add_argument('--tracks', default=None, help='where the generated course is made (default: a temporary folder)')
    args = ap.parse_args()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        traces = run(args.tracks or os.environ.get('GD_TEST_TRACKS') or tmp)
    if args.write:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, 'w') as f:
            json.dump(traces, f, indent=1)
            f.write('\n')
        print(f'wrote {GOLDEN}')
    else:
        print(json.dumps({v: {s: {k: t[k][-1] for k in t} for s, t in sc.items()} for v, sc in traces.items()}, indent=1))


if __name__ == '__main__':
    main()
