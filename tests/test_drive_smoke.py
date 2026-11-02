"""Short drive checks: the shipped policy drives each generated course for about 30 seconds the way
the server would (nothing resets a car), and has to drive like a drift bot: sliding most of the
time, never spinning out, rarely touching a wall or leaving the course. A broken policy, a broken
track tool or a broken simulator change fails here before it reaches a server.

The floors sit well below what the shipped policy does (the first run's numbers are beside each)
so only real breakage trips them; drive_report.py is the tool for judging a policy's quality."""
import os

import numpy as np
import pytest

from conftest import COURSES, SHIPPED_POLICY

CARS = 8
TICKS = 600             # about 30 s
PACE = 1.0
GRIP = 0.55             # RearSlidingGrip, the plugin's default

FLOORS = {
    'slide': ('at least', 0.40),           # share of moving time sliding 20 - 80 deg (first run 0.63 - 0.69)
    'spin': ('at most', 0.02),             # share of time in spin recovery (0 - 0.005)
    'walls': ('at most', 0.05),            # share of ticks pushed out of a wall (0 - 0.021)
    'off_course': ('at most', 0.05),       # share of time off the course (0 - 0.005)
    'kmh': ('at least', 25.0),             # mean speed while moving (48 - 81)
}


@pytest.fixture(scope='module')
def policy():
    from policy import Policy
    return Policy.load_json(SHIPPED_POLICY)


def drive(tracks_dir, course, policy):
    import drift_env as de
    import drive_report as dr
    import fast_sim as fs
    world = fs.load_world(tracks_dir, track=course)
    world.pace_ref = de.course_pace_reference(tracks_dir, course, world)
    row, st = dr.bot(world, policy, PACE, GRIP, TICKS, CARS, {})
    return {'slide': row['slide'], 'spin': row['spin'], 'kmh': row['kmh'],
            'walls': float(np.mean(st['pushout_tick_fraction'])),
            'off_course': float(np.mean(st['offcourse_fraction']))}


@pytest.mark.parametrize('index', range(len(COURSES)))
def test_shipped_policy_drives(tracks_dir, course_names, policy, index):
    course = course_names[index]
    got = drive(tracks_dir, course, policy)
    print(f"{course}: " + ', '.join(f'{k} {v:.3f}' for k, v in got.items()))
    bad = [f"{k} {got[k]:.3f} (want {how} {limit})" for k, (how, limit) in FLOORS.items()
           if (got[k] < limit if how == 'at least' else got[k] > limit)]
    assert not bad, f"{os.path.basename(SHIPPED_POLICY)} on {course}: " + '; '.join(bad)
