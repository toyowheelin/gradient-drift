"""Shared pieces of the regression tests (pytest, from the repository root: python -m pytest tests).

The tests never use game data: the tracks are the trainer's own generated courses
(drift_rl/generate_track.py), which are the same every time for a seed and course number, and the
car is made-up numbers. GD_TEST_TRACKS=<folder> keeps the generated courses between runs (CI caches
it); without it they go to a temporary folder.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT_RL = os.path.join(ROOT, 'drift_rl')
SETUP = os.path.join(ROOT, 'setup')
for path in (DRIFT_RL, SETUP):
    if path not in sys.path:
        sys.path.insert(0, path)

SHIPPED_POLICY = os.path.join(ROOT, 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json')

# generated courses the tests drive: (seed, course number). Seed 1 is the set the shipped policy
# was trained with (its "generated" block); each course draws its size (technical, tight, fast),
# what bounds it (walls, tire stacks on a pad, grass with walls further out) and its ground.
COURSES = [(1, k) for k in range(5)]


@pytest.fixture(scope='session')
def tracks_dir(tmp_path_factory):
    import generate_track as gt
    folder = os.environ.get('GD_TEST_TRACKS') or str(tmp_path_factory.mktemp('tracks'))
    for seed, k in COURSES:
        gt.ensure(folder, seed, k)
    return folder


@pytest.fixture(scope='session')
def course_names():
    import generate_track as gt
    return [gt.track_name(seed, k) for seed, k in COURSES]
