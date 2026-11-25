"""Car avoidance on pace (drift_rl/car_avoid.py, the plugin's AvoidCars): who slows, and how much."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drift_rl'))
import car_avoid as ca  # noqa: E402

H, M, F = 1.2, 0.3, 0.5


def share(own, other):
    return ca.speed_share(own, other, H, M, F)


def test_clear_road_and_cars_behind_change_nothing():
    assert share((0, 0, 15, 0, 0), (25, 0, 0, 0, 0)) == 1.0          # stopped car out of reach
    assert share((0, 0, 15, 0, 0), (8, 0, 15, 0, 0)) == 1.0          # same speed ahead
    assert share((0, 0, 15, 0, 0), (-5, 0, 25, 0, 0)) == 1.0         # faster car behind: it yields
    assert share((0, 0, 15, 0, 0), (5, 4, 15, 0, 0.5)) == 1.0        # alongside on another line


def test_slows_for_a_stopped_car_ahead_and_harder_when_closer():
    far = share((0, 0, 15, 0, 0), (17, 0, 0, 0, 1.5))
    near = share((0, 0, 15, 0, 0), (12, 0, 0, 0, 1.5))
    assert F <= near < far < 1.0


def test_slows_for_a_car_cutting_in_just_ahead():
    assert share((0, 0, 15, 0, 0), (5, 3, 12, -3, 0.3)) < 1.0


def test_touching_a_car_ahead_backs_off_to_the_floor():
    assert share((0, 0, 15, 0, 0.6), (1, 2.8, 15, 0, 0.6)) == F


def test_a_car_pulling_away_never_slows_the_bot():
    assert share((0, 0, 15, 0, 0), (4.5, 0.5, 20, 0, 0.4)) == 1.0


def test_one_of_two_bots_yields():
    a = (0, 0, 15, 0, 0)
    b = (6, 0, 8, 0, 0)
    assert share(a, b) < 1.0 and share(b, a) == 1.0
