"""build_course.py's edge tidying (no track mesh needed): the slope limit and the fold limit."""
import math

import numpy as np

import build_course as bc


def circle_line(radius, n=200, y=0.0):
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    # counter-clockwise in X/Z: the positive cross-track side (normal = (-tan_z, tan_x)) is the inside
    return bc.Line(np.c_[radius * np.cos(a), np.full(n, y), radius * np.sin(a)])


def test_slope_limit_flattens_a_one_point_spike():
    line = circle_line(100.0)
    w = np.full(len(line.sp), 6.0)
    w[50] = 30.0
    out = bc.slope_limit(w, line)
    step = line.seg_len[0]
    assert out[50] <= 6.0 + bc.MAX_EDGE_SLOPE * step + 1e-9
    assert np.allclose(np.delete(out, 50), 6.0)


def test_slope_limit_keeps_narrow_places_and_wraps_the_lap():
    line = circle_line(100.0)
    w = np.full(len(line.sp), 20.0)
    w[0] = 3.0
    out = bc.slope_limit(w, line)
    assert out[0] == 3.0
    step = line.seg_len[0]
    # the last point is one segment before point 0 around the lap
    assert abs(out[-1] - (3.0 + bc.MAX_EDGE_SLOPE * step)) < 1e-9
    assert np.all(out <= w)


def test_fold_limit_caps_the_inside_of_a_corner_at_its_radius():
    line = circle_line(12.0)
    inside = bc.fold_limit(line, +1)
    outside = bc.fold_limit(line, -1)
    assert np.all(inside <= 12.0 + 0.5)
    assert np.all(inside >= 11.0)
    assert np.allclose(outside, bc.PAVED_EDGE_SEARCH_M)


def test_fold_limit_stops_between_two_stretches_side_by_side():
    # a long thin loop: two straights 16 m apart, joined by half circles
    pts = []
    for x in np.arange(0.0, 100.0, 1.0):
        pts.append((x, 0.0, 0.0))
    for a in np.linspace(-math.pi / 2, math.pi / 2, 25, endpoint=False):
        pts.append((100.0 + 8.0 * math.cos(a), 0.0, 8.0 + 8.0 * math.sin(a)))
    for x in np.arange(100.0, 0.0, -1.0):
        pts.append((x, 0.0, 16.0))
    for a in np.linspace(math.pi / 2, 3 * math.pi / 2, 25, endpoint=False):
        pts.append((8.0 * math.cos(a), 0.0, 8.0 + 8.0 * math.sin(a)))
    line = bc.Line(np.array(pts))
    lim = bc.fold_limit(line, +1)
    # mid-straight, the other straight is 16 m away: the edge stops near halfway
    assert 7.0 <= lim[50] <= 8.5


def test_fold_limit_ignores_a_stretch_on_a_bridge():
    line = circle_line(12.0)
    sp = line.sp.copy()
    sp[len(sp) // 2:, 1] = 6.0          # half the lap 6 m higher: never nearer in the fold sense
    lim = bc.fold_limit(bc.Line(sp), +1)
    assert lim[len(sp) // 4] > 12.0


def test_off_track_materials():
    mats = {'Tarmac', 'Background', 'Sand02AC', 'curb', 'WB_Gravel_03', 'Cement004'}
    ground = {'Tarmac': 1000.0, 'Background': 900.0, 'Sand02AC': 40.0, 'curb': 50.0, 'WB_Gravel_03': 500.0,
              'Cement004': 30.0}
    grass = {'Background': 850.0, 'curb': 10.0}
    line = {'Tarmac': 90, 'WB_Gravel_03': 10}
    off = bc.off_track_materials(mats, ground, grass, line, 100)
    # drawn like the grass around it, or named as run-off or an apron; the line's own gravel and the kerb are course
    assert off == {'Background', 'Sand02AC', 'Cement004'}
