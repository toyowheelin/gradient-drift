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


def test_smooth_edge_removes_a_notch_and_wraps():
    w = np.full(100, 6.0)
    w[40:42] = 8.0                      # a notch out onto the apron, two points wide
    w[0] = 7.5                          # one at the lap's seam
    out = bc.smooth_edge(w)
    assert np.allclose(out, 6.0)


def test_smooth_edge_follows_a_gradual_change_and_keeps_a_narrowing():
    w = np.linspace(4.0, 8.0, 100)
    w = np.r_[w, w[::-1]]
    out = bc.smooth_edge(w)
    assert np.all(np.abs(out[10:90] - w[10:90]) < 0.2)
    w = np.full(100, 6.0)
    w[50:60] = 3.0                      # a real narrowing (a wall) ten points long
    out = bc.smooth_edge(w)
    assert np.all(out[52:58] <= 3.0 + 1e-9)


def _strip_along(line, offset, s0, s1, sign=1, width=0.15, y=0.01):
    """Triangles of a painted strip `offset` m out on the `sign` side, from point s0 to s1."""
    tris = []
    normal = np.c_[-line.tan[:, 2], line.tan[:, 0]] * sign
    for i in range(s0, s1):
        j = (i + 1) % len(line.sp)
        a_in = line.sp[i, [0, 2]] + normal[i] * offset
        a_out = line.sp[i, [0, 2]] + normal[i] * (offset + width)
        b_in = line.sp[j, [0, 2]] + normal[j] * offset
        b_out = line.sp[j, [0, 2]] + normal[j] * (offset + width)
        for p, q, r in ((a_in, a_out, b_in), (a_out, b_out, b_in)):
            tris.append([p[0], y, p[1], q[0], y, q[1], r[0], y, r[1]])
    return tris


def test_painted_side_line_sets_the_edge_and_chevrons_do_not():
    line = circle_line(100.0, n=400)        # ~1.57 m between points, 628 m lap
    tris = _strip_along(line, 6.0, 0, 300)  # a side line over most of the lap
    names = ['side'] * len(tris)
    chevron = _strip_along(line, 3.0, 50, 55)     # 8 m long: not a side line
    tris += chevron
    names += ['chevron'] * len(chevron)
    high = _strip_along(line, 4.0, 0, 300, y=10.0)    # power lines overhead
    tris += high
    names += ['wire'] * len(high)
    near = _strip_along(line, 2.3, 100, 140, sign=-1)  # a line the cars drive along, 2.3 m off
    tris += near
    names += ['hairpin line'] * len(near)
    strips = bc.paint_strips(line, tris, names, lambda x, y, z: 0.0)
    assert len(strips) == 2 * 300
    d = bc.paint_distance(line, strips, +1)
    assert np.allclose(d[10:290], 6.0, atol=0.02)
    assert np.isinf(d[320:390]).all()
    assert np.isinf(bc.paint_distance(line, strips, -1)).all()


def test_follow_paint_bridges_short_gaps_and_blends_long_ones():
    line = circle_line(100.0, n=400)
    built = np.full(400, 8.0)
    built[100:105] = 5.0                   # a tyre stack inside the line: it stays
    paint = np.full(400, 6.5)
    paint[200:204] = np.inf                # ~6 m without paint: carried across
    paint[300:] = np.inf                   # ~157 m without paint (the straight): built edge
    paint[50] = 11.5                       # a line further out than built + 3 m: not this side's
    out = bc.follow_paint(built, paint, line)
    assert (out <= built + 1e-9).all()
    assert np.allclose(out[100:105], 5.0)
    assert np.allclose(out[200:204], 6.5)
    assert np.allclose(out[320:380], 8.0)
    assert 6.5 < out[302] < 8.0            # blended in from the painted end
    assert np.allclose(out[50], 6.5)       # carried across from its neighbours
    assert np.abs(np.diff(out[290:310])).max() < 0.25


def test_follow_paint_without_paint_keeps_the_edge():
    built = np.linspace(4.0, 9.0, 100)
    assert np.array_equal(bc.follow_paint(built, np.full(100, np.inf), circle_line(50.0, n=100)), built)


def test_track_triangles_picks_thin_painted_meshes():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    import track_triangles as tt
    # a 0.15 m x 40 m strip in two triangles, and a 4 m x 4 m painted box
    strip = [(0, 0, 0, 40, 0, 0, 0, 0, 0.15), (40, 0, 0, 40, 0, 0.15, 0, 0, 0.15)]
    box = [(0, 0, 0, 4, 0, 0, 0, 0, 4), (4, 0, 0, 4, 0, 4, 0, 0, 4)]
    assert abs(tt.strip_width(strip) - 0.15) < 0.01
    assert tt.is_paint('link_whitepaint_1', 'Trackpaint', strip)
    assert tt.is_paint('mesh12', 'Line White', strip)
    assert not tt.is_paint('kerb_01', 'Kerb', strip)
    assert not tt.is_paint('grid_box', 'Line White', box)
