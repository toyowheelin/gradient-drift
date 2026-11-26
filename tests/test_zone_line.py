"""zone_line.py: the AI line moved into the drift zones, eased in and out, and put back."""
import numpy as np
import pandas as pd

import zone_line as zl


def straight(n=120, half=6.0):
    """A straight road along x, 1 m a point, the line down the middle, edges `half` m either side."""
    x = np.arange(n, dtype=float)
    edges = pd.DataFrame({'PointId': np.arange(n), 'PositiveSideM': half, 'NegativeSideM': half,
                          'PositiveEdgeX': x, 'PositiveEdgeZ': half, 'NegativeEdgeX': x, 'NegativeEdgeZ': -half})
    return np.c_[x, np.zeros(n)], edges


def test_line_moves_into_the_zone_and_back():
    xz, edges = straight()
    zones = pd.DataFrame({'Side': [1, -1], 'FirstPointId': [30, 80], 'LastPointId': [55, 100]})
    moved = zl.move_line(xz, edges, zones, inside_m=2.0, ramp=4, smooth=2, max_slope=0, new_bend_m=0, min_side_m=0)
    assert np.allclose(moved[:, 0], xz[:, 0])                   # only sideways
    assert np.allclose(moved[38:48, 1], 6.0 - 2.0)               # 2 m inside the + edge mid-zone
    assert np.allclose(moved[86:95, 1], -6.0 + 2.0)              # and the - edge
    assert np.allclose(moved[:20, 1], 0) and np.allclose(moved[64:72, 1], 0)   # racing line between
    assert np.all(np.abs(np.diff(moved[:, 1])) < 1.0)            # eased, no jumps
    w = zl.side_widths(moved, edges)
    assert np.isclose(w.PositiveSideM[42], 2.0) and np.isclose(w.NegativeSideM[42], 10.0)


def test_zone_wrapping_past_the_start():
    xz, edges = straight(60)
    zones = pd.DataFrame({'Side': [1], 'FirstPointId': [55], 'LastPointId': [5]})
    moved = zl.move_line(xz, edges, zones, inside_m=3.0, ramp=0, smooth=0, max_slope=0, new_bend_m=0, min_side_m=0)
    assert np.allclose(moved[[55, 59, 0, 5], 1], 3.0) and np.allclose(moved[6:55, 1], 0)


def _line_file(path, xz):
    from line_tools import write_line
    n = len(xz)
    write_line(str(path), np.c_[xz[:, 0], np.zeros(n), xz[:, 1]], None, np.full(n, 5.0), np.full(n, 5.0))


def test_line_file_round_trip(tmp_path):
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    xz = np.c_[30 * np.cos(t), 30 * np.sin(t)]
    _line_file(tmp_path / 'l.ai', xz)
    raw = (tmp_path / 'l.ai').read_bytes()
    xyz = zl.read_line(raw)
    assert xyz.shape == (80, 3) and np.allclose(xyz[:, [0, 2]], xz, atol=1e-4)
    xyz[:, [0, 2]] *= 2.0 / 3.0                                   # a 20 m circle
    (tmp_path / 'm.ai').write_bytes(zl.write_line(raw, xyz))
    from ai_spline import load_fast_lane
    moved = load_fast_lane(str(tmp_path / 'm.ai'))
    assert np.allclose([p.position for p in moved], xyz, atol=1e-4)
    assert abs(np.median([p.radius for p in moved]) - 20.0) < 0.5    # radius worked out again


def test_never_writes_through_a_link(tmp_path):
    target = tmp_path / 'shared.txt'
    target.write_bytes(b'original')
    link = tmp_path / 'link.txt'
    link.symlink_to(target)
    zl.write_file(str(link), b'moved')
    assert target.read_bytes() == b'original' and link.read_bytes() == b'moved' and not link.is_symlink()


def _track(tmp_path, n=160):
    xz, edges = straight(n)
    folder = tmp_path / 't'
    folder.mkdir()
    _line_file(folder / 't_fast_lane.ai', xz)
    edges.to_csv(folder / 't_course_edges.csv', index=False)
    pd.DataFrame({'Zone': [1], 'Kind': ['outside zone'], 'Side': [1], 'FirstPointId': [30],
                  'LastPointId': [100]}).to_csv(folder / 't_course_zones.csv', index=False)
    return folder


def test_command_line_moves_undoes_and_starts_fresh(tmp_path):
    folder = _track(tmp_path)
    line = folder / 't_fast_lane.ai'
    original = line.read_bytes()
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--inside', '2'])
    assert np.isclose(zl.read_line(line.read_bytes())[65, 2], 4.0)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--inside', '3'])     # again: from the original
    assert np.isclose(zl.read_line(line.read_bytes())[65, 2], 3.0)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--undo'])
    assert line.read_bytes() == original
    # a rebuilt line (setup_track): --fresh keeps what is there now as the original
    zl.main(['--tracks', str(tmp_path), '--track', 't'])
    line.write_bytes(original)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--fresh'])
    assert (folder / 't_fast_lane.ai.before_zone_line.old').exists()
    assert (folder / 't_fast_lane.ai.before_zone_line').read_bytes() == original


def test_short_zone_far_out_gets_a_gentle_line():
    """A short zone far from the line: the move is capped so the line bends no faster than MAX_SLOPE."""
    xz, edges = straight(200, half=12.0)
    zones = pd.DataFrame({'Side': [1], 'FirstPointId': [100], 'LastPointId': [105]})
    moved = zl.move_line(xz, edges, zones, inside_m=2.0)
    lateral = np.abs(np.diff(moved[:, 1]))
    assert lateral.max() <= zl.MAX_SLOPE + 1e-6          # 1 m a point here
    assert 2.0 < moved[100:106, 1].max() < 10.0           # moved out, but not all the way to 10 m


def test_no_new_bends_where_the_racing_line_runs_straight():
    # zones on alternate sides close together: moving straight across between them would put sharp
    # bends on a straight (Spirit Peaks, where the zones are far from the racing line); they are eased
    xz, edges = straight(200, half=10.0)
    zones = pd.DataFrame({'Side': [1, -1, 1], 'FirstPointId': [40, 80, 125], 'LastPointId': [65, 110, 150]})
    raw = zl.move_line(xz, edges, zones, new_bend_m=0, min_side_m=0)
    assert zl.new_bends(xz, raw).any()
    moved = zl.move_line(xz, edges, zones)
    assert not zl.new_bends(xz, moved).any()
    assert moved[52, 1] > 2.0 and moved[95, 1] < -2.0 and moved[137, 1] > 2.0     # still out toward each zone


def test_keeps_off_an_edge_that_bulges_in():
    xz, edges = straight(160, half=8.0)
    bump = 2.0 * np.exp(-((np.arange(160) - 80) / 3.0) ** 2)      # the + edge comes in 2 m over a few metres
    edges['PositiveEdgeZ'] = 8.0 - bump
    zones = pd.DataFrame({'Side': [1], 'FirstPointId': [50], 'LastPointId': [110]})
    raw = zl.move_line(xz, edges, zones, new_bend_m=0, min_side_m=0)
    assert (edges.PositiveEdgeZ - raw[:, 1]).min() < 2.5             # the smoothing cuts the corner of the bump
    moved = zl.move_line(xz, edges, zones, new_bend_m=0)
    assert (edges.PositiveEdgeZ - moved[:, 1]).min() > 2.95


def circle(radius, n=300):
    """A closed circle, anticlockwise, n points."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a)]


def test_a_line_parallel_to_a_corner_bends_as_it_should():
    # 5 m further out on a 20 m corner bends 20/25 as tightly: inside both bounds
    assert not zl.off_bends(circle(20.0), circle(25.0)).any()


def test_a_bump_on_a_corner_is_flagged():
    xz = circle(20.0)
    a = np.linspace(0, 2 * np.pi, len(xz), endpoint=False)
    r = 20.0 + 3.0 * np.exp(-((np.arange(len(xz)) - 150) / 2.0) ** 2)     # out 3 m and back over a few points
    bumped = np.c_[r * np.cos(a), r * np.sin(a)]
    bad = zl.off_bends(xz, bumped)
    assert bad[145:156].any() and not bad[:120].any() and not bad[180:].any()


def stadium(straight_m=100, radius=15.0, half=8.0):
    """A closed stadium-shaped course, anticlockwise, about 1 m a point, `half` m of road either side
    of the line: (line (n, 2), edges table, left normals (n, 2))."""
    pts = []
    for x in np.arange(0, straight_m):
        pts.append((x, -radius))
    for a in np.arange(-np.pi / 2, np.pi / 2, 1 / radius):
        pts.append((straight_m + radius * np.cos(a), radius * np.sin(a)))
    for x in np.arange(straight_m, 0, -1.0):
        pts.append((x, radius))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, 1 / radius):
        pts.append((radius * np.cos(a), radius * np.sin(a)))
    xz = np.array(pts)
    t = np.roll(xz, -1, 0) - np.roll(xz, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    left = np.c_[-t[:, 1], t[:, 0]]
    n = len(xz)
    p, m = np.full(n, half), np.full(n, half)
    edges = pd.DataFrame({'PointId': np.arange(n), 'PositiveSideM': p, 'NegativeSideM': m})
    edges[['PositiveEdgeX', 'PositiveEdgeZ']] = xz + left * half
    edges[['NegativeEdgeX', 'NegativeEdgeZ']] = xz - left * half
    return xz, edges, left


def test_an_edge_bulge_in_a_hairpin_does_not_carry_the_line_out_and_back():
    # the outside (negative, right-hand) edge of the first hairpin bulges out 4 m for 12 points:
    # without the bend check the moved line follows it out and back and straightens at the zone's
    # end; with it the line keeps within the bounds of the racing line's own bend
    xz, edges, left = stadium()
    corner = np.arange(100, 147)                       # the first hairpin's points
    bulge = np.zeros(len(xz))
    bulge[corner[20:32]] = 4.0
    edges['NegativeSideM'] += bulge
    edges[['NegativeEdgeX', 'NegativeEdgeZ']] = xz - left * edges.NegativeSideM.to_numpy()[:, None]
    zones = pd.DataFrame({'Side': [-1], 'FirstPointId': [corner[5]], 'LastPointId': [corner[-5]]})
    raw = zl.move_line(xz, edges, zones, bend_tight=0)
    assert zl.off_bends(xz, raw).any()
    moved = zl.move_line(xz, edges, zones)
    assert not zl.off_bends(xz, moved).any()
    k0, k1 = zl.curvature(xz), zl.curvature(moved)
    d = np.linalg.norm(moved - xz, axis=1)
    turning = np.abs(k0) > 1 / zl.TURN_RADIUS_M
    ratio = k1[turning] / k0[turning]
    assert (ratio >= zl.BEND_LOOSE - 1e-9).all()
    assert (ratio <= zl.BEND_TIGHT * (1 + d[turning] * np.abs(k0[turning])) + 1e-9).all()
    assert np.median(d[corner[8:-8]]) > 1.0            # still moved out toward the zone
