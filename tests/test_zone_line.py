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
    moved = zl.move_line(xz, edges, zones, inside_m=2.0, ramp=4, smooth=2)
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
    moved = zl.move_line(xz, edges, zones, inside_m=3.0, ramp=0, smooth=0)
    assert np.allclose(moved[[55, 59, 0, 5], 1], 3.0) and np.allclose(moved[6:55, 1], 0)


def test_line_file_round_trip(tmp_path):
    n = 5
    raw = bytearray(4 * 4 + 20 * n)
    import struct
    struct.pack_into('<iiii', raw, 0, 7, n, 0, 0)
    for i in range(n):
        struct.pack_into('<fffi', raw, 16 + 20 * i, i, 1.0, -i, i)
    xyz = zl.read_line(bytes(raw))
    assert xyz.shape == (n, 3) and np.allclose(xyz[:, 0], np.arange(n))
    xyz[:, 2] += 2.5
    back = zl.read_line(zl.write_line(bytes(raw), xyz))
    assert np.allclose(back, xyz)


def test_never_writes_through_a_link(tmp_path):
    target = tmp_path / 'shared.txt'
    target.write_bytes(b'original')
    link = tmp_path / 'link.txt'
    link.symlink_to(target)
    zl.write_file(str(link), b'moved')
    assert target.read_bytes() == b'original' and link.read_bytes() == b'moved' and not link.is_symlink()


def _track(tmp_path, n=120):
    import struct
    xz, edges = straight(n)
    folder = tmp_path / 't'
    folder.mkdir()
    raw = bytearray(16 + 20 * n)
    struct.pack_into('<iiii', raw, 0, 7, n, 0, 0)
    for i in range(n):
        struct.pack_into('<fff', raw, 16 + 20 * i, xz[i, 0], 0.0, 0.0)
    (folder / 't_fast_lane.ai').write_bytes(bytes(raw))
    edges.to_csv(folder / 't_course_edges.csv', index=False)
    pd.DataFrame({'Zone': [1], 'Kind': ['outside zone'], 'Side': [1], 'FirstPointId': [30],
                  'LastPointId': [55]}).to_csv(folder / 't_course_zones.csv', index=False)
    return folder


def test_command_line_moves_undoes_and_starts_fresh(tmp_path):
    folder = _track(tmp_path)
    line = folder / 't_fast_lane.ai'
    original = line.read_bytes()
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--inside', '2'])
    assert np.isclose(zl.read_line(line.read_bytes())[42, 2], 4.0)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--inside', '3'])     # again: from the original
    assert np.isclose(zl.read_line(line.read_bytes())[42, 2], 3.0)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--undo'])
    assert line.read_bytes() == original
    # a rebuilt line (setup_track): --fresh keeps what is there now as the original
    zl.main(['--tracks', str(tmp_path), '--track', 't'])
    line.write_bytes(original)
    zl.main(['--tracks', str(tmp_path), '--track', 't', '--fresh'])
    assert (folder / 't_fast_lane.ai.before_zone_line.old').exists()
    assert (folder / 't_fast_lane.ai.before_zone_line').read_bytes() == original
