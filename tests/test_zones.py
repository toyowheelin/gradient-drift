"""The drift zones setup's zone editor changes (setup/gradientdrift_setup/zones.py, the `zones`
command, drift_rl/build_zones.py's file): read and written without losing anything the trainer
reads, hand-drawn zones kept (and following the course) when the track's data is made again, and a
recorded run scored the way build_zones.score does it."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from conftest import COURSES


@pytest.fixture
def track(tracks_dir, tmp_path):
    """A copy of the first generated course in a folder of its own (the tests change its zones)."""
    import generate_track as gt
    name = gt.track_name(*COURSES[0])
    shutil.copytree(os.path.join(tracks_dir, name), tmp_path / name)
    return str(tmp_path), name


def fast_sim_columns(tracks, name):
    import track_files as tf
    return pd.read_csv(tf.path(tracks, 'course_zones', name))[['Side', 'FirstPointId', 'LastPointId']].to_numpy()


def test_auto_zones_round_trip(track):
    import build_zones as bz
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, zones, edited = Z.load(tracks, name)
    assert not edited and zones
    placed = bz.place_zones(course.edges)
    assert [(z['kind'], z['side'], z['first'], z['last']) for z in zones] == \
           [(z['kind'], z['side'], z['first'], z['last']) for z in placed]
    assert all(z['length'] > 0 for z in zones)


def test_save_keeps_hand_drawn_zones(track, capsys):
    import build_zones as bz
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, zones, _ = Z.load(tracks, name)
    before = fast_sim_columns(tracks, name)
    drawn = Z.new_zone(course, 1, course.n // 2 + 40, course.n // 2)      # given backwards: the short way round
    assert (drawn['first'], drawn['last']) == (course.n // 2, course.n // 2 + 40)
    zones = [Z.move(course, zones[0], kind=Z.KINDS[0])] + zones[1:] + [drawn]
    path, backup, edited = Z.save(tracks, name, course, zones)
    assert edited and backup and os.path.exists(backup)
    assert np.array_equal(pd.read_csv(backup)[['Side', 'FirstPointId', 'LastPointId']].to_numpy(), before)
    _, again, edited = Z.load(tracks, name)
    assert edited and len(again) == len(zones)
    assert (drawn['side'], drawn['first'], drawn['last']) in [(z['side'], z['first'], z['last']) for z in again]
    saved = fast_sim_columns(tracks, name)
    bz.main(['--tracks', tracks, '--track', name])               # what setup_track runs when it makes the data again
    assert 'kept the hand-drawn zones' in capsys.readouterr().out
    assert np.array_equal(fast_sim_columns(tracks, name), saved)
    bz.main(['--tracks', tracks, '--track', name, '--replace'])
    _, placed, edited = Z.load(tracks, name)
    assert not edited and len(placed) == len(bz.place_zones(course.edges))


def test_zones_follow_a_course_built_again(track):
    """The course made again with its points starting elsewhere: hand-drawn zones keep their place."""
    import build_zones as bz
    import track_files as tf
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, _, _ = Z.load(tracks, name)
    drawn = Z.new_zone(course, -1, 100, 160)
    Z.save(tracks, name, course, [drawn])
    path = tf.path(tracks, 'course_edges', name)
    edges = pd.read_csv(path)
    shifted = pd.concat([edges.iloc[25:], edges.iloc[:25]], ignore_index=True)
    shifted['PointId'] = np.arange(len(shifted))
    shifted.to_csv(path, index=False)
    zones, edited = bz.read_zones(tf.path(tracks, 'course_zones', name), shifted)
    assert edited and (zones[0]['first'], zones[0]['last']) == (75, 135)


def test_problems_and_wrapping(track):
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, _, _ = Z.load(tracks, name)
    over_start = Z.new_zone(course, 1, course.n - 10, 15)
    assert len(course.points(over_start['first'], over_start['last'])) == 26
    assert Z.problems(course, [over_start, Z.new_zone(course, 1, 5, 30)]) == ['zones 1 and 2 overlap']
    assert Z.problems(course, [Z.new_zone(course, 1, 5, 6)]) == ['zone 1 is only 2 point(s) long']


def monitor_recording(course, path, inside_m):
    """An AI Car Monitor recording of one driver going round three times inside_m in from the left
    edge, sliding 30 degrees with the nose to the left."""
    t = np.arange(course.n * 3) % course.n
    inward = (course.neg - course.pos)[t]
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    car = course.pos[t] + inward * inside_m
    d = np.roll(course.mid, -1, 0)[t] - course.mid[t]
    d /= np.linalg.norm(d, axis=1)[:, None]
    a = np.radians(30)
    body = np.c_[d[:, 0] * np.cos(a) - d[:, 1] * np.sin(a), d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)]
    side = np.c_[-body[:, 1], body[:, 0]] * 0.75
    rear, front = car - 1.3 * body, car + 1.3 * body
    cols = dict(ClientTimeMs=np.arange(len(t)) * 50, CarId=3, DriverName='Tester', CarModel='car',
                X=car[:, 0], Y=0.0, Z=car[:, 1], SpeedKMH=60.0, Velocity_X=d[:, 0] * 16, Velocity_Y=0.0,
                Velocity_Z=d[:, 1] * 16)
    for wheel, base, sign in (('FL', front, 1), ('FR', front, -1), ('RL', rear, 1), ('RR', rear, -1)):
        cols[f'Wheel{wheel}_X'] = base[:, 0] + sign * side[:, 0]
        cols[f'Wheel{wheel}_Y'] = 0.0
        cols[f'Wheel{wheel}_Z'] = base[:, 1] + sign * side[:, 1]
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def test_score_a_recording(track, tmp_path):
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, _, _ = Z.load(tracks, name)
    left, right = Z.new_zone(course, 1, 100, 160), Z.new_zone(course, -1, 300, 360)
    runs = Z.load_runs(monitor_recording(course, str(tmp_path / 'rec.csv.gz'), 1.5))
    assert [r.name for r in runs] == ['Tester']
    assert abs(np.median(runs[0].slip_deg) - 30) < 1
    rows = Z.score(course, [left, right], runs[0])
    assert rows[0][0] > 0.95 and 1.5 < rows[0][1] < 2.5         # rear ~2.15 m in from the left edge
    assert rows[1][0] == 0.0 and rows[1][1] > 5                 # nowhere near the right edge
    assert Z.line_score(rows) == pytest.approx(40 * np.mean([rows[0][0], 0.0]))
    out = Z.picture(course, [left, right], str(tmp_path / 'zones.svg'), run=runs[0])
    text = open(out, encoding='utf-8').read()
    assert text.startswith('<svg') and text.count('<polygon') >= 3 and 'stroke="#f5c400"' in text


def test_zones_command(track, tmp_path, capsys):
    from gradientdrift_setup.cli import main
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, zones, _ = Z.load(tracks, name)
    words = ['zones', name, '--track-data', tracks]
    assert main(words + ['list']) == 0
    assert main(words + ['add', 'left', '100', '160', 'touch']) == 0
    _, after, edited = Z.load(tracks, name)
    assert edited and len(after) == len(zones) + 1
    k = 1 + next(i for i, z in enumerate(Z.ordered(course, after)) if z['first'] == 100 and z['side'] == 1)
    assert main(words + ['move', str(k), '90', f'{course.along[170]:.1f}m']) == 0
    _, after, _ = Z.load(tracks, name)
    moved = next(z for z in after if z['side'] == 1 and z['first'] == 90)
    assert moved['kind'] == Z.KINDS[1] and abs(moved['last'] - 170) <= 1
    assert main(words + ['delete', '99']) == 1
    assert main(words + ['delete', str(k)]) == 0
    assert len(Z.load(tracks, name)[1]) == len(zones)
    assert main(words + ['auto']) == 0
    assert not Z.load(tracks, name)[2]           # the automatic zones again: no longer marked as edited
    rec = monitor_recording(course, str(tmp_path / 'rec.csv'), 1.5)
    assert main(words + ['score', rec]) == 0
    assert main(words + ['picture', str(tmp_path / 'map.png'), rec, 'Tester']) == 0
    assert os.path.exists(tmp_path / 'map.svg')
    assert 'line score' in capsys.readouterr().out


def test_paint_shapes(track):
    """What the plugin paints (DriftZones.cs mirrors it; tests/plugin checks the two agree): a closed
    frame round each zone and stripes inside it, sparser in a touch-and-go area."""
    from gradientdrift_setup import zones as Z
    tracks, name = track
    course, _, _ = Z.load(tracks, name)
    long_oz = Z.new_zone(course, 1, 100, 160)
    long_tg = Z.new_zone(course, 1, 100, 160, kind=Z.KINDS[1])
    tiny = Z.new_zone(course, -1, 300, 301)
    (oz, f1, s1), (tg, f2, s2), (_, f3, s3) = Z.paint_shapes(course, [long_oz, long_tg, tiny], 1.6)
    assert oz and not tg and len(f1) == 2 * 61 and len(s3) == 0
    assert len(s1) > 10 and abs(len(s1) / max(len(s2), 1) - 2) < 0.3
    outer, inner = course.zone_outline(long_oz, inset=0.15)      # the outer line half its 0.3 m width in

    def off(line, p):                                 # distance from p to a polyline
        a, b = line[:-1], line[1:]
        t = np.clip(np.sum((p - a) * (b - a), 1) / np.maximum(np.sum((b - a) ** 2, 1), 1e-9), 0, 1)
        return np.min(np.hypot(*(a + t[:, None] * (b - a) - p).T))
    for a, b in s1:                                   # each stripe runs from the outer edge to 3 m in
        assert off(outer, a) < 0.01 and off(inner, b) < 0.01
        assert 2.0 < np.hypot(*(a - b)) < 4.5


def test_spikes_taken_out(tracks_dir):
    """An edge that jumps out across open run-off for a few points (or dips in at one) is held near
    its neighbours before zones are placed, drawn, painted and scored; a clean course is untouched."""
    import build_zones as bz
    import generate_track as gt
    import track_files as tf
    for seed, k in COURSES:                       # the generated courses: nothing to take out
        edges = pd.read_csv(tf.path(tracks_dir, 'course_edges', gt.track_name(seed, k)))
        assert bz.smooth_edges(edges).equals(edges)
    edges = pd.read_csv(tf.path(tracks_dir, 'course_edges', gt.track_name(*COURSES[0])))
    spiked = edges.copy()
    pos, neg = bz.edge_arrays(edges)
    out = (pos - neg) / np.linalg.norm(pos - neg, axis=1)[:, None]
    for i, change in ((50, 15.0), (51, 20.0), (52, 15.0), (200, -4.0)):
        spiked.loc[i, ['PositiveEdgeX', 'PositiveEdgeZ']] = pos[i] + out[i] * change
        spiked.loc[i, 'PositiveSideM'] += change
    smooth = bz.smooth_edges(spiked)
    assert smooth.attrs['smoothed'] and bz.smooth_edges(smooth) is smooth
    p_new, _ = bz.edge_arrays(spiked)
    assert np.abs(p_new[[50, 51, 52]] - pos[[50, 51, 52]]).max() < bz.SPIKE_M + 0.01   # back within a metre
    assert np.allclose(p_new[200], spiked.loc[200, ['PositiveEdgeX', 'PositiveEdgeZ']].to_numpy(float))  # a dip stays
    widths = bz.smooth_edges(spiked).PositiveSideM.to_numpy()
    assert (widths <= spiked.PositiveSideM.to_numpy() + 1e-9).all()              # the edge only ever comes in
    near = [i for c in (50, 51, 52, 200) for i in range(c - bz.SMOOTH_POINTS // 2, c + bz.SMOOTH_POINTS // 2 + 1)]
    assert np.allclose(np.delete(p_new, near, 0), np.delete(pos, near, 0))
    assert [(z['side'], z['first'], z['last']) for z in bz.place_zones(spiked)] == \
           [(z['side'], z['first'], z['last']) for z in bz.place_zones(edges)]
