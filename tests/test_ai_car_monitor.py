"""The AI Car Monitor in-game app (tools/ai_car_monitor.py), run against a stand-in for the game's
Python API: it samples every car 20 times a second whatever the frame rate, writes a .csv.gz the
crowd readers (telemetry_ghosts.crowd_paths, crowd_fit, amp_reference) take, and a recording the
game never closed (a crash) still reads up to its last flush."""
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

CARS = 5            # car 3 is a disconnected slot


def fake_game(clock):
    ac = types.ModuleType('ac')
    acsys = types.ModuleType('acsys')

    class CS:
        pass
    for name in ('WorldPosition', 'TyreContactPoint', 'SpeedKMH', 'Gas', 'Brake', 'Steer', 'RPM',
                 'Gear', 'Velocity', 'SlipRatio'):
        setattr(CS, name, name)
    acsys.CS = CS
    acsys.WHEELS = types.SimpleNamespace(FL=0, FR=1, RL=2, RR=3)

    def state(car, key, wheel=None):
        a = clock[0] / 5.0 + car            # each car on a 50 m circle, 10 m/s
        x, z = 50 * math.cos(a), 50 * math.sin(a)
        if key == 'WorldPosition':
            return (x, 1.0, z)
        if key == 'TyreContactPoint':
            h = a + math.pi / 2
            ahead = 1.3 if wheel in (0, 1) else -1.3
            side = 0.7 if wheel in (0, 2) else -0.7
            return (x + ahead * math.cos(h) - side * math.sin(h), 0.9,
                    z + ahead * math.sin(h) + side * math.cos(h))
        if key == 'Velocity':
            return (-10 * math.sin(a), 0.0, 10 * math.cos(a))
        if key == 'SlipRatio':
            return [0.0, 0.0, 1.0, 1.0] if wheel is None else 1.0
        if key == 'SpeedKMH':
            return 36.0
        return {'RPM': 5000.4, 'Gear': 3}.get(key, 0.5)

    ac.getCarState = state
    ac.getCarsCount = lambda: CARS
    ac.isConnected = lambda car: car != 3
    ac.getDriverName = lambda car: 'driver{}'.format(car)
    ac.getCarName = lambda car: 'test_car'
    ac.log = lambda message: None
    for name in ('newApp', 'setSize', 'addLabel', 'addButton', 'setPosition',
                 'addOnClickedListener', 'setText'):
        setattr(ac, name, lambda *args: 0)
    return ac, acsys


def record(tmp_path, monkeypatch, fps, seconds, close=True):
    clock = [0.0]
    ac, acsys = fake_game(clock)
    monkeypatch.setitem(sys.modules, 'ac', ac)
    monkeypatch.setitem(sys.modules, 'acsys', acsys)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    spec = importlib.util.spec_from_file_location('ai_car_monitor', os.path.join(ROOT, 'tools', 'ai_car_monitor.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    monkeypatch.setattr(app.time, 'time', lambda: 1.0e9 + clock[0])
    app.acMain(1)
    app.on_toggle()
    for _ in range(int(fps * seconds)):
        clock[0] += 1.0 / fps
        app.acUpdate(1.0 / fps)
    path = app.recorder.path
    if close:
        app.on_toggle()
    else:
        app.recorder.file.flush()
    return path


@pytest.mark.parametrize('fps', [27, 60, 144])
def test_twenty_a_second_at_any_frame_rate(tmp_path, monkeypatch, fps):
    path = record(tmp_path, monkeypatch, fps, 30)
    assert path.endswith('.csv.gz')
    d = pd.read_csv(path)
    assert sorted(d.CarId.unique()) == [0, 1, 2, 4]
    for _, car in d.groupby('CarId'):
        rate = (len(car) - 1) / ((car.ClientTimeMs.max() - car.ClientTimeMs.min()) / 1000.0)
        assert rate == pytest.approx(20.0, abs=0.3)
    # every car sampled at the same instant, so cars can be compared row by row
    assert (d.groupby('ClientTimeMs').CarId.count() == 4).all()
    assert d.RPM.dtype.kind == 'i' and d.Gear.dtype.kind == 'i'


def test_crowd_readers_take_it(tmp_path, monkeypatch):
    import telemetry_ghosts as tg
    path = record(tmp_path, monkeypatch, 60, 20)
    assert tg.is_crowd(path)
    parts = tg.crowd_paths(path, 0.13, 250.0)
    assert sorted(car for car, _, _ in parts) == [0, 1, 2, 4]
    for _, model, p in parts:
        assert model == 'test_car'
        assert np.median(p['speed']) == pytest.approx(10.0, rel=0.02)
        assert np.median(np.abs(p['slip'])) < 0.02          # gripping round the circle


def test_unclosed_recording_reads_to_its_last_flush(tmp_path, monkeypatch):
    import telemetry_ghosts as tg
    path = record(tmp_path, monkeypatch, 60, 20, close=False)
    with open(path, 'rb') as f:
        data = f.read()
    cut = str(tmp_path / 'cut.csv.gz')
    with open(cut, 'wb') as f:
        f.write(data[:int(len(data) * 0.7)])
    with pytest.raises(EOFError):
        pd.read_csv(cut)
    d = tg.read_monitor(cut)
    assert len(d) > 100 and not d.isna().any().any()
