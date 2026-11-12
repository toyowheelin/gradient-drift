"""tools/players_to_monitor.py: a RecordPlayers file (the plugin) reads as an AI Car Monitor
recording, with the car's heading, slip and steering coming back through crowd_paths."""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'drift_rl'))
import players_to_monitor as ptm  # noqa: E402
import telemetry_ghosts as tg  # noqa: E402


def test_circle_in_a_drift(tmp_path):
    # one player on a 30 m circle at 15 m/s, the body turned 30 deg into the turn, road wheels at 10 deg
    t = np.arange(0, 20, 0.05)
    w = 15.0 / 30.0
    ang = w * t
    course = ang + math.pi / 2
    slip = math.radians(30.0)
    d = pd.DataFrame({'ServerTimeMs': (t * 1000).astype(int), 'SessionId': 3, 'DriverId': 'pabc', 'CarModel': 'miata',
                      'PosX': 30 * np.cos(ang), 'PosY': 1.0, 'PosZ': 30 * np.sin(ang), 'HeadingRad': course + slip,
                      'VelX': 15 * np.cos(course), 'VelY': 0.0, 'VelZ': 15 * np.sin(course), 'SpeedKmh': 54.0,
                      'WheelAngleByte': 127 + 20, 'SteerAngleByte': 127, 'GasByte': 255, 'Gear': 3,
                      'EngineRpm': 6000, 'StatusFlag': 0})
    f = tmp_path / 'p.csv'
    d.to_csv(f, index=False)
    out = tmp_path / 'm.csv.gz'
    ptm.convert(pd.read_csv(f)).to_csv(out, index=False, compression='gzip')
    paths = tg.crowd_paths(str(out), 0.13, 250.0)
    assert len(paths) == 1 and paths[0][0] == 3 and paths[0][1] == 'miata'
    p = paths[0][2]
    mid = slice(20, -20)
    assert abs(np.median(np.abs(np.degrees(p['slip'][mid]))) - 30.0) < 1.0
    assert abs(np.median(p['speed'][mid]) - 15.0) < 0.3
    assert abs(np.median(np.degrees(p['steer'][mid])) - 10.0) < 0.2
    assert abs(np.median(p['pedal'][mid]) - 1.0) < 1e-6
