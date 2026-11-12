"""
players_to_monitor.py - a RecordPlayers file from the plugin in the AI Car Monitor's format.

    python tools/players_to_monitor.py gradientdrift_players_20261112_201500.csv.gz [out.csv.gz]

The plugin's RecordPlayers setting logs every connected player as their own client reports it
(gradientdrift_players_<date>_<time>.csv.gz). drift_rl's readers of crowd recordings
(telemetry_ghosts.read_monitor / crowd_paths, crowd_fit.py, amp_reference.py and the style
catalogue) read the AI Car Monitor's columns, so this writes those:
  ClientTimeMs     the server's clock, ms
  CarId            the player's session id; DriverName their salted id (no names in the file)
  Wheel*_X/Y/Z     made from the position and heading (front axle WHEEL_FRONT_M ahead, rear
                   WHEEL_REAR_M behind): the readers take only the heading and a plausible
                   wheelbase from them
  Gas              the client's throttle byte / 255; Brake: not sent to the server, left empty
  Steer            the road wheels' angle from the client's wheel byte (127 = straight,
                   --units-per-degree, the plugin's SteeringVisualWheelByteUnitsPerDegree) over
                   STEER_RATIO, so crowd_paths' steering ratio gives the road wheels back
  SlipRatio_RL/RR  not sent: empty
Out: beside the input as monitor_<same name>, unless given.
"""
import argparse
import os

import numpy as np
import pandas as pd

WHEEL_FRONT_M = 1.3
WHEEL_REAR_M = 1.2
STEER_RATIO = 0.13          # telemetry_ghosts.crowd_paths' steer_ratio as drift_rl calls it

COLUMNS = ['ClientTimeMs', 'CarId', 'DriverName', 'CarModel', 'X', 'Y', 'Z',
           'WheelFL_X', 'WheelFL_Y', 'WheelFL_Z', 'WheelFR_X', 'WheelFR_Y', 'WheelFR_Z',
           'WheelRL_X', 'WheelRL_Y', 'WheelRL_Z', 'WheelRR_X', 'WheelRR_Y', 'WheelRR_Z',
           'SpeedKMH', 'Gas', 'Brake', 'Steer', 'RPM', 'Gear', 'Velocity_X', 'Velocity_Y', 'Velocity_Z',
           'SlipRatio_RL', 'SlipRatio_RR']


def convert(d, units_per_degree=2.0, sign=1.0):
    """A RecordPlayers table -> the AI Car Monitor's columns."""
    h = d.HeadingRad.to_numpy(float)
    fx, fz = np.cos(h), np.sin(h)
    x, y, z = (d[c].to_numpy(float) for c in ('PosX', 'PosY', 'PosZ'))
    out = pd.DataFrame({'ClientTimeMs': d.ServerTimeMs.astype('int64'), 'CarId': d.SessionId.astype(int),
                        'DriverName': d.DriverId.astype(str), 'CarModel': d.CarModel.astype(str),
                        'X': x, 'Y': y, 'Z': z})
    for name, k in (('FL', WHEEL_FRONT_M), ('FR', WHEEL_FRONT_M), ('RL', -WHEEL_REAR_M), ('RR', -WHEEL_REAR_M)):
        out[f'Wheel{name}_X'] = x + k * fx
        out[f'Wheel{name}_Y'] = y
        out[f'Wheel{name}_Z'] = z + k * fz
    road_deg = sign * (d.WheelAngleByte.to_numpy(float) - 127.0) / units_per_degree
    out['SpeedKMH'] = d.SpeedKmh.to_numpy(float)
    out['Gas'] = d.GasByte.to_numpy(float) / 255.0
    out['Brake'] = np.nan
    out['Steer'] = road_deg / STEER_RATIO
    out['RPM'] = d.EngineRpm.to_numpy(float)
    out['Gear'] = d.Gear.to_numpy(float)
    out['Velocity_X'], out['Velocity_Y'], out['Velocity_Z'] = (d[c].to_numpy(float) for c in ('VelX', 'VelY', 'VelZ'))
    out['SlipRatio_RL'] = np.nan
    out['SlipRatio_RR'] = np.nan
    return out[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('players')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--units-per-degree', type=float, default=2.0,
                    help="the plugin's SteeringVisualWheelByteUnitsPerDegree (default 2)")
    ap.add_argument('--sign', type=float, default=1.0, help="the plugin's SteeringVisualSign (default 1)")
    args = ap.parse_args()
    d = pd.read_csv(args.players)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.players)),
                                   'monitor_' + os.path.basename(args.players))
    m = convert(d, args.units_per_degree, args.sign)
    m.to_csv(out, index=False, compression='gzip' if out.endswith('.gz') else None)
    minutes = (d.ServerTimeMs.max() - d.ServerTimeMs.min()) / 60000 if len(d) else 0.0
    print(f'{len(d):,} rows, {d.DriverId.nunique()} players, {minutes:.1f} min -> {out}')


if __name__ == '__main__':
    main()
