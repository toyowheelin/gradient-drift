"""
tandem_report.py - how the tandem-train followers in a plugin telemetry file behaved.

    python tandem_report.py <drift_physics_telemetry_*.csv>

Per follower: the share of its ticks it was following (else the server's own AI had it, waiting for
its lead's history), how far behind its lead's replay it ran (delay), how far it was from the car in
front (gap, nose to nose), its sideways offset, and jumps (a move between two ticks more than twice
what its speed allows, plus 1 m).
"""
import sys

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    df = pd.read_csv(sys.argv[1], low_memory=False, on_bad_lines='skip')
    if 'FollowLeadId' not in df.columns:
        sys.exit('no follower columns: a build without tandem-train telemetry wrote this file')
    for c in ('Tick', 'ServerTimeMs', 'SessionId', 'PosX', 'PosZ', 'SpeedKmh', 'FollowLeadId', 'FollowPlace',
              'FollowDelayMs', 'FollowGapM', 'FollowOffsetM'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    f = df[df.FollowLeadId >= 0].sort_values(['SessionId', 'Tick'])
    if f.empty:
        sys.exit('no follower rows (TrainSize was 1, or no train formed)')
    print(f'{len(f)} follower rows, {f.SessionId.nunique()} followers')
    print(f"{'car':>4} {'lead':>4} {'place':>5} {'following':>9}   delay ms p5/50/95      gap m p5/50/95 (min)    "
          f"offset m range   jumps")
    for sid, g in f.groupby('SessionId'):
        on = g[g.FollowDelayMs >= 0]
        dt = np.diff(g.ServerTimeMs.to_numpy()) / 1000.0
        step = np.hypot(np.diff(g.PosX.to_numpy()), np.diff(g.PosZ.to_numpy()))
        allowed = 2.0 * (g.SpeedKmh.to_numpy()[1:] / 3.6) * dt + 1.0
        jumps = int(np.sum((step > allowed) & (dt > 0) & (dt < 1.0)))
        if on.empty:
            print(f'{int(sid):>4} {int(g.FollowLeadId.iloc[0]):>4} {int(g.FollowPlace.iloc[0]):>5}       0%   never followed')
            continue
        d = np.percentile(on.FollowDelayMs, [5, 50, 95])
        gp = np.percentile(on.FollowGapM, [5, 50, 95])
        print(f'{int(sid):>4} {int(g.FollowLeadId.iloc[0]):>4} {int(g.FollowPlace.iloc[0]):>5} {len(on) / len(g):>8.0%}   '
              f'{d[0]:5.0f} {d[1]:5.0f} {d[2]:5.0f}      {gp[0]:5.1f} {gp[1]:5.1f} {gp[2]:5.1f} ({on.FollowGapM.min():.1f})    '
              f'{on.FollowOffsetM.min():+.2f} {on.FollowOffsetM.max():+.2f}    {jumps}')


if __name__ == '__main__':
    main()
