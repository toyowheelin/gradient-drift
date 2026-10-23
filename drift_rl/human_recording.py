"""
human_recording.py - what a human driver does on this course, from a client-side recording.

    python human_recording.py <recording>.zip

The recording is the in-game telemetry app's CSV (167 samples a second; columns X, Y, Z,
SpeedKMH, RPM, Throttle, Brake, Gear, G_Lat, G_Long, Steer, SlipAngle_RL/RR, SlipRatio_..,
DriveTrainSpeed, Load..), zipped or not. Prints the numbers the plugin's vehicle model and its
drivetrain visual are compared against: speed, sideways (centripetal) acceleration, how far the
rear slides, engine RPM, gear, throttle and driven-wheel speed.

"Rear slip" is the mean of the two rear tires' slip angles: the same quantity as the plugin
telemetry's RearSlipAngleDeg.
"""
import sys

import numpy as np
import pandas as pd

SAMPLE_S = 0.006
MIN_SPEED_KMH = 25.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    d = pd.read_csv(sys.argv[1])
    rear = ((d.SlipAngle_RL + d.SlipAngle_RR) / 2).abs()
    moving = d.SpeedKMH > MIN_SPEED_KMH

    def smooth(a, n=41):
        return np.convolve(a, np.ones(n) / n, mode='same')
    vx = smooth(np.gradient(d.X.values, SAMPLE_S))
    vz = smooth(np.gradient(d.Z.values, SAMPLE_S))
    turn_rate = smooth(np.gradient(np.unwrap(np.arctan2(vz, vx)), SAMPLE_S))
    centripetal = pd.Series(np.abs(np.hypot(vx, vz) * turn_rate))
    wheel_ratio = d.DriveTrainSpeed / d.SpeedKMH.clip(lower=1.0)

    total = d.LoadFL + d.LoadFR + d.LoadRL + d.LoadRR
    print(f"{len(d):,} samples, {len(d) * SAMPLE_S / 60:.1f} min, {moving.sum() * SAMPLE_S:.0f} s above {MIN_SPEED_KMH:g} km/h")
    print(f"car: about {total[moving].median() / 9.80665:.0f} kg, {((d.LoadFL + d.LoadFR) / total)[moving].median():.0%} of it on the front axle while driving")
    print(f"gear changes: {(np.diff(d.Gear.values) != 0).sum() / (len(d) * SAMPLE_S / 60):.1f} a minute; "
          f"highest RPM {d.RPM.max():.0f}; largest steering angle {d.Steer.abs().max():.0f} deg at the steering wheel\n")

    rows = [('all', rear >= 0), ('rear slip under 8 deg', rear < 8), ('rear slip 8-20 deg', (rear >= 8) & (rear < 20)),
            ('rear slip 20-50 deg', (rear >= 20) & (rear < 50)), ('rear slip over 50 deg', rear >= 50)]
    print(f"{'':24s}{'time':>7s}{'speed':>8s}{'sideways':>10s}{'RPM':>7s}{'gear':>6s}{'throttle':>10s}{'wheel /':>9s}{'steering':>10s}")
    print(f"{'':24s}{'':>7s}{'km/h':>8s}{'m/s2':>10s}{'':>7s}{'':>6s}{'':>10s}{'car':>9s}{'wheel deg':>10s}")
    for name, m in rows:
        m = m & moving
        if m.sum() < 200:
            continue
        gear = int(d.Gear[m].mode().iloc[0]) - 1          # the game numbers reverse 0, neutral 1, first 2
        print(f"{name:24s}{m.sum() / moving.sum():>7.0%}{d.SpeedKMH[m].median():>8.1f}{centripetal[m].median():>10.2f}"
              f"{d.RPM[m].median():>7.0f}{gear:>6d}{d.Throttle[m].median():>10.2f}{wheel_ratio[m].median():>9.2f}"
              f"{d.Steer.abs()[m].median():>10.0f}")
    m = moving
    print(f"\nRPM: 10th / 50th / 90th percentile {d.RPM[m].quantile(.1):.0f} / {d.RPM[m].median():.0f} / {d.RPM[m].quantile(.9):.0f}")
    print(f"throttle: {(d.Throttle[m] >= 0.95).mean():.0%} of the time at 95% or more, {(d.Throttle[m] <= 0.05).mean():.0%} at 5% or less")
    print(f"rear slip angle: median {rear[m].median():.1f} deg, 90th percentile {rear[m].quantile(.9):.1f} deg")
    w = m & (d.DriveTrainSpeed > 40)
    rpm_per_kmh = (d.RPM[w] / d.DriveTrainSpeed[w]).median()
    print(f"engine RPM per km/h of driven-wheel speed: {rpm_per_kmh:.2f}")


if __name__ == '__main__':
    main()
