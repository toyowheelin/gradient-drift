"""
pace_model.py - a course's reference speed from its shape and the car's numbers, with no policy.

What a pace of 1.0 asks for at each AI-line point (progress speed along the line, km/h): the
speed a quick human drifts each part of the course at, a little faster (HUMAN_MARGIN). It replaces
measuring the reference with a policy driving flat out (runs/gen3/policy.json until 2026-10-18),
which asked whatever that policy did: where it cut a corner it asked the impossible (Drift
Playground's 9 - 11 m hairpin, ~70 km/h asked, the owner drifts it at 35 - 46), and a policy file
had to travel with every install for setup to make a new track's reference.

The model, per point:
  * cornering: v = sqrt(CORNER_GRIP x mu x g / k), k the line's curvature averaged over
    CURVE_WINDOW_M (a drift takes a wider arc than the AI line's tightest bend), mu the car's
    sideways grip at its static loads (both axles, weighted by load);
  * accelerating out of a corner: CORNER_ACCEL x the drive force the rear tires can put down
    (the engine's force curve when the car has one, never more than the rear's grip) over the mass;
  * braking into one: BRAKE_GRIP x mu x g;
  * a drift's line: it runs RADIUS_OFFSET_M wider than the AI line's radius (which matters only in
    tight corners), and its slowest point comes LAG_M after the geometry's, since a drift scrubs its
    speed through the corner rather than braking it all off before (pace_model 1 asked Drift
    Playground's hairpin for 34 - 41 km/h where the owner carries 39 - 51, and gen14 braked for it
    and drove the exit straight).
The constants are fitted to the owner's recorded laps in his E36 Touring on Drift Playground (6
laps, 2026-10-16) and Spirit Peaks (5 laps, 2026-10-06), each compared at the smoothing its
recording has (Drift Playground +-4 m, Spirit Peaks +-20 m); pace_model 2 was fitted to both at
+-20 m, which filled in the hairpin's apex dip and asked ~37 km/h where the owner carries 23 - 33.
Lab notebook, "A pace reference without gen3", "The hairpin and pace_model 2" and "pace_model 3".

A car's PACE_SCALE (its speed against the reference car's) comes from the same model: the default
car (fast_sim.make_params) is the reference.
"""
import math

import numpy as np

import fast_sim as fs

MODEL = 'pace_model 3'          # what policy files and car files record as their pace reference
HUMAN_MARGIN = 1.05             # the owner: the bots should max out "slightly faster than me"
CORNER_GRIP = 0.606             # share of the tires' grip a drift corners on
CURVE_WINDOW_M = 15.3           # the curvature is the heading change over this much line, per metre
CORNER_ACCEL = 0.324            # share of the rear's drive force that speeds the car up while drifting
BRAKE_GRIP = 0.308              # share of the tires' grip that slows it into a corner
RADIUS_OFFSET_M = 0.63          # a drift's line runs this much wider than the AI line's radius
LAG_M = 6.1                     # and reaches its slowest this far after the AI line's tightest point
MAX_SPEED_MS = 60.0
G = 9.81


def _axle_mu(peak, per_kn, expy, fz0, axle_load_n):
    wheel = 0.5 * axle_load_n
    if expy > 0.0 and fz0 > 0.0:                 # the newer tyre model (fast_sim LS_EXPY)
        return peak * (wheel / fz0) ** (expy - 1.0)
    return peak + per_kn * wheel / 1000.0


def car_terms(car=None):
    """The numbers the model reads from a car: a car_param_values() dict (fast_sim names), or None
    for the default car. mu = sideways grip at static load, weighted by axle load."""
    values = {name: float(fs.make_params()[i]) for name, i in fs.P.items()}
    values.update({k: float(v) for k, v in (car or {}).items() if k in fs.P})
    fz_f, fz_r = values['FRONT_STATIC_LOAD_N'], values['REAR_STATIC_LOAD_N']
    mu_f = _axle_mu(values['LAT_FRONT_PEAK_FRICTION'], values['FRONT_LOAD_SENSITIVITY_PER_KN'],
                    values['FRONT_LS_EXPY'], values['FRONT_FZ0_N'], fz_f)
    mu_r = _axle_mu(values['LAT_REAR_PEAK_FRICTION'], values['REAR_LOAD_SENSITIVITY_PER_KN'],
                    values['REAR_LS_EXPY'], values['REAR_FZ0_N'], fz_r)
    engine = (np.array([values[f'ENGINE_FORCE_{k:02d}'] for k in range(fs.ENGINE_CURVE_POINTS)])
              if values['EXT_ENGINE'] > 0.5 else None)
    return dict(mu=(mu_f * fz_f + mu_r * fz_r) / (fz_f + fz_r), rear_grip_n=mu_r * fz_r,
                mass=(fz_f + fz_r) / G, engine=engine)


def curvature(world, window_m=CURVE_WINDOW_M):
    """|heading change| per metre of the AI line, over window_m centred on each point (round the loop)."""
    xz = world.sp_pos[:, [0, 2]]
    step = np.roll(xz, -1, axis=0) - xz
    heading = np.arctan2(step[:, 1], step[:, 0])
    turn = np.diff(np.r_[heading, heading[0]])
    turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
    lengths = np.asarray(world.sp_length, dtype=np.float64)
    arc = np.r_[0.0, np.cumsum(lengths[:-1])]
    lap = float(lengths.sum())
    cum = np.r_[0.0, np.cumsum(turn)]
    knots = np.r_[arc, lap]

    def heading_at(a):
        laps = np.floor(a / lap)
        return np.interp(a - laps * lap, knots, cum) + laps * cum[-1]
    return np.abs(heading_at(arc + 0.5 * window_m) - heading_at(arc - 0.5 * window_m)) / window_m


def course_speed(world, car=None, margin=HUMAN_MARGIN):
    """The reference speed per AI-line point, km/h, for this car (None = the default car)."""
    t = car_terms(car)
    lengths = np.asarray(world.sp_length, dtype=np.float64)
    n = len(lengths)
    k = curvature(world)
    k = k / (1.0 + RADIUS_OFFSET_M * k)
    v = np.minimum(np.sqrt(CORNER_GRIP * t['mu'] * G / np.maximum(k, 1e-4)), MAX_SPEED_MS)

    def accel(speed):
        force = t['rear_grip_n']
        if t['engine'] is not None:
            x = speed / fs.ENGINE_CURVE_STEP_MS
            k = min(int(x), fs.ENGINE_CURVE_POINTS - 2)
            w = min(x - k, 1.0)
            force = min(force, t['engine'][k] * (1.0 - w) + t['engine'][k + 1] * w)
        return CORNER_ACCEL * max(force, 0.0) / t['mass']
    brake = BRAKE_GRIP * t['mu'] * G
    for _ in range(2):                           # twice round, so the start/finish joins up
        for i in range(1, 2 * n):
            j, prev = i % n, (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[prev] ** 2 + 2.0 * accel(v[prev]) * lengths[prev]))
        for i in range(2 * n - 2, -1, -1):
            j, nxt = i % n, (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[nxt] ** 2 + 2.0 * brake * lengths[j]))
    arc = np.r_[0.0, np.cumsum(lengths[:-1])]
    lap = float(lengths.sum())
    v = np.interp((arc - LAG_M) % lap, np.r_[arc, lap], np.r_[v, v[0]])
    return np.maximum(v * 3.6 * margin, 5.0)


def mean_speed(world, kmh):
    """The lap's average speed, km/h, at these speeds per point."""
    lengths = np.asarray(world.sp_length, dtype=np.float64)
    return float(lengths.sum() / np.sum(lengths / (np.maximum(kmh, 1.0) / 3.6)) * 3.6)


def car_pace_scale(worlds, car):
    """This car's PACE_SCALE: its mean speed round each world over the default car's, averaged."""
    return float(np.mean([mean_speed(w, course_speed(w, car)) / mean_speed(w, course_speed(w)) for w in worlds]))
