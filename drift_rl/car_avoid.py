"""Car avoidance on pace (the plugin's AvoidCars, CarAvoidance.cs): the reference the plugin checker
(tests/plugin) holds the C# copy to.

A velocity-obstacle test (Fiorini and Shiller 1998; reciprocal form van den Berg, Lin and Manocha
2008, ORCA van den Berg, Guy, Lin and Manocha 2011) worked in the AI line's frame, where both cars
follow the same curves: s along the line, d across it. Each car is a box whose half-extents along and
across the line follow its angle to the line (a car drifting at 40 deg is shorter along the line and
wider across it). The other car moves relative to this one at constant velocity; it would collide
within the horizon when that straight relative path enters the two boxes' Minkowski sum (grown by a
margin) before the horizon (the slab test).

The only control is the bot's pace, so the only answer is to slow down: the bot scales its velocity
by k (1 down to a floor) and takes the largest k that keeps the other car out for the horizon. The
car behind along the line is the one that yields (ORCA's split of responsibility, reduced to who can
act by slowing): a car ahead, or one alongside and ahead, is avoided; a car behind is left to its
own driver or bot. Nothing changes when no k helps except slowing harder: the floor.
"""
from __future__ import annotations

import math

HALF_LENGTH_M = 1.95       # (CAR_FRONT_M + CAR_REAR_M) / 2, drift_env
HALF_WIDTH_M = 0.85        # CAR_HALF_WIDTH_M
K_STEP = 0.02              # the speed shares tried, 1 down to the floor
T_STEP = 0.1               # the times looked at, s, 0 to the horizon


def overlap(ds, dd, th0, th1, margin):
    """Whether two car outlines overlap (separating axis test): one at the origin at angle th0,
    the other at (ds, dd) at angle th1, both grown by margin / 2 all round."""
    hl, hw = HALF_LENGTH_M + 0.5 * margin, HALF_WIDTH_M + 0.5 * margin
    axes = []
    for th in (th0, th1):
        c, s = math.cos(th), math.sin(th)
        axes.append((c, s, -s, c))
    for th, (c, s, nc, ns) in zip((th0, th1), axes):
        for ax, az in ((c, s), (nc, ns)):
            # projections of both boxes on this axis
            r = 0.0
            for (c2, s2, nc2, ns2) in axes:
                r += hl * abs(c2 * ax + s2 * az) + hw * abs(nc2 * ax + ns2 * az)
            if abs(ds * ax + dd * az) > r:
                return False
    return True


def first_contact(ds, dd, us, ud, th0, th1, margin, horizon):
    """The first time looked at (0, T_STEP, ...) when the other car, at (ds, dd) moving at
    (us, ud) relative to this one, overlaps it; inf when not within the horizon."""
    n = int(horizon / T_STEP + 1e-6)
    for j in range(n + 1):
        t = j * T_STEP
        if overlap(ds + us * t, dd + ud * t, th0, th1, margin):
            return t
    return math.inf


def speed_share(own, other, horizon, margin, floor):
    """The share k of its own speed (floor..1) this car drives at to keep clear of other.
    own, other: (s, d, vs, vd, theta) in the line frame, s the distance along the line (other's
    taken relative to own's, wrapped), d across it, vs / vd the velocity along / across the line,
    theta the car's heading minus the line's direction. 1 when the other car is behind, when there
    is no contact within the horizon, or when slowing would not put it off."""
    s0, d0, vs0, vd0, th0 = own
    s1, d1, vs1, vd1, th1 = other
    ds, dd = s1 - s0, d1 - d0
    if ds <= 0.0:
        return 1.0                      # the other car is behind: it yields
    base = first_contact(ds, dd, vs1 - vs0, vd1 - vd0, th0, th1, margin, horizon)
    if base == math.inf:
        return 1.0
    if base == 0.0:
        return floor                    # touching already, the other car ahead: back off
    best_k, best_t = 1.0, base
    for i in range(1, int((1.0 - floor) / K_STEP + 1e-6) + 1):
        k = 1.0 - i * K_STEP
        t = first_contact(ds, dd, vs1 - k * vs0, vd1 - k * vd0, th0, th1, margin, horizon)
        if t == math.inf:
            return k
        if t > best_t:
            best_k, best_t = k, t
    # no share keeps clear: the slowest that puts the contact off, or none when slowing does not help
    return best_k
