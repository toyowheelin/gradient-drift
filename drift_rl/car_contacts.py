"""Car-on-car contacts and near misses in a plugin telemetry file (EnableTelemetry).

Every tick holds each bot's row (position, heading, speed) and the one human car the plugin logs
beside it. For every pair of cars in the same tick this measures the gap between their outlines
(rectangles CAR_FRONT_M / CAR_REAR_M / CAR_HALF_WIDTH_M, as ChaseInputs.OutlineGap) and sorts each
contact (outlines overlapping, one event per pair until they are 0.5 s apart) by how the cars met:

  behind    the striking car's nose into the other's tail, both travelling the same way: what a
            chase does when it closes too hard (tandem), or a bot not chasing running up a car
  side      side by side (the other car's centre beside the striker, not ahead of it)
  crossing  travel directions more than 60 deg apart (spun car, a car rejoining, wrong way)

The striker is the car whose nose is nearer the contact; "chasing" counts contacts where the
striker is a bot whose telemetry ChaseLeadId (logged since AvoidCars came in) is the
other car. Older files have no chase column: the column then reads "-".

  python drift_rl/car_contacts.py drift_physics_telemetry_*.csv [--near 0.5]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

CAR_FRONT_M, CAR_REAR_M, CAR_HALF_WIDTH_M = 1.80, 2.10, 0.85
HUMAN = 1000  # key offset for the human car (its session id + HUMAN)
TOUCH_GAP_MS = 500


def corners(x, z, h):
    c, s = math.cos(h), math.sin(h)
    out = []
    for f, l in ((CAR_FRONT_M, CAR_HALF_WIDTH_M), (CAR_FRONT_M, -CAR_HALF_WIDTH_M),
                 (-CAR_REAR_M, -CAR_HALF_WIDTH_M), (-CAR_REAR_M, CAR_HALF_WIDTH_M)):
        out.append((x + f * c - l * s, z + f * s + l * c))
    return out


def _seg_dist(p, a, b):
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    t = ((p[0] - ax) * dx + (p[1] - az) * dz) / max(dx * dx + dz * dz, 1e-9)
    t = min(1.0, max(0.0, t))
    return math.hypot(p[0] - ax - t * dx, p[1] - az - t * dz)


def _inside(p, x, z, h):
    c, s = math.cos(h), math.sin(h)
    f = (p[0] - x) * c + (p[1] - z) * s
    l = -(p[0] - x) * s + (p[1] - z) * c
    return -CAR_REAR_M <= f <= CAR_FRONT_M and abs(l) <= CAR_HALF_WIDTH_M


def outline_gap(a, b):
    """Gap between two car rectangles (x, z, heading); <= 0 when they overlap."""
    ca, cb = corners(*a), corners(*b)
    if any(_inside(p, *b) for p in ca) or any(_inside(p, *a) for p in cb):
        return 0.0
    # edges crossing without a corner inside (a cross shape) also overlap
    d = min(min(_seg_dist(p, cb[i], cb[(i + 1) % 4]) for i in range(4)) for p in ca)
    d = min(d, min(min(_seg_dist(p, ca[i], ca[(i + 1) % 4]) for i in range(4)) for p in cb))
    return d


def nose_share(a, b):
    """How far forward b's centre lies on a, -1 (tail) .. 1 (nose)."""
    x, z, h = a
    f = (b[0] - x) * math.cos(h) + (b[1] - z) * math.sin(h)
    return f / max(math.hypot(b[0] - x, b[1] - z), 1e-6)


def load(path):
    cols = ["Tick", "ServerTimeMs", "SessionId", "PosX", "PosZ", "SpeedKmh", "RealHeadingDeg",
            "HumanSessionId", "HumanPosX", "HumanPosZ", "HumanRotationX", "HumanSpeedKmh"]
    head = pd.read_csv(path, nrows=0).columns
    has_chase = "ChaseLeadId" in head
    use = cols + (["ChaseLeadId"] if has_chase else []) + (["PolicyActive"] if "PolicyActive" in head else [])
    df = pd.read_csv(path, usecols=use, low_memory=False)
    return df, has_chase


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--near", type=float, default=0.5, help="near-miss outline gap, m (default 0.5)")
    args = ap.parse_args(argv)

    for path in args.files:
        df, has_chase = load(path)
        df = df[df["SessionId"] != 255]
        ticks = df.groupby("Tick", sort=True)
        events = []                 # (kind, other_is_bot, chasing, striker_kmh, other_kmh, closing)
        near_ticks = defaultdict(int)
        last_touch = {}
        bot_ms = 0.0
        t_prev = None
        human_ms = 0.0
        for tick, g in ticks:
            t = float(g["ServerTimeMs"].iloc[0])
            dt = 0.0 if t_prev is None else min(200.0, t - t_prev)
            t_prev = t
            bot_ms += dt * len(g)
            cars = {}
            for r in g.itertuples(index=False):
                cars[int(r.SessionId)] = (r.PosX, r.PosZ, math.radians(r.RealHeadingDeg), r.SpeedKmh,
                                          int(r.ChaseLeadId) if has_chase else -1)
            r0 = g.iloc[0]
            if int(r0["HumanSessionId"]) >= 0 and not math.isnan(r0["HumanPosX"]):
                human_ms += dt
                cars[HUMAN + int(r0["HumanSessionId"])] = (r0["HumanPosX"], r0["HumanPosZ"],
                                                           r0["HumanRotationX"] + math.pi / 2, r0["HumanSpeedKmh"], -1)
            keys = sorted(cars)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ka, kb = keys[i], keys[j]
                    if ka >= HUMAN and kb >= HUMAN:
                        continue
                    a, b = cars[ka], cars[kb]
                    if math.hypot(a[0] - b[0], a[1] - b[1]) > 8.0:
                        continue
                    gap = outline_gap(a[:3], b[:3])
                    other_is_bot = ka < HUMAN and kb < HUMAN
                    if gap < args.near:
                        near_ticks["bot-bot" if other_is_bot else "bot-player"] += 1
                    if gap > 0.0:
                        continue
                    key = (ka, kb)
                    fresh = key not in last_touch or t - last_touch[key] > TOUCH_GAP_MS
                    last_touch[key] = t
                    if not fresh:
                        continue
                    # striker: the car whose nose points at the other
                    sa, sb = nose_share(a[:3], b[:3]), nose_share(b[:3], a[:3])
                    (ks, s, ko, o) = (ka, a, kb, b) if sa >= sb else (kb, b, ka, a)
                    dh = abs((s[2] - o[2] + math.pi) % (2 * math.pi) - math.pi)
                    share = max(sa, sb)
                    if dh > math.radians(60):
                        kind = "crossing"
                    elif share > 0.7:
                        kind = "behind"
                    else:
                        kind = "side"
                    chasing = "-" if not has_chase else ("yes" if ks < HUMAN and s[4] == (ko % HUMAN) else "no")
                    closing = s[3] - o[3]
                    events.append((kind, other_is_bot, chasing, ks >= HUMAN, s[3], o[3], closing))
        hours = bot_ms / 3.6e6
        print(f"{path}: {len(ticks)} ticks, {hours:.2f} bot-hours, human on {human_ms / 6e4:.1f} min")
        if not events:
            print("  no contacts")
        ev = pd.DataFrame(events, columns=["kind", "bot_bot", "chasing", "player_struck", "striker_kmh",
                                           "other_kmh", "closing_kmh"])
        if len(ev):
            ev["with"] = np.where(ev["bot_bot"], "bot-bot", np.where(ev["player_struck"], "player hits bot", "bot hits player"))
            summary = ev.groupby(["with", "kind", "chasing"]).agg(
                n=("kind", "size"), striker_kmh=("striker_kmh", "median"), closing_kmh=("closing_kmh", "median"))
            print(summary.to_string())
            print(f"  contacts: {len(ev)} ({len(ev) / max(hours, 1e-9):.1f} per bot-hour)")
        for k, v in near_ticks.items():
            print(f"  ticks within {args.near} m ({k}): {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
