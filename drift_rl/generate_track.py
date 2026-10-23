"""
generate_track.py - makes random closed drift courses with everything the simulator reads for a real
track: the AI line, the course edges, what bounds the course, the ground (with hills and camber),
the road mesh and the outside zones. No model file. For training a policy that has to drive tracks
it has never seen.

    python generate_track.py --count 20 [--seed 1] [--plot] [--tracks <folder>]

writes tracks/gen<V>_<seed>_<n>/ (one folder per course, the same files and names a real track
has; course n of a seed is the same every time, and one already there is kept; V changes
whenever the generator does, so old courses are never reused) and, with --plot, a .png beside each.
train_ppo_gpu.py --generated makes them itself.

Each course draws (the kinds the owner's servers have - ROADMAP):
* its size: technical (40%; 500 - 1400 m of narrow walled road, 9 - 13 m wide, twisting as much as
  Spirit Peaks and RiverSide, its legs as close as 3 m apart wall to wall), tight (30%; 450 - 900 m:
  hairpins, switchbacks (two opposite hairpins a few metres apart), corners, sweepers that sometimes
  open out in the middle, kinks, S bends) or fast (30%;
  1 - 2.5 km: wide sweepers, long straights, a few hairpins);
* what bounds it: walls at the edges (House of RWD), tire stacks on a paved pad with grass beyond
  (Spirit Peaks), or a road edged by grass with walls along part of it further out (RiverSide,
  Ridge, Drift Playground, Klutch Kickers);
* its ground: flat, or hills up to a 12% grade (often with a climb or a drop through each tight
  corner), with or without camber up to 8%;
* a grass-edged road is often narrow (9 - 13 m) and may have things on the grass inside its tight
  corners, which cutting the apex hits.
The pieces' turns add up to one full turn and the straights' lengths are solved so the loop closes;
a course that comes within reach of itself is drawn again. Width (12 - 20 m) varies along the lap
and the AI line runs down the middle.
"""
import argparse
import gzip
import math
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sensor_view import prism   # noqa: E402  (a tire stack, as the sensors see one)

VERSION = 5                 # part of every course's name: a changed generator makes new courses
SPACING_M = 1.5             # AI-line point spacing (real tracks: about 1.5 m)
WALL_HEIGHT_M = 1.0
WALL_STEP_M = 1.5
GROUND_MARGIN_M = 30.0      # ground beyond the course's bounding box
GROUND_CELL_M = 3.0
MESH_CELL_M = 2.0           # road-mesh point spacing
STYLES = {'walled': 0.25, 'stacks': 0.25, 'grass': 0.5}
SIZES = {'technical': 0.4, 'tight': 0.3, 'fast': 0.3}
LAP_M = {'technical': (500.0, 1400.0), 'tight': (450.0, 900.0), 'fast': (1000.0, 2500.0)}
PIECES = {   # kind: (probability, radius range m, turn range deg)
    'tight': {'hairpin': (0.25, (9, 16), (150, 220)), 'corner': (0.30, (14, 45), (60, 150)),
              'sweeper': (0.20, (25, 90), (100, 210)), 'kink': (0.13, (30, 120), (10, 35)),
              'switchback': (0.12, (8, 16), (160, 200))},
    'fast': {'hairpin': (0.10, (12, 20), (140, 200)), 'corner': (0.30, (25, 80), (40, 130)),
             'sweeper': (0.40, (60, 250), (60, 180)), 'kink': (0.20, (80, 300), (10, 40))},
}
STRAIGHT_M = {'tight': (8.0, 60.0, 150.0), 'fast': (30.0, 200.0, 350.0)}   # drawn from, drawn to, longest


def pieces(rng, size):
    """A random list of ('S', length) and ('A', radius, signed turn in radians), turns summing to
    +-2 pi."""
    sense = rng.choice([-1.0, 1.0])
    kinds = PIECES[size]
    names = list(kinds)
    probs = [kinds[k][0] for k in names]
    lo, hi, _ = STRAIGHT_M[size]
    out = []
    while len(out) <= 60:
        out.append(('S', rng.uniform(lo, hi)))
        kind = rng.choice(names, p=probs)
        _, (r0, r1), (d0, d1) = kinds[kind]
        # a turn against the lap's sense is an S bend; it is paid back later
        sign = -sense if rng.random() < (0.45 if kind != 'sweeper' else 0.2) else sense
        r, turn = rng.uniform(r0, r1), sign * math.radians(rng.uniform(d0, d1))
        if kind == 'switchback':
            # two hairpins the opposite way round joined by a short straight, a mountain road's
            # (Drift Playground: +190 deg at 15 m radius, 15 m on, -190 deg at 9.5 - 12 m)
            out += [('A', r, turn), ('S', rng.uniform(6.0, 25.0)),
                    ('A', rng.uniform(r0, r1), -turn * rng.uniform(0.9, 1.1))]
        elif kind == 'sweeper' and rng.random() < 0.5:
            # opens out for a few metres in the middle (House of RWD's left-hander)
            out += [('A', r, turn * 0.5), ('A', r * rng.uniform(2.0, 4.0), sign * math.radians(rng.uniform(4.0, 10.0))),
                    ('A', r * rng.uniform(0.7, 1.2), turn * 0.5)]
        else:
            out.append(('A', r, turn))
        left = sense * 2.0 * math.pi - sum(p[2] for p in out if p[0] == 'A')
        if sum(p[0] == 'A' for p in out) >= 8 and abs(left) < math.radians(220.0) and sense * left > math.radians(40):
            out.append(('S', rng.uniform(lo, hi)))
            out.append(('A', rng.uniform(12.0, 50.0) * (2.0 if size == 'fast' else 1.0), left))
            return out
    return None


def technical_line(rng):
    """A technical course's line: a smooth star-shaped loop (so it never crosses itself) with
    S bends 35 - 75 m long and 4 - 10 m either side snaking along part of it. Measured over 150
    courses: 109 - 235 deg of turning per 100 m and 12 - 44% of the lap under 30 m radius
    (10th - 90th percentile), against Spirit Peaks' 148 and 27%, RiverSide's 239 and 51%."""
    th = np.linspace(0.0, 2.0 * math.pi, 6000, endpoint=False)
    r = np.ones_like(th)
    for k in range(2, int(rng.integers(4, 9)) + 1):
        r += rng.uniform(0.0, 0.5 / k ** 0.5) * np.cos(k * th + rng.uniform(0.0, 2.0 * math.pi))
    r = np.maximum(r, 0.25) * rng.uniform(80.0, 140.0)
    line, lap = resample(np.c_[r * np.cos(th), r * np.sin(th)], SPACING_M)
    n = len(line)
    f = np.roll(line, -1, 0) - np.roll(line, 1, 0)
    f /= np.linalg.norm(f, axis=1)[:, None]
    waves = max(3, int(lap / rng.uniform(35.0, 75.0)))
    share = np.clip(periodic_noise(rng, n, -0.3, 1.0, knots=6), 0.0, 1.0)
    off = rng.uniform(4.0, 10.0) * share * np.sin(2.0 * math.pi * waves * np.arange(n) / n)
    pts = line + np.c_[-f[:, 1], f[:, 0]] * off[:, None]
    return resample(np.r_[pts, pts[:1]], SPACING_M)


def lay_out(parts, straights):
    """Points (every ~0.25 m) of the pieces laid end to end with the given straight lengths."""
    x = z = h = 0.0
    pts = [(x, z)]
    k = 0
    for p in parts:
        if p[0] == 'S':
            L = straights[k]
            k += 1
            n = max(1, int(L / 0.25))
            for i in range(1, n + 1):
                pts.append((x + math.cos(h) * L * i / n, z + math.sin(h) * L * i / n))
            x += math.cos(h) * L
            z += math.sin(h) * L
        else:
            r, turn = p[1], p[2]
            n = max(2, int(abs(turn) * r / 0.25))
            cx, cz = x - math.sin(h) * r * math.copysign(1, turn), z + math.cos(h) * r * math.copysign(1, turn)
            a0 = math.atan2(z - cz, x - cx)
            for i in range(1, n + 1):
                a = a0 + turn * i / n
                pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
            x, z = pts[-1]
            h += turn
    return np.array(pts)


def close(parts, longest):
    """Straight lengths (6 m to longest each) that make the loop close, or None."""
    from scipy.optimize import lsq_linear
    n = sum(1 for p in parts if p[0] == 'S')
    zero = lay_out(parts, np.zeros(n))[-1]           # where the arcs alone end
    h, cols = 0.0, []
    for p in parts:                                  # each straight adds its length along its heading
        if p[0] == 'S':
            cols.append((math.cos(h), math.sin(h)))
        else:
            h += p[2]
    A = np.array(cols).T
    want = np.array([p[1] for p in parts if p[0] == 'S'])
    # least change from the drawn lengths subject to closing: the closure rows weigh heavily
    res = lsq_linear(np.vstack([A * 100.0, np.eye(n)]), np.r_[-zero * 100.0, want], bounds=(6.0, longest))
    if np.hypot(*lay_out(parts, res.x)[-1]) > 0.5:
        return None
    return res.x


def resample(pts, spacing):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    s = np.linspace(0.0, arc[-1], int(arc[-1] / spacing), endpoint=False)
    return np.c_[np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], arc[-1]


def self_clear(line, reach, gap):
    """True when no two parts of the course (out to reach either side) come within gap of each other."""
    n = len(line)
    d = np.linalg.norm(line[:, None, :] - line[None, :, :], axis=2)
    idx = np.arange(n)
    along = np.abs(idx[:, None] - idx[None, :])
    along = np.minimum(along, n - along) * SPACING_M
    need = reach[:, None] + reach[None, :] + gap
    return bool(np.all((d >= need) | (along < 2.5 * need)))


def periodic_noise(rng, n, lo, hi, knots=8):
    """Smooth values in [lo, hi] round a closed loop of n points."""
    k = rng.uniform(lo, hi, knots)
    s = np.linspace(0, knots, n, endpoint=False)
    i = np.floor(s).astype(int)
    t = s - i
    t = t * t * (3 - 2 * t)
    return k[i % knots] * (1 - t) + k[(i + 1) % knots] * t


def heights(rng, line):
    """Height of the line round the lap: flat (30% of courses) or a few smooth hills, scaled to a
    steepest grade of 2 - 12%. On 60% of the hilly ones each tight corner (under 22 m radius) also
    gets a climb or a drop of 4 - 10% through it (Drift Playground's hairpin, 8% up its exit, which
    gen5 cut)."""
    n = len(line)
    if rng.random() < 0.3:
        return np.zeros(n)
    s = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    h = sum(rng.uniform(0.3, 1.0) * np.sin(k * s + rng.uniform(0, 2 * np.pi)) for k in rng.choice([1, 2, 3, 4, 5], 3, replace=False))
    grade = np.abs(np.diff(np.r_[h, h[:1]])) / SPACING_M
    h = h * rng.uniform(0.02, 0.12) / grade.max()
    if rng.random() < 0.6:
        arc = np.arange(n) * SPACING_M
        lap = n * SPACING_M
        for apex in tight_corners(line):
            sigma = rng.uniform(20.0, 40.0)
            # a hill whose steep flank (sigma from its top) lies on the corner: before it (a drop
            # into the corner) or after it (a climb out); its top grade 0.858 A / sigma
            centre = arc[apex] + rng.choice([-1.0, 1.0]) * sigma * rng.uniform(0.8, 1.3)
            d = np.abs(arc - centre) % lap
            d = np.minimum(d, lap - d)
            h += rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.10) * sigma / 0.858 * np.exp(-(d / sigma) ** 2)
        grade = np.abs(np.diff(np.r_[h, h[:1]])) / SPACING_M
        if grade.max() > 0.12:
            h *= 0.12 / grade.max()
    return h


def tight_corners(line, radius_m=22.0):
    """The apex (point of least radius) of each stretch of the line under radius_m."""
    r = radius_3pt(line)
    tight = r < radius_m
    if tight.all() or not tight.any():
        return []
    start = int(np.argmin(tight))                    # walk the loop from a point that is not tight
    out, run = [], []
    for k in range(len(line)):
        i = (start + k) % len(line)
        if tight[i]:
            run.append(i)
        elif run:
            out.append(run[int(np.argmin(r[run]))])
            run = []
    if run:
        out.append(run[int(np.argmin(r[run]))])
    return out


def generate(rng):
    for _ in range(10):                     # a size and kind that will not close: draw them again
        size = str(rng.choice(list(SIZES), p=list(SIZES.values())))
        # a technical course is a narrow walled road whose legs may run close beside each other
        style = 'walled' if size == 'technical' else rng.choice(list(STYLES), p=list(STYLES.values()))
        for _ in range(200):
            if size == 'technical':
                line, lap = technical_line(rng)
                if radius_3pt(line).min() < 6.0:
                    continue
            else:
                parts = pieces(rng, size)
                if parts is None:
                    continue
                straights = close(parts, STRAIGHT_M[size][2])
                if straights is None:
                    continue
                line, lap = resample(lay_out(parts, straights), SPACING_M)
            if not LAP_M[size][0] <= lap <= LAP_M[size][1]:
                continue
            # a grass-edged road is often narrow (RiverSide 10 m, Drift Playground 11 m)
            narrow = size == 'technical' or (style == 'grass' and rng.random() < 0.6)
            w0, w1 = (4.5, 6.5) if narrow else (6.0, 10.0)
            half = np.stack([periodic_noise(rng, len(line), w0, w1), periodic_noise(rng, len(line), w0, w1)])
            # walls or a paved pad further out need room too (a grass road keeps its walls within 12 m)
            extra = {'walled': 0.5, 'stacks': 12.0, 'grass': 12.0}[style]
            if self_clear(line, half.max(axis=0) + extra, 3.0 if size == 'technical' else 4.0):
                return dict(line=line, half=half, size=size, style=str(style), y=heights(rng, line),
                            camber=np.zeros(len(line)) if rng.random() < 0.4 else periodic_noise(rng, len(line), -0.08, 0.08))
    raise RuntimeError('no course found')


def radius_3pt(line):
    a, b, c = np.roll(line, 1, 0), line, np.roll(line, -1, 0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cr = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    r = ab * bc * ca / np.maximum(2.0 * cr, 1e-9)
    # AC's recorded radius is smooth along the line; a little averaging keeps it so
    pad = np.r_[r[-2:], r, r[:2]]
    return np.minimum(np.convolve(np.minimum(pad, 1e4), np.ones(5) / 5, mode='valid'), 1e4)


def write_track(data, name, c, rng):
    import pandas as pd
    from scipy.spatial import cKDTree
    import track_files as tf
    from build_course import Line, write as write_course
    from build_zones import place_zones
    line, half, y, camber, style = c['line'], c['half'], c['y'], c['camber'], c['style']
    n = len(line)
    sp = np.c_[line[:, 0], y, line[:, 1]]
    with open(tf.writable_path(data, 'fast_lane', name), 'wb') as f:       # version -1: x y z radius camber
        f.write(struct.pack('<ii', -1, n))
        for (x, yy, z), r in zip(sp, radius_3pt(line)):
            f.write(struct.pack('<fffff', x, yy, z, r, 0.0))
    L = Line(sp)
    normal = np.c_[-L.tan[:, 2], L.tan[:, 0]]
    edge_pos = line + normal * half[0][:, None]
    edge_neg = line - normal * half[1][:, None]
    tree = cKDTree(line)
    lat_cap = half.max() + 30.0

    def ground_at(px, pz):
        """Height and signed offset from the line (positive = the positive edge's side) of points."""
        _, i = tree.query(np.c_[px, pz])
        lat = (px - line[i, 0]) * normal[i, 0] + (pz - line[i, 1]) * normal[i, 1]
        return y[i] + camber[i] * np.clip(lat, -lat_cap, lat_cap), lat, i

    def along(edge, out, offset, step):
        """Points every step metres along an edge pushed offset outward, with their ground height."""
        e = edge + out * offset
        seg = np.linalg.norm(np.roll(e, -1, 0) - e, axis=1)
        arc = np.r_[0.0, np.cumsum(seg)]
        s = np.arange(0.0, arc[-1], step)
        p = np.c_[np.interp(s, arc, np.r_[e[:, 0], e[0, 0]]), np.interp(s, arc, np.r_[e[:, 1], e[0, 1]])]
        return p, ground_at(p[:, 0], p[:, 1])[0], s / arc[-1]

    markers, sensors, walls = [], [], []
    sides = ((edge_pos, normal, 1.0), (edge_neg, -normal, -1.0))
    if style == 'stacks':
        # tire stacks along both edges: what the sensors see (Wall, below the fan rays); the game
        # does not collide with them
        for edge, out, sign in sides:
            p, gy, frac = along(edge, out, 0.0, 1.0)
            k = 0
            while k < len(p):
                r, h = rng.uniform(0.28, 0.40), rng.uniform(0.40, 0.70)
                i = int(frac[k] * n) % n
                cx, cz = p[k] + out[i] * r
                markers.append(dict(kind='tire stack', x=cx, z=cz, radius=r, height=h, ground_y=gy[k], arc=L.arc[i],
                                    lateral=sign * (half[0 if sign > 0 else 1][i] + r), used=True))
                sensors += prism(cx, gy[k], cz, r, h)
                k += int(rng.uniform(5.0, 14.0))
    elif style in ('walled', 'grass'):
        for edge, out, sign in sides:
            if style == 'walled':
                spans, offset = [(0.0, 1.0)], 0.3
            else:
                # walls along part of a grass-edged road, a few metres out (none on some courses)
                spans, offset = [], rng.uniform(3.0, 10.0)
                for _ in range(rng.integers(0, 4)):
                    a = rng.uniform(0, 1)
                    spans.append((a, a + rng.uniform(0.05, 0.25)))
            p, gy, frac = along(edge, out, offset, WALL_STEP_M)
            inside = np.zeros(len(p), bool)
            for a, b in spans:
                inside |= ((frac >= a) & (frac <= b)) | ((frac + 1.0 >= a) & (frac + 1.0 <= b))
            for k in np.nonzero(inside & np.roll(inside, -1))[0]:
                (ax, az), (bx, bz) = p[k], p[(k + 1) % len(p)]
                ya, yb = gy[k], gy[(k + 1) % len(p)]
                walls.append((ax, ya, az, bx, yb, bz, bx, yb + WALL_HEIGHT_M, bz, 'Wall'))
                walls.append((ax, ya, az, bx, yb + WALL_HEIGHT_M, bz, ax, ya + WALL_HEIGHT_M, az, 'Wall'))
    if style == 'grass':
        # things on the grass inside some tight corners, where cutting the apex hits them (Drift
        # Playground's hairpin): in the sensors' view and the game's collision both
        a, b = np.roll(line, -3, 0) - line, line - np.roll(line, 3, 0)
        curv = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        for apex in tight_corners(line):
            if rng.random() > 0.6:
                continue
            inside = 0 if curv[apex] < 0 else 1                     # the edge on the corner's inside
            out = normal * (1.0 if inside == 0 else -1.0)
            room = radius_3pt(line)[apex] - half[inside][apex]       # from the inside edge to the corner's centre
            for _ in range(int(rng.integers(3, 9))):
                i = (apex + int(rng.integers(-7, 8))) % n
                d = rng.uniform(1.5, min(7.0, room - 1.0)) if room > 3.0 else None
                if d is None:
                    break
                px, pz = line[i] + out[i] * (half[inside][i] + d)
                k = tree.query([px, pz])[1]
                if np.hypot(px - line[k, 0], pz - line[k, 1]) < half[:, k].max() + 1.0:
                    continue                                         # on another part of the road
                r, h = rng.uniform(0.3, 0.7), rng.uniform(0.5, 1.2)
                walls += prism(px, ground_at(np.array([px]), np.array([pz]))[0][0], pz, r, h)
    write_course(dict(half=half, edge_pos=edge_pos, edge_neg=edge_neg,
                      from_marker=np.full((2, n), style == 'stacks'), markers=markers), data, name)
    edges = pd.read_csv(tf.path(data, 'course_edges', name))
    with open(tf.writable_path(data, 'course_zones', name), 'w') as f:
        f.write('Zone,Kind,Side,FirstPointId,LastPointId,LengthM,Corner,CornerTurnsDeg,CornerRadiusM\n')
        for i, z in enumerate(place_zones(edges), 1):
            f.write(f"{i},{z['kind']},{z['side']},{z['first']},{z['last']},{z['length']:.1f},{z['corner']},"
                    f"{z['turned']:.0f},{z['radius']:.0f}\n")
    # the ground: a height field following the line's height and camber; Road where the course's
    # kind is paved (all of it inside walls; the course plus a pad for tire stacks; the course
    # itself for a grass-edged road), Grass beyond
    lo = np.minimum(edge_pos.min(0), edge_neg.min(0)) - GROUND_MARGIN_M
    hi = np.maximum(edge_pos.max(0), edge_neg.max(0)) + GROUND_MARGIN_M
    xs = np.arange(lo[0], hi[0] + GROUND_CELL_M, GROUND_CELL_M)
    zs = np.arange(lo[1], hi[1] + GROUND_CELL_M, GROUND_CELL_M)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    gy = ground_at(gx.ravel(), gz.ravel())[0].reshape(gx.shape)
    cx, cz = np.meshgrid((xs[:-1] + xs[1:]) / 2, (zs[:-1] + zs[1:]) / 2, indexing='ij')
    _, lat, i = ground_at(cx.ravel(), cz.ravel())
    pad = {'walled': 1e9, 'stacks': rng.uniform(4.0, 12.0), 'grass': 0.0}[style]
    paved = ((lat >= 0) & (lat <= half[0][i] + pad)) | ((lat < 0) & (-lat <= half[1][i] + pad))
    kind = np.where(paved, 'Road', 'Grass').reshape(cx.shape)
    near = (np.abs(lat) <= lat_cap).reshape(cx.shape)     # nothing far from the course (a fast
    ground = []                                           # course's infield is mostly empty)
    for a in range(len(xs) - 1):
        for b in range(len(zs) - 1):
            if not near[a, b]:
                continue
            p00 = (xs[a], gy[a, b], zs[b]); p10 = (xs[a + 1], gy[a + 1, b], zs[b])
            p11 = (xs[a + 1], gy[a + 1, b + 1], zs[b + 1]); p01 = (xs[a], gy[a, b + 1], zs[b + 1])
            ground.append((*p00, *p10, *p11, kind[a, b]))
            ground.append((*p00, *p11, *p01, kind[a, b]))
    header = 'X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,SurfaceType\n'
    for name_kind, rows in (('triangles', walls + sensors), ('collision', walls + ground)):
        with gzip.open(tf.writable_path(data, name_kind, name), 'wt') as f:
            f.write(header)
            for r in rows:
                f.write(','.join(f'{v:.4f}' for v in r[:9]) + ',' + r[9] + '\n')
    mx, mz = np.meshgrid(np.arange(lo[0], hi[0], MESH_CELL_M), np.arange(lo[1], hi[1], MESH_CELL_M), indexing='ij')
    my, mlat, _ = ground_at(mx.ravel(), mz.ravel())
    keep = np.abs(mlat) <= half.max() + 2.0
    pd.DataFrame({'X': mx.ravel()[keep], 'Y': my[keep], 'Z': mz.ravel()[keep]}).to_csv(
        tf.writable_path(data, 'road_mesh', name), index=False)
    return edge_pos, edge_neg


def plot(path, c, edge_pos, edge_neg, data, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import track_files as tf
    line = c['line']
    fig, ax = plt.subplots(figsize=(6, 6))
    mk = pd.read_csv(tf.path(data, 'course_markers', name))
    walls = c['style'] == 'walled'
    for e in (edge_pos, edge_neg):
        ax.plot(*np.r_[e, e[:1]].T, color='0.25' if walls else ('#7aa86a' if c['style'] == 'grass' else '0.75'), lw=1.2)
    if len(mk):
        ax.scatter(mk.X, mk.Z, s=4, color='0.15', zorder=3)
    ax.plot(*np.r_[line, line[:1]].T, color='0.6', lw=0.6, ls='--')
    for _, z in pd.read_csv(tf.path(data, 'course_zones', name)).iterrows():
        e = edge_pos if z.Side > 0 else edge_neg
        idx = np.arange(int(z.FirstPointId), int(z.LastPointId) + 1) % len(line)
        ax.plot(*e[idx].T, color='#d95f02', lw=3)
    ax.annotate('', xy=line[8], xytext=line[0], arrowprops=dict(arrowstyle='->', color='#1b9e77', lw=2))
    lap = float(np.linalg.norm(np.diff(np.r_[line, line[:1]], axis=0), axis=1).sum())
    ax.set_title(f"{name}: {c['style']}, {c['size']}, {lap:.0f} m, hills {np.ptp(c['y']):.0f} m, "
                 f"camber to {np.abs(c['camber']).max() * 100:.0f}%", fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def track_name(seed, k):
    return f'gen{VERSION}_{seed}_{k:03d}'


def ensure(data, seed, k, plot_too=False):
    """Makes generated course k of a seed (the same course every time) unless its files are there.
    Returns its name."""
    import track_files as tf
    name = track_name(seed, k)
    if not all(os.path.exists(tf.path(data, kind, name)) for kind in tf.KINDS):
        os.makedirs(data, exist_ok=True)
        rng = np.random.default_rng([seed, k])
        c = generate(rng)
        ep, en = write_track(data, name, c, rng)
        if plot_too:
            plot(os.path.join(tf.folder(data, name), name + '.png'), c, ep, en, data, name)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    import track_files as tf
    tf.tracks_arg(ap)
    ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    for k in range(args.count):
        print(ensure(args.tracks, args.seed, k, args.plot), flush=True)


if __name__ == '__main__':
    main()
