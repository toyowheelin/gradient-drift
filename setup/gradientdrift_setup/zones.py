"""A track's drift zones: the outside zones and touch-and-go areas the trainer pays the rear of the
car to fill (<track>_course_zones.csv in the track files folder, drift_rl/build_zones.py). Setup
places them from the course's shape when it makes a track's data; here they are read, changed,
scored against a recorded run and saved. The zone editor (zones_gui.py) and the `zones` command
(cli.py) are two faces of this file.

The plugin paints the zones on the road for players (GradientDriftPlugin/DriftZones.cs, ShowZones;
paint_shapes below is the same geometry). For the bots they only change what the trainer pays for:
a policy drives differently on a track only once it has been trained on that track's zones.

A zone is a dict as build_zones.place_zones makes it: kind ('outside zone' or 'touch and go'),
side (+1 the left edge, -1 the right edge, looking along the direction of travel), first and last
(course point ids, going round the lap; first > last runs past the start), and the corner facts
the automatic placing knew (None for a zone drawn by hand).
"""
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

from . import game  # noqa: F401  (puts drift_rl on the path)
import build_zones as bz  # noqa: E402
import track_files as tf  # noqa: E402

KINDS = bz.ZONE_KINDS
DEPTH_M = bz.ZONE_DEPTH_M
SLIDING_DEG = 20.0          # build_zones.score: the rear only counts in a zone while the car slides this much
MOVING_KMH = 25.0
SHORT_POINTS = 3            # a zone over fewer course points than this is too short to fill
BACKUPS = 'zone_backups'    # in the track's folder: each save keeps the file it replaces here


def side_name(side):
    return 'left' if side > 0 else 'right'


@dataclass
class Course:
    """A track's course (course_edges.csv): its two edges and the distance along it."""
    track: str
    pos: np.ndarray             # (n, 2) left edge x, z
    neg: np.ndarray             # (n, 2) right edge
    edges: object               # the table, as build_zones wants it
    mid: np.ndarray = None
    along: np.ndarray = None    # metres from the start at each point, down the middle

    def __post_init__(self):
        self.mid = 0.5 * (self.pos + self.neg)
        step = np.linalg.norm(np.diff(np.r_[self.mid, self.mid[:1]], axis=0), axis=1)
        self.along = np.r_[0.0, np.cumsum(step)[:-1]]
        self.lap_m = float(step.sum())

    @property
    def n(self):
        return len(self.pos)

    def edge(self, side):
        return self.pos if side > 0 else self.neg

    def points(self, first, last):
        return bz.zone_points(first, last, self.n)

    def nearest_on_edge(self, side, x, z):
        e = self.edge(side)
        return int(np.argmin(np.hypot(e[:, 0] - x, e[:, 1] - z)))

    def nearest_edge(self, x, z):
        """(side, point id, distance m) of the edge point nearest to x, z."""
        dp = np.hypot(self.pos[:, 0] - x, self.pos[:, 1] - z)
        dn = np.hypot(self.neg[:, 0] - x, self.neg[:, 1] - z)
        i, j = int(np.argmin(dp)), int(np.argmin(dn))
        return (1, i, float(dp[i])) if dp[i] <= dn[j] else (-1, j, float(dn[j]))

    def point_at(self, metres):
        """The point nearest to so many metres from the start (wraps round the lap)."""
        m = metres % self.lap_m
        return int(np.argmin(np.abs(self.along - m)))

    def zone_outline(self, zone, depth=DEPTH_M, inset=0.0):
        """(outer, inner): the zone's edge as one smooth curve (inset metres in from it) and the
        line depth metres in from it, (k, 2) each (outline below; DriftZones.Build is the same)."""
        return outline(self.edge(zone['side']), self.edge(-zone['side']), zone['first'], zone['last'], depth, inset)


# The course edge comes point by point from the AI line's points (~2 m apart), and its curvature
# jumps from point to point (Klutch Kickers' corners turned 2 deg/m on average with kinks of
# 8 deg/m), so a box drawn through the raw points looked pieced together from several bends. The
# outline is the edge smoothed along itself (a Gaussian over OUTLINE_SIGMA_POINTS, weights out to
# OUTLINE_HALF_POINTS either side; it moves the edge in by up to ~0.3 m round a tight corner and
# out by at most the inset, so the paint stays on the course), drawn as a Catmull-Rom curve through
# the smoothed points every OUTLINE_STEP_M or less, with its depth measured square to the curve.
OUTLINE_SIGMA_POINTS = 1.5
OUTLINE_HALF_POINTS = 4
OUTLINE_STEP_M = 0.5


def outline_weights():
    k = np.arange(-OUTLINE_HALF_POINTS, OUTLINE_HALF_POINTS + 1)
    w = np.exp(-0.5 * (k / OUTLINE_SIGMA_POINTS) ** 2)
    return w / w.sum()


def outline(edge, other, first, last, depth=DEPTH_M, inset=0.0):
    """(outer, inner) for the zone on `edge` (n, 2) from point first to last round the lap, the
    course's other edge `other` giving the inward way and the width there (see OUTLINE_STEP_M)."""
    n = len(edge)
    count = (last - first) % n + 1
    pad = OUTLINE_HALF_POINTS + 1
    idx = (first - pad + np.arange(count + 2 * pad)) % n
    e = np.asarray(edge, dtype=float)[idx]
    o = np.asarray(other, dtype=float)[idx]
    width = np.linalg.norm(o - e, axis=1)
    toward = (o - e) / np.maximum(width, 1e-6)[:, None]
    w = outline_weights()
    k = OUTLINE_HALF_POINTS
    s = e.copy()
    for i in range(pad - 1, pad + count + 1):
        s[i] = w @ e[i - k:i + k + 1]
    # never further out than the inset (the paint's outer line is drawn that far in)
    d = np.sum((s - e) * toward, axis=1)
    s += np.maximum(-inset - d, 0.0)[:, None] * toward
    core, cw, cu = s[pad - 1:pad + count + 1], width[pad - 1:pad + count + 1], toward[pad - 1:pad + count + 1]
    pts, tans, ws, us = [], [], [], []
    for i in range(1, count):
        p0, p1, p2, p3 = core[i - 1], core[i], core[i + 1], core[i + 2]
        m = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / OUTLINE_STEP_M - 1e-3)))
        for j in range(m):
            t = j / m
            pts.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
            tans.append(0.5 * ((p2 - p0) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t + 3 * (3 * p1 - p0 - 3 * p2 + p3) * t * t))
            ws.append(cw[i] + (cw[i + 1] - cw[i]) * t)
            us.append(cu[i] + (cu[i + 1] - cu[i]) * t)
    pts.append(core[count])
    tans.append(0.5 * (core[count + 1] - core[count - 1]))
    ws.append(cw[count])
    us.append(cu[count])
    pts, tans, ws, us = np.array(pts), np.array(tans), np.array(ws), np.array(us)
    tl = np.linalg.norm(tans, axis=1)
    nrm = np.where(tl[:, None] > 1e-9, np.c_[-tans[:, 1], tans[:, 0]] / np.maximum(tl, 1e-9)[:, None], us)
    nrm *= np.where(np.sum(nrm * us, axis=1) < 0, -1.0, 1.0)[:, None]
    return pts + np.minimum(inset, ws)[:, None] * nrm, pts + np.minimum(depth, ws)[:, None] * nrm


def zones_path(tracks, track):
    return tf.path(tracks, 'course_zones', track)


def has_course(tracks, track):
    return bool(tracks) and os.path.exists(tf.path(tracks, 'course_edges', track))


def load_course(tracks, track):
    import pandas as pd
    edges = pd.read_csv(tf.path(tracks, 'course_edges', track))
    pos, neg = bz.edge_arrays(edges)
    return Course(track=track, pos=pos, neg=neg, edges=edges)


def load(tracks, track):
    """(course, zones, edited) for a track in a track files folder. No zones file: no zones."""
    course = load_course(tracks, track)
    path = zones_path(tracks, track)
    if not os.path.exists(path):
        return course, [], False
    zones, edited = bz.read_zones(path, course.edges)
    return course, zones, edited


def auto(course):
    """The zones build_zones places from the course's shape."""
    return bz.place_zones(course.edges)


def new_zone(course, side, first, last, kind=KINDS[0]):
    """A zone drawn by hand between two points of one edge. Given the wrong way round (against the
    direction of travel), the shorter way between them is taken."""
    first, last = int(first) % course.n, int(last) % course.n
    if (last - first) % course.n > course.n // 2:
        first, last = last, first
    return bz.finish_zone({'kind': kind, 'side': 1 if side > 0 else -1, 'first': first, 'last': last,
                           'corner': None, 'turned': None, 'radius': None}, course.pos, course.neg)


def move(course, zone, first=None, last=None, side=None, kind=None):
    """A copy of zone with an end moved, its side changed, or its kind changed. It stops being the
    automatic placing's zone (the corner facts go) once its ends or side change."""
    z = dict(zone)
    if kind is not None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
        z['kind'] = kind
    if first is not None or last is not None or side is not None:
        z = new_zone(course, zone['side'] if side is None else side,
                     zone['first'] if first is None else first, zone['last'] if last is None else last, z['kind'])
    return bz.finish_zone(z, course.pos, course.neg)


def ordered(course, zones):
    """Zones in order round the lap from the start."""
    return sorted(zones, key=lambda z: (course.along[z['first']], z['side']))


def problems(course, zones):
    """What looks wrong with a set of zones: very short ones, and ones that overlap (the trainer
    reads one zone per point: where two overlap, the later one wins)."""
    out = []
    owner = {}
    for i, z in enumerate(zones, 1):
        pts = course.points(z['first'], z['last'])
        if len(pts) < SHORT_POINTS:
            out.append(f"zone {i} is only {len(pts)} point(s) long")
        for p in pts:
            if p in owner and owner[p] != i:
                out.append(f"zones {owner[p]} and {i} overlap")
                break
        for p in pts:
            owner.setdefault(p, i)
    return list(dict.fromkeys(out))


def same_as_auto(course, zones):
    key = lambda z: (z['kind'], z['side'], int(z['first']), int(z['last']))  # noqa: E731
    return sorted(map(key, zones)) == sorted(map(key, auto(course)))


def save(tracks, track, course, zones):
    """Writes the zones, keeping the file it replaces in the track's zone_backups folder. They are
    marked as hand-edited (so setup's track builder keeps them) unless they are just the automatic
    placing's. Returns (path, backup or None, edited)."""
    path = tf.writable_path(tracks, 'course_zones', track)
    backup = None
    if os.path.exists(path):
        folder = os.path.join(tf.folder(tracks, track), BACKUPS)
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, f"{track}_course_zones_{time.strftime('%Y%m%d_%H%M%S')}")
        backup, k = stem + '.csv', 1
        while os.path.exists(backup):
            backup, k = f'{stem}_{k}.csv', k + 1
        shutil.copy2(path, backup)
    edited = not same_as_auto(course, zones)
    bz.write_zones(path, ordered(course, zones), course.edges, source='edited' if edited else 'auto')
    return path, backup, edited


def describe(course, zone):
    pts = course.points(zone['first'], zone['last'])
    a, b = course.along[zone['first']], course.along[zone['last']]
    corner = (f"; corner turns {zone['turned']:.0f} deg" if zone.get('turned') is not None else '; drawn by hand')
    return (f"{zone['kind']}, {side_name(zone['side'])} edge, {a:.0f} - {b:.0f} m from the start "
            f"(points {zone['first']} - {zone['last']}, {len(pts)}), {zone['length']:.0f} m long{corner}")


# --- painted in game -----------------------------------------------------------------------------

STRIPE_SLANT_M = 2.1        # GradientDriftPlugin/DriftZones.cs: the same numbers
STRIPE_MARGIN_M = 0.6


def paint_shapes(course, zones, stripe_spacing_m=1.6, line_width_m=0.3):
    """What the plugin paints on the road for each zone (DriftZones.Build, the same geometry):
    (outside zone?, frame (k, 2): its outer edge forward then its inner side back, stripes (s, 2, 2):
    each from the outer edge to the inner side STRIPE_SLANT_M further on). The outer line is half
    line_width_m in from the edge (the plugin's ZoneLineWidthMeters) so the paint stays on the course."""
    inset = min(max(line_width_m, 0.05), 1.0) / 2
    out = []
    for z in zones:
        outer, inner = course.zone_outline(z, inset=inset)
        along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(outer, axis=0), axis=1))]
        spacing = max(0.3, stripe_spacing_m) * (1.0 if z['kind'] == KINDS[0] else 2.0)
        stripes = []
        t = STRIPE_MARGIN_M
        while t + STRIPE_SLANT_M <= along[-1] - STRIPE_MARGIN_M + 1e-4:
            a = np.array([np.interp(t, along, outer[:, 0]), np.interp(t, along, outer[:, 1])])
            u = t + STRIPE_SLANT_M
            b = np.array([np.interp(u, along, inner[:, 0]), np.interp(u, along, inner[:, 1])])
            stripes.append((a, b))
            t += spacing
        out.append((z['kind'] == KINDS[0], np.r_[outer, inner[::-1]], np.array(stripes).reshape(-1, 2, 2)))
    return out


# --- recorded runs -----------------------------------------------------------------------------

@dataclass
class Run:
    """One driver's recorded driving: the car's middle and rear axle, and how far it slides."""
    name: str
    x: np.ndarray
    z: np.ndarray
    rear_x: np.ndarray
    rear_z: np.ndarray
    slip_deg: np.ndarray
    breaks: np.ndarray = field(default=None)    # True where the path jumps (a new lap, a reset)


def _breaks(x, z, metres=8.0):
    return np.r_[True, np.hypot(np.diff(x), np.diff(z)) > metres]


def load_runs(path):
    """The runs in a recording: AI Car Monitor's (one run per driver), the plugin's telemetry, or the
    game's telemetry recorder (one run). Rows under MOVING_KMH are left out."""
    import pandas as pd
    head = list(pd.read_csv(path, nrows=1).columns)
    if 'WheelRL_X' in head and 'DriverName' in head:
        d = pd.read_csv(path, usecols=['CarId', 'DriverName', 'X', 'Z', 'SpeedKMH', 'Velocity_X', 'Velocity_Z',
                                       'WheelFL_X', 'WheelFL_Z', 'WheelFR_X', 'WheelFR_Z',
                                       'WheelRL_X', 'WheelRL_Z', 'WheelRR_X', 'WheelRR_Z'])
        d = d[(d.SpeedKMH > MOVING_KMH) & (d.X.abs() < 5000) & (d.Z.abs() < 5000)]
        runs = []
        for name, g in d.groupby(d.DriverName.astype(str), sort=False):
            rx, rz = 0.5 * (g.WheelRL_X + g.WheelRR_X).to_numpy(), 0.5 * (g.WheelRL_Z + g.WheelRR_Z).to_numpy()
            hx, hz = 0.5 * (g.WheelFL_X + g.WheelFR_X).to_numpy() - rx, 0.5 * (g.WheelFL_Z + g.WheelFR_Z).to_numpy() - rz
            vx, vz = g.Velocity_X.to_numpy(), g.Velocity_Z.to_numpy()
            cos = (hx * vx + hz * vz) / np.maximum(np.hypot(hx, hz) * np.hypot(vx, vz), 1e-6)
            x, z = g.X.to_numpy(), g.Z.to_numpy()
            runs.append(Run(name, x, z, rx, rz, np.degrees(np.arccos(np.clip(cos, -1, 1))), _breaks(x, z)))
        return sorted(runs, key=lambda r: -len(r.x))
    x, z, heading, slip = bz.load_laps(path)
    if isinstance(heading, tuple):
        # the game's recorder: the path's direction and the rear tires' slip angle, whose sign to the
        # body is not known. In a drift the nose points into the turn, so the sign that agrees with
        # which way the path is turning most of the time is the right one.
        path_angle, rear_slip = heading
        turning = np.sign(np.gradient(np.unwrap(path_angle)))
        s = 1.0 if np.mean(np.sign(rear_slip) == turning) >= 0.5 else -1.0
        heading = path_angle + s * rear_slip
    back = bz.REAR_AXLE_BEHIND_CENTRE_M
    return [Run(os.path.splitext(os.path.basename(path))[0], x, z, x - back * np.cos(heading),
                z - back * np.sin(heading), slip, _breaks(x, z))]


def score(course, zones, run):
    """Per zone: (filled, rear m inside the edge (median), share past the edge), as build_zones.score
    measures it: of the time the car is on the zone's stretch of the course, the share with its
    rear axle within DEPTH_M of the zone's edge while sliding at least SLIDING_DEG. NaN for a zone
    the run hardly reaches."""
    from scipy.spatial import cKDTree
    _, point = cKDTree(course.mid).query(np.c_[run.x, run.z])
    rows = []
    for zone in zones:
        here = np.isin(point, course.points(zone['first'], zone['last']))
        if here.sum() < 5:
            rows.append((np.nan, np.nan, np.nan))
            continue
        p = point[here]
        edge = course.edge(zone['side'])[p]
        inward = course.edge(-zone['side'])[p] - edge
        inward /= np.maximum(np.linalg.norm(inward, axis=1), 1e-6)[:, None]
        depth = (run.rear_x[here] - edge[:, 0]) * inward[:, 0] + (run.rear_z[here] - edge[:, 1]) * inward[:, 1]
        filled = (depth >= 0.0) & (depth <= DEPTH_M) & (run.slip_deg[here] >= SLIDING_DEG)
        rows.append((float(filled.mean()), float(np.median(depth)), float((depth < 0.0).mean())))
    return rows


def line_score(rows):
    """Formula Drift's 40 points for the line: the mean fill of the zones the run reached x 40."""
    fills = [r[0] for r in rows if not np.isnan(r[0])]
    return 40.0 * float(np.mean(fills)) if fills else float('nan')


def picture(course, zones, out, run=None, title=None):
    """A map of the course and its zones (striped yellow boxes, as the plugin paints them), with a
    run's rear-axle path if given, saved as an SVG (any web browser opens it). Plain text, so the
    packaged setup needs nothing more to draw it."""
    from xml.sax.saxutils import escape
    pts = [course.pos, course.neg] + ([np.c_[run.rear_x, run.rear_z]] if run is not None else [])
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - 25.0, allp.max(0) + 25.0
    size = hi - lo
    scale = 1100.0 / max(size)
    w, h = size * scale

    def xy(p):                      # map from above: x across, z up
        return f"{(p[0] - lo[0]) * scale:.1f},{(hi[1] - p[1]) * scale:.1f}"

    def poly(arr):
        return ' '.join(xy(p) for p in arr)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h + 40:.0f}" '
           f'viewBox="0 -40 {w:.0f} {h + 40:.0f}" font-family="sans-serif">',
           f'<rect x="0" y="-40" width="{w:.0f}" height="{h + 40:.0f}" fill="#6f8f55"/>',
           f'<polygon points="{poly(np.r_[course.pos, course.neg[::-1]])}" fill="#3b3b3d"/>',
           f'<polygon points="{poly([course.pos[-1], course.pos[0], course.neg[0], course.neg[-1]])}" fill="#3b3b3d"/>']
    for e in (course.pos, course.neg):
        svg.append(f'<polygon points="{poly(e)}" fill="none" stroke="white" stroke-width="1.2"/>')
    svg.append(f'<polyline points="{poly([course.pos[0], course.neg[0]])}" stroke="white" stroke-width="2" '
               'stroke-dasharray="4 3"/>')
    step = max(1, course.n // 14)
    for i in range(0, course.n, step):
        a, b = course.mid[i], course.mid[(i + 2) % course.n]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        nrm = np.array([-d[1], d[0]])
        svg.append(f'<polygon points="{poly([a + 4.0 * d, a - 1.5 * nrm, a + 1.5 * nrm])}" fill="#9a9a9a"/>')
    width = max(1.5, 0.3 * scale)           # the frame as the plugin paints it (0.3 m), at least 1.5 px
    for i, (z, (_, frame, stripes)) in enumerate(zip(zones, paint_shapes(course, zones)), 1):
        outer, inner = course.zone_outline(z)
        opacity = 1.0 if z['kind'] == KINDS[0] else 0.75
        svg.append(f'<polygon points="{poly(frame)}" fill="none" stroke="#f5c400" stroke-opacity="{opacity}" '
                   f'stroke-width="{width:.1f}" stroke-linejoin="miter"/>')
        for a, b in stripes:
            svg.append(f'<polyline points="{poly([a, b])}" stroke="#f5c400" stroke-opacity="{opacity}" '
                       f'stroke-width="{0.8 * width:.1f}"/>')
        mid = len(outer) // 2
        d = outer[mid] - inner[mid]
        spot = outer[mid] + (6.0 / max(np.linalg.norm(d), 1e-6)) * d
        x, y = (float(v) for v in xy(spot).split(','))
        svg.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" font-weight="bold" '
                   f'fill="white" stroke="#222" stroke-width="0.6">{i}</text>')
    if run is not None:
        parts, cur = [], []
        for k in range(len(run.rear_x)):
            if run.breaks[k] and cur:
                parts.append(cur)
                cur = []
            cur.append((run.rear_x[k], run.rear_z[k]))
        parts.append(cur)
        for part in parts:
            if len(part) > 1:
                svg.append(f'<polyline points="{poly(part)}" fill="none" stroke="#2a78d6" stroke-width="1.2" '
                           'stroke-opacity="0.85"/>')
    lines = [title or f"{course.track}: {len(zones)} zones, numbered round the lap",
             "solid stripes = outside zone, sparse = touch and go, arrows = direction of travel"
             + (f", blue = {run.name}'s rear axle" if run is not None else '')]
    svg.append(f'<rect x="0" y="-40" width="{w:.0f}" height="40" fill="white"/>'
               f'<text x="8" y="-23" font-size="14" fill="#222">{escape(lines[0])}</text>'
               f'<text x="8" y="-7" font-size="11" fill="#555">{escape(lines[1])}</text></svg>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    return out
