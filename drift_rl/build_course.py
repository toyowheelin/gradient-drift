"""
build_course.py - works out the COURSE LAYOUT from the track mesh and writes it as two CSV files.

Why this exists: on many drift tracks the paved surface is much wider than the course (Spirit
Peaks, where this was found, among them). The course is the lane between the tire stacks, cones
and barrels standing on that surface. Before this, "on the track" meant "on the paved surface"
(on Spirit Peaks a 25-30 m wide ribbon), so a car that cut inside the stacks or ran wide past
them was, by every measure in this folder, still on the track - and it drove through a stack
about 11 times a minute, because none of its sensors can see them (fan rays leave the car 0.90
m above the ground, pins at 0.55 m; the stacks are 0.51-0.62 m tall).

What it does:
  1. Finds the markers. A marker is a small free-standing object in the triangle mesh:
       tire stack  Wall surface,    0.55-0.80 m across, 0.40-0.70 m tall   (58 triangles' worth)
       cone        Unknown surface, under 0.6 m across, 0.25-0.60 m tall   (24-36 triangles)
       barrel      Unknown surface, under 1 m across,   0.90-1.40 m tall   (116-132 triangles)
  2. Puts each marker on one side of the racing line (the sign of its cross-track offset, the
     same sign convention as the plugin's CrossTrackErrorM) at its distance along the line.
  3. Joins up the markers on each side: the course edge at any point along the line is the
     straight-line interpolation (in distance along the line) between the neighbouring markers on
     that side, or the edge of the paved surface or the first wall where that is nearer (read
     along three parallel rays and the middle one kept, so one crack or gap does not count). With
     the track's collision materials, paving drawn as sand or grass ends it too (OFF_TRACK_RUN_M).
  4. Tidies the edge: no nearer another stretch of the line than its own point (the inside of a
     tight corner, two stretches side by side), and widening at most MAX_EDGE_SLOPE m per m along
     the line, so open run-off reached for a point or two no longer spikes the edge out.

Output (in the track's folder, drift_rl/tracks/<track>/):
  <track>_course_edges.csv    one row per spline point: how far the course extends to
                               each side of the line there, and where that edge is
  <track>_course_markers.csv  one row per marker

    python build_course.py            # writes both files, prints a summary
    python build_course.py --plot     # also writes course_layout.png

    python build_course.py --track <name> --sides <track folder>
        another track: its data in <folder>/data (fast_lane.ai, <name>_triangles.csv.gz, <name>_collision.csv.gz), the course
        edges from the track's own data/side_l.csv and side_r.csv instead of from markers (for a
        track whose walls are the course, such as House of RWD). No markers file entries then.

Needs scipy (only this script does; the trainers read the CSV files).
"""
import argparse
import math
import os
import sys

import numpy as np
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))

# markers further than this from the racing line belong to something else (the paddock, another
# part of the venue), not to the course
MAX_MARKER_DISTANCE_M = 19.0
# never let the course edge come closer to the racing line than this: the line itself is on the
# course by definition, and a marker this close to it is a clipping point, not a wall
MIN_HALF_WIDTH_M = 2.0
PAVED_EDGE_SEARCH_M = 30.0
# a wall crossing the line's normal ends the course there; probed this high above the ground (a low
# tire wall and the gap under a guard rail)
WALL_PROBE_HEIGHTS_M = (0.3, 0.9)
# Each side's edge is read along RAYS_PER_POINT parallel rays RAY_SPREAD_M apart along the line,
# and the middle reading kept: one ray alone read a single crack in the mesh or a gap between two
# wall pieces as the edge (one-point dips to MIN_HALF_WIDTH_M, one-point spikes 15-18 m out).
RAYS_PER_POINT = 3
RAY_SPREAD_M = 0.75
# How fast the edge may move away from the line, in metres sideways per metre along it (0.35 =
# about 19 degrees). An edge that only ends where the paving ends or at the first wall ran 10 - 30 m
# out across open run-off for a point or two and straight back (Klutch Kickers' corners); a real
# course widens gradually. The limit only lowers an edge, never raises one.
MAX_EDGE_SLOPE = 0.35
# The edge may not reach a point nearer another stretch of the line than its own point (minus
# FOLD_TOLERANCE_M): on the inside of a corner tighter than the walk, or between two stretches
# side by side, the edge otherwise crossed into the other stretch's course. The racing-line tools
# of TUM's global race trajectory optimisation refuse track widths whose normals cross for the
# same reason. Line points more than FOLD_MAX_DY_M above or below do not count (a bridge).
FOLD_TOLERANCE_M = 0.5
FOLD_MAX_DY_M = 2.5
# A track's builder sometimes gives the road's surface KEY (counts as track, full grip) to strips
# drawn as sand, cement or grass beside the tarmac: on Klutch Kickers 44% of the road-keyed ground
# is drawn with the satellite-photo ground of the grass around it, and a 1 m sand strip runs along
# most of one side, so the edge ran 1 - 2 m past the visible tarmac. With the track's
# <track>_collision_material.csv.gz the edge also ends where the ground under the ray is drawn with
# an off-track material for OFF_TRACK_RUN_M or more (a painted line or a small patch does not end
# it). Off-track: a material with at least OFF_TRACK_GRASS_SHARE of its ground area under a key that
# does not count as track, or whose name says sand, grass, dirt, gravel, mud, soil or cement (Klutch
# Kickers' aprons; not concrete, which RiverSide's course crosses on a pad). A material
# under at least LINE_MATERIAL_SHARE of the racing line's points never is (RiverSide's course is
# gravel); kerbs, rumble strips and other tarmac stay part of the course.
OFF_TRACK_RUN_M = 0.75
OFF_TRACK_GRASS_SHARE = 0.5
OFF_TRACK_NAME_WORDS = ('sand', 'grass', 'dirt', 'gravel', 'mud', 'soil', 'cement')
LINE_MATERIAL_SHARE = 0.05
# Markers that stand near the course but are NOT part of its edge (X, Z; matched within 1 m).
# From the track's owner, checking course_layout.png: "the cones aren't part of the
# track, just the tires in that spot" - the edge was bridging out to two cones around X 80-92,
# Z -48 to -57, and to one on the opposite side around X 60, Z 40. They stay in the marker list
# (driving through one still counts as a hit); they just no longer pull the edge outwards.
NOT_PART_OF_THE_EDGE = [(92.0, -57.0), (89.9, -49.4), (60.0, 38.5)]


def find_markers(world, fs):
    """Small free-standing objects near the track -> list of dicts (kind, x, z, radius, height, ground_y)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    (tv, ts), sp = world.sensor_triangles(), world.sp_pos
    kd = cKDTree(sp[:, [0, 2]])
    centroid = tv.mean(axis=1)
    dist, _ = kd.query(centroid[:, [0, 2]])
    normal = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ny = np.abs(normal[:, 1]) / np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
    # anything that is not lying flat: the sides of objects standing on the ground
    steep = (dist < 45.0) & (ny < 0.7)
    T, Ts = tv[steep], ts[steep]
    pts = np.concatenate([T.reshape(-1, 3), T.mean(axis=1)])
    tid = np.concatenate([np.repeat(np.arange(len(T)), 3), np.arange(len(T))])
    cell = 0.25
    x0, z0 = pts[:, 0].min(), pts[:, 2].min()
    ix = ((pts[:, 0] - x0) / cell).astype(int)
    iz = ((pts[:, 2] - z0) / cell).astype(int)
    occupied = np.zeros((ix.max() + 1, iz.max() + 1), bool)
    occupied[ix, iz] = True
    labels, _ = ndimage.label(ndimage.binary_dilation(occupied, iterations=1))
    point_label = labels[ix, iz]
    R = world.ray_arrays()

    def ground_around(x, z, y, r):
        vals = []
        for k in range(8):
            a = k * math.pi / 4
            d, _ = fs.ground_below(x + r * math.cos(a), y + 3.0, z + r * math.sin(a), 12.0, *R)
            if d >= 0:
                vals.append(y + 3.0 - d)
        return float(np.median(vals)) if vals else float('nan')

    order = np.argsort(point_label)
    sorted_labels = point_label[order]
    bounds = np.r_[0, np.nonzero(np.diff(sorted_labels))[0] + 1, len(sorted_labels)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        P = pts[idx]
        ex, ez = np.ptp(P[:, 0]), np.ptp(P[:, 2])
        size, small = max(ex, ez), min(ex, ez)
        if size > 1.0 or small < 0.2:
            continue
        cx, cz = float(P[:, 0].mean()), float(P[:, 2].mean())
        gy = ground_around(cx, cz, P[:, 1].max(), size / 2 + 0.4)
        if not np.isfinite(gy):
            continue
        top, bottom = P[:, 1].max() - gy, P[:, 1].min() - gy
        tris = np.unique(tid[idx])
        surface = int(np.bincount(Ts[tris], minlength=5).argmax())
        n = len(tris)
        kind = None
        if surface == fs.SURF_WALL and 0.55 <= size <= 0.80 and 0.40 <= top <= 0.70:
            kind = 'stack'
        elif surface == fs.SURF_UNKNOWN and abs(bottom) < 0.15:
            if 16 <= n <= 40 and size < 0.6 and 0.25 <= top <= 0.60:
                kind = 'cone'
            elif n >= 100 and 0.90 <= top <= 1.40:
                kind = 'barrel'
        if kind:
            out.append(dict(kind=kind, x=cx, z=cz, radius=size / 2, height=float(top), ground_y=gy))
    return out


class Line:
    """The racing line as a closed polyline in the X/Z plane."""

    def __init__(self, sp):
        self.sp = sp
        seg = np.roll(sp, -1, axis=0) - sp
        seg[:, 1] = 0.0
        self.seg = seg
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tan = seg / self.seg_len[:, None]
        self.arc = np.r_[0.0, np.cumsum(self.seg_len)[:-1]]
        self.lap = float(self.seg_len.sum())

    def project(self, x, z):
        """-> (distance along the line, cross-track offset with CrossTrackErrorM's sign, point index)."""
        sp, tan = self.sp, self.tan
        dx, dz = x - sp[:, 0], z - sp[:, 2]
        t = np.clip((dx * tan[:, 0] + dz * tan[:, 2]) / self.seg_len, 0.0, 1.0)
        px, pz = sp[:, 0] + t * self.seg[:, 0], sp[:, 2] + t * self.seg[:, 2]
        i = int(np.argmin((x - px) ** 2 + (z - pz) ** 2))
        lat = tan[i, 0] * (z - pz[i]) - tan[i, 2] * (x - px[i])
        return self.arc[i] + t[i] * self.seg_len[i], lat, i


def fold_limit(line, sign, search_m=PAVED_EDGE_SEARCH_M, step=0.25):
    """For each line point, how far out along its normal (on the `sign` side) the edge may reach
    before a point there is nearer another stretch of the line than this one (FOLD_TOLERANCE_M):
    the inside of a corner tighter than that, or the middle between two stretches side by side.
    Line points more than FOLD_MAX_DY_M higher or lower are left out (a bridge over the course)."""
    from scipy.spatial import cKDTree
    sp = line.sp
    N = len(sp)
    xz = sp[:, [0, 2]]
    kd = cKDTree(xz)
    normal = np.c_[-line.tan[:, 2], line.tan[:, 0]] * sign
    ds = np.arange(step, search_m + step / 2, step)
    limit = np.full(N, search_m)
    k = min(16, N)
    for i in range(N):
        pts = xz[i] + normal[i] * ds[:, None]
        dist, idx = kd.query(pts, k=k)
        level = np.abs(sp[idx, 1] - sp[i, 1]) <= FOLD_MAX_DY_M
        nearest = np.where(level, dist, np.inf).min(axis=1)
        # the segment from point i to its neighbours is where this point's own distance is
        # measured from: a point on the normal is exactly ds away from point i
        crossed = nearest < ds - FOLD_TOLERANCE_M
        if crossed.any():
            j = int(np.argmax(crossed))
            limit[i] = ds[j - 1] if j > 0 else step
    return limit


def slope_limit(width, line, slope=MAX_EDGE_SLOPE):
    """The widths lowered so that no point is more than `slope` x the distance along the line wider
    than any other point (a closed lap): spikes become gentle bulges, narrow places stay narrow."""
    w = np.asarray(width, dtype=np.float64).copy()
    w[~np.isfinite(w)] = PAVED_EDGE_SEARCH_M
    seg = line.seg_len
    # two passes each way around the lap carry the limit across the start/finish
    for _ in range(2):
        for i in range(1, 2 * len(w)):
            a, b = (i - 1) % len(w), i % len(w)
            w[b] = min(w[b], w[a] + slope * seg[a])
        for i in range(2 * len(w) - 2, -1, -1):
            a, b = (i + 1) % len(w), i % len(w)
            w[b] = min(w[b], w[a] + slope * seg[b])
    return w


def off_track_materials(materials, ground_area, grass_area, line_counts, n_line):
    """The set of material names counted as off the track (see OFF_TRACK_GRASS_SHARE): `ground_area`
    and `grass_area` give each material's ground area in all and under keys that do not count as
    track, `line_counts` how many of the racing line's `n_line` points stand on it."""
    off = set()
    for m in materials:
        if line_counts.get(m, 0) >= LINE_MATERIAL_SHARE * n_line:
            continue
        area = ground_area.get(m, 0.0)
        if (area > 0 and grass_area.get(m, 0.0) >= OFF_TRACK_GRASS_SHARE * area) \
                or any(w in m.lower() for w in OFF_TRACK_NAME_WORDS):
            off.add(m)
    return off


def material_lookup(world, fs, materials):
    """(ground_material(x, y, z) -> material name or None, off-track set) from the collision rows'
    material names (<track>_collision_material.csv.gz)."""
    R = world.ray_arrays()
    n0 = world.n_sensor
    surf = world.tri_surf[n0:]
    v = world.tri_v[n0:]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    ground_area, grass_area = {}, {}
    for m, s, a in zip(materials, surf, area):
        if s == fs.SURF_WALL:
            continue
        ground_area[m] = ground_area.get(m, 0.0) + a
        if s == fs.SURF_GRASS:
            grass_area[m] = grass_area.get(m, 0.0) + a

    def ground_material(x, y, z):
        for dx, dz in ((0.0, 0.0), (fs.CRACK_PROBE_M, 0.0), (-fs.CRACK_PROBE_M, 0.0),
                       (0.0, fs.CRACK_PROBE_M), (0.0, -fs.CRACK_PROBE_M)):
            ray_y, left = y + 3.0, 12.0
            for _ in range(12):
                d, idx = fs.raycast_index(x + dx, ray_y, z + dz, 0.0, -1.0, 0.0, left, *R, 1)
                if d < 0.0:
                    break
                if world.tri_surf[idx] != fs.SURF_WALL:
                    return materials[idx - n0]
                ray_y -= d + 0.001
                left -= d + 0.001
                if left <= 0.0:
                    break
        return None

    line_counts = {}
    for p in world.sp_pos:
        m = ground_material(p[0], p[1], p[2])
        if m is not None:
            line_counts[m] = line_counts.get(m, 0) + 1
    off = off_track_materials(set(materials), ground_area, grass_area, line_counts, len(world.sp_pos))
    return ground_material, off


def build(world, fs, de, materials=None):
    """`materials`: each collision row's material name (<track>_collision_material.csv.gz), or None
    (the edge then ends only where the surface KEY stops counting as track, or at a wall)."""
    line = Line(world.sp_pos)
    sp, N = world.sp_pos, len(world.sp_pos)
    markers = find_markers(world, fs)
    for m in markers:
        m['arc'], m['lateral'], m['point'] = line.project(m['x'], m['z'])
        m['used'] = (abs(m['lateral']) < MAX_MARKER_DISTANCE_M
                     and not any(math.hypot(m['x'] - x, m['z'] - z) < 1.0 for x, z in NOT_PART_OF_THE_EDGE))
    track = de.track_surface_code(world)

    R = world.ray_arrays()
    ground_material, off_track = material_lookup(world, fs, materials) if materials is not None else (None, set())
    run_steps = int(round(OFF_TRACK_RUN_M / 0.25))

    def paved_edge(i, sign, along=0.0):
        """How far the paving runs from the line at point i, stopping at the first wall: on a track
        walled along a road that is paved beyond the walls too (a street circuit, House of RWD's
        kind), the paving alone put the edge PAVED_EDGE_SEARCH_M out on both sides. `along` moves
        the ray's start that far along the line (RAYS_PER_POINT)."""
        nx, nz = -line.tan[i, 2] * sign, line.tan[i, 0] * sign
        x0, z0 = sp[i, 0] + line.tan[i, 0] * along, sp[i, 2] + line.tan[i, 2] * along
        step = 0.25
        y = sp[i, 1]
        run_start, run = None, 0       # the off-track drawn material run (OFF_TRACK_RUN_M)
        for d in np.arange(0.5, PAVED_EDGE_SEARCH_M + 0.01, step):
            x, z = x0 + nx * d, z0 + nz * d
            if de.surface_under(world, x, y, z) != track:
                # an off-track strip narrower than OFF_TRACK_RUN_M against the grass is off-track too
                return run_start if run_start is not None else d
            if off_track:
                if ground_material(x, y, z) in off_track:
                    run_start = d if run_start is None else run_start
                    run += 1
                    if run > run_steps:
                        return run_start
                else:
                    run_start, run = None, 0
            down, _ = fs.ground_below(x, y + 3.0, z, 12.0, *R)
            if down >= 0:
                y = y + 3.0 - down
            for h in WALL_PROBE_HEIGHTS_M:
                hit, _ = fs.raycast(x, y + h, z, nx, 0.0, nz, step, *R, 2)
                if hit >= 0:
                    return run_start if run_start is not None else d + hit
        return PAVED_EDGE_SEARCH_M

    half = np.zeros((2, N))        # [0] = positive cross-track side, [1] = negative side
    from_marker = np.zeros((2, N), bool)
    for side, sign in ((0, +1), (1, -1)):
        offsets = (np.arange(RAYS_PER_POINT) - (RAYS_PER_POINT - 1) / 2) * RAY_SPREAD_M
        paved = np.array([np.median([paved_edge(i, sign, a) for a in offsets]) for i in range(N)])
        paved = np.minimum(paved, fold_limit(line, sign))
        ms = sorted((m for m in markers if m['used'] and m['lateral'] * sign > 0), key=lambda m: m['arc'])
        s = np.array([m['arc'] for m in ms])
        w = np.array([abs(m['lateral']) for m in ms])
        if len(s):
            joined = np.interp(line.arc, np.r_[s - line.lap, s, s + line.lap], np.r_[w, w, w])
        else:                      # no markers on this side: the paving (up to a wall) is the course
            joined = np.full(N, np.inf)
        half[side] = np.maximum(slope_limit(np.minimum(joined, paved), line), MIN_HALF_WIDTH_M)
        from_marker[side] = joined <= paved
    normal = np.c_[-line.tan[:, 2], line.tan[:, 0]]
    edge_pos = sp[:, [0, 2]] + normal * half[0][:, None]
    edge_neg = sp[:, [0, 2]] - normal * half[1][:, None]
    return dict(line=line, markers=markers, half=half, from_marker=from_marker, edge_pos=edge_pos, edge_neg=edge_neg,
                off_track=off_track)


def read_sides(folder, sp):
    """The track's two edge lines (data/side_l.csv, data/side_r.csv: X, Y, Z, progress) as X/Z points.
    Some tracks store Z negated against the model (House of RWD): whichever sign puts the edges
    nearer the racing line is used."""
    from scipy.spatial import cKDTree
    sides = [np.loadtxt(os.path.join(folder, 'data', name), delimiter=',', usecols=(0, 2), ndmin=2)
             for name in ('side_l.csv', 'side_r.csv')]
    line = sp[:, [0, 2]]
    best = None
    for zsign in (1.0, -1.0):
        flipped = [s * [1.0, zsign] for s in sides]
        d = np.median(np.concatenate([cKDTree(s).query(line)[0] for s in flipped]))
        if best is None or d < best[0]:
            best = (d, zsign, flipped)
    return best[2], best[1]


def build_from_sides(world, folder):
    """The course edge at each racing-line point: where the line's normal first crosses one of the
    track's edge lines, on each side."""
    line = Line(world.sp_pos)
    sp, N = world.sp_pos, len(world.sp_pos)
    sides, zsign = read_sides(folder, sp)
    segs = np.concatenate([np.stack([s, np.roll(s, -1, axis=0)], axis=1)[:-1] for s in sides])   # (M, 2, 2)
    a, b = segs[:, 0], segs[:, 1]
    normal = np.c_[-line.tan[:, 2], line.tan[:, 0]]
    half = np.full((2, N), np.nan)
    for i in range(N):
        p, n = sp[i, [0, 2]], normal[i]
        # p + t n = a + u (b - a): solve for t (along the normal) and u (along the edge segment)
        e = b - a
        den = n[0] * e[:, 1] - n[1] * e[:, 0]
        ok = np.abs(den) > 1e-9
        rel = a - p
        t = np.where(ok, (rel[:, 0] * e[:, 1] - rel[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), np.nan)
        u = np.where(ok, (rel[:, 0] * n[1] - rel[:, 1] * n[0]) / np.where(ok, den, 1.0), np.nan)
        hit = ok & (u >= 0.0) & (u <= 1.0)
        for side, sign in ((0, 1.0), (1, -1.0)):
            ts = t[hit] * sign
            ts = ts[(ts > 0.0) & (ts < PAVED_EDGE_SEARCH_M)]
            if ts.size:
                half[side, i] = ts.min()
    missing = np.isnan(half)
    for side in (0, 1):
        if missing[side].any():
            good = ~missing[side]
            half[side] = np.interp(line.arc, line.arc[good], half[side, good], period=line.lap)
    for side, sign in ((0, 1.0), (1, -1.0)):
        half[side] = slope_limit(np.minimum(half[side], fold_limit(line, sign)), line)
    half = np.maximum(half, MIN_HALF_WIDTH_M)
    edge_pos = sp[:, [0, 2]] + normal * half[0][:, None]
    edge_neg = sp[:, [0, 2]] - normal * half[1][:, None]
    return dict(line=line, markers=[], half=half, from_marker=np.zeros((2, N), bool), edge_pos=edge_pos,
                edge_neg=edge_neg, zsign=zsign, missing=missing)


def write(course, data_dir, track=None):
    track = track or tf.TRACK
    edges = tf.writable_path(data_dir, 'course_edges', track)
    with open(edges, 'w') as f:
        f.write('PointId,PositiveSideM,NegativeSideM,PositiveEdgeX,PositiveEdgeZ,NegativeEdgeX,NegativeEdgeZ,'
                'PositiveFromMarkers,NegativeFromMarkers\n')
        for i in range(course['half'].shape[1]):
            f.write(f"{i},{course['half'][0, i]:.3f},{course['half'][1, i]:.3f},"
                    f"{course['edge_pos'][i, 0]:.3f},{course['edge_pos'][i, 1]:.3f},"
                    f"{course['edge_neg'][i, 0]:.3f},{course['edge_neg'][i, 1]:.3f},"
                    f"{int(course['from_marker'][0, i])},{int(course['from_marker'][1, i])}\n")
    marks = tf.writable_path(data_dir, 'course_markers', track)
    with open(marks, 'w') as f:
        f.write('Kind,X,Z,RadiusM,HeightM,GroundY,ArcM,CrossTrackM,UsedForEdge\n')
        for m in sorted(course['markers'], key=lambda m: m['arc']):
            f.write(f"{m['kind']},{m['x']:.3f},{m['z']:.3f},{m['radius']:.3f},{m['height']:.3f},{m['ground_y']:.3f},"
                    f"{m['arc']:.2f},{m['lateral']:.3f},{int(m['used'])}\n")
    return edges, marks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    ap.add_argument('--sides', default=None, metavar='TRACK_FOLDER',
                    help="take the edges from this track folder's data/side_l.csv and side_r.csv")
    args = ap.parse_args(argv)
    sys.path.insert(0, _HERE)
    import fast_sim as fs
    import drift_env as de
    track = args.track or tf.TRACK
    # the markers are found in what the track's model shows (sensor_view.py)
    visual = 'visual' if os.path.exists(tf.path(args.tracks, 'visual', track)) else 'triangles'
    world = fs.load_world(args.tracks, track=track, sensors=visual)
    materials = None
    material_path = tf.path(args.tracks, 'collision_material', track)
    if not args.sides and os.path.exists(material_path):
        import pandas as pd
        materials = pd.read_csv(material_path, keep_default_na=False)['Material'].astype(str).tolist()
        if len(materials) != len(world.tri_surf) - world.n_sensor:
            print(f"WARNING: {material_path} has {len(materials)} rows, the collision file "
                  f"{len(world.tri_surf) - world.n_sensor}: made from different track files, so the edge ignores "
                  "how the ground is drawn (run setup_track.py again).", file=sys.stderr)
            materials = None
    course = build_from_sides(world, args.sides) if args.sides else build(world, fs, de, materials)
    edges, marks = write(course, args.tracks, track)
    if args.sides:
        width = course['half'][0] + course['half'][1]
        print(f"edges from the track's side files (Z {'as stored' if course['zsign'] > 0 else 'negated'}); "
              f"no crossing found at {course['missing'][0].sum()} + {course['missing'][1].sum()} points (interpolated)")
        print(f"course width: {width.min():.1f} - {width.max():.1f} m, median {np.median(width):.1f} m; "
              f"left {np.median(course['half'][0]):.1f} m, right {np.median(course['half'][1]):.1f} m (medians)")
        print(f"written: {edges}\n         {marks}")
        if args.plot:
            plot(world, course, track)
        return
    ms = course['markers']
    kinds = {k: sum(1 for m in ms if m['kind'] == k) for k in ('stack', 'cone', 'barrel')}
    used = [m for m in ms if m['used']]
    width = course['half'][0] + course['half'][1]
    print(f"markers found: {kinds}; within {MAX_MARKER_DISTANCE_M:g} m of the line and used: {len(used)} "
          f"({sum(1 for m in used if m['lateral'] > 0)} on the positive cross-track side, "
          f"{sum(1 for m in used if m['lateral'] < 0)} on the negative)")
    print(f"course width: {width.min():.1f} - {width.max():.1f} m, median {np.median(width):.1f} m; "
          f"edge set by markers on {course['from_marker'].mean():.0%} of it, by the paved edge or a wall on the rest")
    if course.get('off_track'):
        print(f"ground drawn as off the track ends the edge: {', '.join(sorted(course['off_track']))}")
    print(f"written: {edges}\n         {marks}")
    if args.plot:
        plot(world, course, track)

def plot(world, course, track):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sp = world.sp_pos
    fig, ax = plt.subplots(figsize=(12, 10))
    ep, en = course['edge_pos'], course['edge_neg']
    ax.fill(np.r_[ep[:, 0], ep[0, 0], en[0, 0], en[::-1, 0]], np.r_[ep[:, 1], ep[0, 1], en[0, 1], en[::-1, 1]],
            color='#bcd6f5', lw=0)
    for e in (ep, en):
        ax.plot(np.r_[e[:, 0], e[0, 0]], np.r_[e[:, 1], e[0, 1]], color='#2a78d6', lw=1.3)
    ax.plot(sp[:, 0], sp[:, 2], color='#8a8983', lw=0.8, ls='--')
    ms = course['markers']
    for kind, marker in (('stack', 'o'), ('cone', '^'), ('barrel', 's')):
        q = [m for m in ms if m['kind'] == kind]
        ax.scatter([m['x'] for m in q], [m['z'] for m in q], s=28, marker=marker, facecolor='white',
                   edgecolor='black', linewidth=1.0, zorder=5, label=kind)
    ax.set_aspect('equal')
    if ms:
        ax.legend(frameon=False)
    ax.set_title(f"Course layout ({'markers' if ms else 'side files'}; blue band); dashed = racing line", loc='left')
    out = os.path.join(_HERE, 'course_layout.png' if track == tf.TRACK else f'course_layout_{track}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"written: {out}")


if __name__ == '__main__':
    main()
