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
     that side, or the edge of the paved surface or the first wall where that is nearer.

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


def build(world, fs, de):
    line = Line(world.sp_pos)
    sp, N = world.sp_pos, len(world.sp_pos)
    markers = find_markers(world, fs)
    for m in markers:
        m['arc'], m['lateral'], m['point'] = line.project(m['x'], m['z'])
        m['used'] = (abs(m['lateral']) < MAX_MARKER_DISTANCE_M
                     and not any(math.hypot(m['x'] - x, m['z'] - z) < 1.0 for x, z in NOT_PART_OF_THE_EDGE))
    track = de.track_surface_code(world)

    R = world.ray_arrays()

    def paved_edge(i, sign):
        """How far the paving runs from the line at point i, stopping at the first wall: on a track
        walled along a road that is paved beyond the walls too (a street circuit, House of RWD's
        kind), the paving alone put the edge PAVED_EDGE_SEARCH_M out on both sides."""
        nx, nz = -line.tan[i, 2] * sign, line.tan[i, 0] * sign
        step = 0.25
        y = sp[i, 1]
        for d in np.arange(0.5, PAVED_EDGE_SEARCH_M + 0.01, step):
            x, z = sp[i, 0] + nx * d, sp[i, 2] + nz * d
            if de.surface_under(world, x, y, z) != track:
                return d
            down, _ = fs.ground_below(x, y + 3.0, z, 12.0, *R)
            if down >= 0:
                y = y + 3.0 - down
            for h in WALL_PROBE_HEIGHTS_M:
                hit, _ = fs.raycast(x, y + h, z, nx, 0.0, nz, step, *R, 2)
                if hit >= 0:
                    return d + hit
        return PAVED_EDGE_SEARCH_M

    half = np.zeros((2, N))        # [0] = positive cross-track side, [1] = negative side
    from_marker = np.zeros((2, N), bool)
    for side, sign in ((0, +1), (1, -1)):
        paved = np.array([paved_edge(i, sign) for i in range(N)])
        ms = sorted((m for m in markers if m['used'] and m['lateral'] * sign > 0), key=lambda m: m['arc'])
        s = np.array([m['arc'] for m in ms])
        w = np.array([abs(m['lateral']) for m in ms])
        if len(s):
            joined = np.interp(line.arc, np.r_[s - line.lap, s, s + line.lap], np.r_[w, w, w])
        else:                      # no markers on this side: the paving (up to a wall) is the course
            joined = np.full(N, np.inf)
        half[side] = np.maximum(np.minimum(joined, paved), MIN_HALF_WIDTH_M)
        from_marker[side] = joined <= paved
    normal = np.c_[-line.tan[:, 2], line.tan[:, 0]]
    edge_pos = sp[:, [0, 2]] + normal * half[0][:, None]
    edge_neg = sp[:, [0, 2]] - normal * half[1][:, None]
    return dict(line=line, markers=markers, half=half, from_marker=from_marker, edge_pos=edge_pos, edge_neg=edge_neg)


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
    course = build_from_sides(world, args.sides) if args.sides else build(world, fs, de)
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
