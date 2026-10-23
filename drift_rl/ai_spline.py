"""
ai_spline.py - Python port of AssettoServer's own AI spline system (FastLaneParser.cs,
AiSpline.cs, SplinePointOperations.cs, and the FindInterpolationSegment helper from
GradientDriftPlugin.cs itself) - the racing line the base game ships with the track, which the
plugin queries every tick to know which way the track goes from wherever the car actually is.

STATUS: VALIDATED against real data, on Spirit Peaks: loads its fast_lane.ai (548 points,
confirmed via byte-offset alignment, plausible position/radius/camber ranges, and an exact
cross-match of the decoded minimum radius against telemetry's own recorded UpcomingMinRadius
minimum). raw_spline_angle() checked against every real recorded RawSplineAngle across 56,538
pooled ticks (5 real runs): median error 0.002deg, p99 0.017deg, max 4.27deg - essentially
float32(C#)-vs-float64(Python) rounding noise. See validate_ai_spline.py for the full
methodology, including a tick-alignment gotcha worth knowing before calling this against
telemetry again: a given CSV row's PosX/PosY/PosZ is the position AFTER that tick's own
movement/collision integration (GradientDriftPlugin.cs lines 2115/2183/2614), one step ahead of the
position its OWN RawSplineAngle was actually computed from (line 735, before that integration)
- so validating row N's RawSplineAngle means calling raw_spline_angle() on row N-1's position,
not row N's own. This module's own API is unaffected (raw_spline_angle(position) always means
exactly the position you pass it); this only matters when reconstructing "what position
produced this telemetry row's angle" after the fact.

WHY NOT RECONSTRUCT THE SPLINE FROM TELEMETRY INSTEAD: tried first, before reading this far
into source - telemetry's CurrentPointId column turned out to be aiState.CurrentSplinePointId,
which GradientDriftPlugin.cs's own comment documents as unreliable for this: it's the BASE GAME's
internal tracking, advanced by the base game's OWN speed assumption, independent of this
plugin's actual speed - confirmed on real data (same point ID showing up to ~100m apart in a
single continuous run). This is exactly why the plugin stopped using it for direction and
switched to _spline.WorldToSpline(position) instead (see GradientDriftPlugin.cs's "CORE FIX"
comment) - reconstructing from the unreliable column would rebuild the bug the live plugin
already moved away from, so this ports the REPLACEMENT mechanism instead, against the real
spline file, once available.

FORMAT (from FastLaneParser.cs - AC's native fast_lane.ai binary format, version 7 is what
every modern track ships, version -1 is an older/simpler variant also supported):
  int32 version
  -- v7 --
  int32 detailCount, int32 lapTime(unused), int32 sampleCount(unused)
  per point (detailCount times): 3x float32 position(x,y,z), float32 length(unused, recomputed
    below), int32 id(unused, reassigned below)
  int32 extraCount (must equal detailCount)
  per point (detailCount times, SAME order): float32 speed(unused), gas(unused), brake(unused),
    obsoleteLatG(unused), float32 radius(KEPT), sideLeft(unused), sideRight(unused),
    float32 camber_raw, float32 camber_direction (camber = raw*direction, KEPT), 3x float32
    normal(unused), float32 length(KEPT, overwrites the first one), 3x float32
    forwardVector(unused - recomputed from neighbor positions instead), tag(unused),
    grade(unused)
  -- v-1 (older/simpler) --
  int32 detailCount; per point: 3x float32 position, float32 radius, float32 camber

POST-PROCESSING (also from FastLaneParser.cs, applied after either format loads):
  - PreviousId/NextId assigned sequentially, circularly (point 0's previous = last point,
    last point's next = point 0) - this is RAW, before the open/closed check below.
  - Length recomputed per point as the real distance to NextId's position (overwrites
    whatever was in the file).
  - Last point's radius reset to 1000 if the file had it under 1 (point-to-point splines
    sometimes leave it near-zero/garbage).
  - CLOSED-LOOP CHECK: if distance(point[0], point[-1]) >= 50, the spline is NOT closed -
    point[0].PreviousId and point[-1].NextId are both set to -1, breaking the circular link
    assigned above, and point[-1].Length is forced to 1. A drift track's line is usually a
    closed circuit, but this isn't assumed - it's checked the same way the real loader checks it.
"""
import struct
import math
from dataclasses import dataclass, field


@dataclass
class SplinePoint:
    id: int
    position: tuple  # (x, y, z)
    radius: float = 0.0
    camber: float = 0.0
    length: float = 0.0
    previous_id: int = -1
    next_id: int = -1
    left_id: int = -1
    right_id: int = -1


def _read_f32(f):
    return struct.unpack('<f', f.read(4))[0]


def _read_i32(f):
    return struct.unpack('<i', f.read(4))[0]


def _read_vec3(f):
    return (_read_f32(f), _read_f32(f), _read_f32(f))


def _from_file_v7(f):
    detail_count = _read_i32(f)
    _read_i32(f)  # LapTime, unused
    _read_i32(f)  # SampleCount, unused

    points = []
    for i in range(detail_count):
        pos = _read_vec3(f)
        _read_f32(f)  # Length (file copy, unused - recomputed below)
        _read_i32(f)  # Id (file copy, unused - reassigned below)
        points.append(SplinePoint(id=i, position=pos))

    extra_count = _read_i32(f)
    if extra_count != detail_count:
        raise ValueError(f"extraCount ({extra_count}) != detailCount ({detail_count}) - "
                          f"malformed or unsupported fast_lane.ai")

    for i in range(detail_count):
        _read_f32(f)  # Speed, unused
        _read_f32(f)  # Gas, unused
        _read_f32(f)  # Brake, unused
        _read_f32(f)  # ObsoleteLatG, unused
        radius = _read_f32(f)
        _read_f32(f)  # SideLeft, unused
        _read_f32(f)  # SideRight, unused
        camber_raw = _read_f32(f)
        camber_direction = _read_f32(f)
        _read_vec3(f)  # Normal, unused
        length = _read_f32(f)
        _read_vec3(f)  # ForwardVector, unused (recomputed from neighbor positions)
        _read_f32(f)  # Tag, unused
        _read_f32(f)  # Grade, unused

        points[i].radius = radius
        points[i].camber = camber_raw * camber_direction
        points[i].length = length

    return points


def _from_file_vn1(f):
    detail_count = _read_i32(f)
    points = []
    for i in range(detail_count):
        pos = _read_vec3(f)
        radius = _read_f32(f)
        camber = _read_f32(f)
        points.append(SplinePoint(id=i, position=pos, radius=radius, camber=camber))
    return points


def load_fast_lane(path):
    """Loads a single fast_lane*.ai file (NOT the .aip zip package - add zipfile handling
    here if a track turns out to ship that form instead). Returns a list of SplinePoint
    with PreviousId/NextId/Length/closed-loop handling already applied, matching
    FastLaneParser.FromFile's post-processing exactly."""
    with open(path, 'rb') as f:
        version = _read_i32(f)
        if version == 7:
            points = _from_file_v7(f)
        elif version == -1:
            points = _from_file_vn1(f)
        else:
            raise ValueError(f"unknown fast_lane.ai version {version} (expected 7 or -1)")

    n = len(points)
    for i in range(n):
        if i == n - 1 and points[i].radius < 1:
            points[i].radius = 1000.0
        points[i].previous_id = points[i - 1].id if i > 0 else points[n - 1].id
        next_point = points[(i + 1) % n]
        points[i].next_id = next_point.id
        points[i].length = _dist(points[i].position, next_point.position)

    closed_loop = _dist(points[0].position, points[-1].position) < 50.0
    if not closed_loop:
        points[0].previous_id = -1
        points[-1].next_id = -1
        points[-1].length = 1.0

    return points


def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _length_sq(a):
    return _dot(a, a)


class Spline:
    """Thin wrapper mirroring AiSpline's query surface (WorldToSpline, Operations.
    GetForwardVector) plus the plugin's own FindInterpolationSegment helper - everything
    GradientDriftPlugin.cs actually calls to turn a position into a target heading."""

    def __init__(self, points):
        self.points = points  # index == id, by construction above

    def world_to_spline(self, position):
        """Direct port of AiSpline.WorldToSpline - nearest point by Euclidean distance.
        Real AssettoServer uses a KD-tree for this (Supercluster.KDTree) since it may run
        against thousands of points across multiple traffic splines; a plain linear scan is
        exactly equivalent and plenty fast at ~500-row scale, so there's no reason to port a
        KD-tree just to re-derive the same answer slower to write and no faster to run here."""
        best_id, best_dist_sq = -1, float('inf')
        for p in self.points:
            d = (p.position[0]-position[0])**2 + (p.position[1]-position[1])**2 + (p.position[2]-position[2])**2
            if d < best_dist_sq:
                best_dist_sq = d
                best_id = p.id
        return best_id, best_dist_sq

    def get_forward_vector(self, point_id):
        """Direct port of SplinePointOperations.GetForwardVector."""
        point = self.points[point_id]
        if point.next_id >= 0:
            return _sub(self.points[point.next_id].position, point.position)
        return (0.0, 0.0, 0.0)

    def find_interpolation_segment(self, our_position, nearest_point_id):
        """Direct port of GradientDriftPlugin.cs's own FindInterpolationSegment: checks BOTH the
        segment to NextId and the segment from PreviousId, projects our_position onto each
        (clamped to the segment), and returns whichever segment's projection is actually
        closest - not just "the nearest point's own forward vector", so direction stays
        smooth as the car's actual position moves along a segment rather than snapping at
        each point. Returns (pointA, pointB, t) for a t-blend between the two points'
        forward vectors."""
        if nearest_point_id < 0 or nearest_point_id >= len(self.points):
            return nearest_point_id, nearest_point_id, 0.0

        nearest = self.points[nearest_point_id]
        best_dist_sq = float('inf')
        best_a, best_b, best_t = nearest_point_id, nearest_point_id, 0.0

        if nearest.next_id >= 0:
            next_pos = self.points[nearest.next_id].position
            segment = _sub(next_pos, nearest.position)
            seg_len_sq = _length_sq(segment)
            if seg_len_sq > 0.0001:
                t = _dot(_sub(our_position, nearest.position), segment) / seg_len_sq
                t = max(0.0, min(1.0, t))
                projected = tuple(nearest.position[i] + segment[i]*t for i in range(3))
                dist_sq = sum((our_position[i]-projected[i])**2 for i in range(3))
                if dist_sq < best_dist_sq:
                    best_dist_sq, best_a, best_b, best_t = dist_sq, nearest_point_id, nearest.next_id, t

        if nearest.previous_id >= 0:
            prev_pos = self.points[nearest.previous_id].position
            segment = _sub(nearest.position, prev_pos)
            seg_len_sq = _length_sq(segment)
            if seg_len_sq > 0.0001:
                t = _dot(_sub(our_position, prev_pos), segment) / seg_len_sq
                t = max(0.0, min(1.0, t))
                projected = tuple(prev_pos[i] + segment[i]*t for i in range(3))
                dist_sq = sum((our_position[i]-projected[i])**2 for i in range(3))
                if dist_sq < best_dist_sq:
                    best_dist_sq, best_a, best_b, best_t = dist_sq, nearest.previous_id, nearest_point_id, t

        return best_a, best_b, best_t

    def raw_spline_angle(self, position, reverse_direction=False):
        """End-to-end port of the exact sequence GradientDriftPlugin.cs runs each tick (the
        "CORE FIX" block): find our nearest point, find the best interpolation segment from
        OUR actual position (not just snapping to the nearest point), blend that segment's
        two endpoints' forward vectors by t, flatten to horizontal, normalize, atan2. Returns
        radians, atan2(Z,X) convention - directly comparable to RawSplineAngle in telemetry."""
        nearest_id, _ = self.world_to_spline(position)
        a, b, t = self.find_interpolation_segment(position, nearest_id)
        fa = self.get_forward_vector(a)
        fb = self.get_forward_vector(b)
        raw_forward = tuple(fa[i] + (fb[i]-fa[i])*t for i in range(3))
        if reverse_direction:
            raw_forward = tuple(-c for c in raw_forward)
        horiz = (raw_forward[0], 0.0, raw_forward[2])
        norm = math.sqrt(horiz[0]**2 + horiz[2]**2)
        if norm < 0.01:
            return None  # degenerate (spline end with Zero forward vector) - caller must fall back
        return math.atan2(horiz[2]/norm, horiz[0]/norm)
