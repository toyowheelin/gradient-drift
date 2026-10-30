"""
fast_sim.py - the plugin's vehicle model and sensing, numba-compiled, for training the drift AI.

One tick does what GradientDriftPlugin.cs does for a car a policy drives: where the car is on the AI
spline (with the NearestPointCorrector and the angle-glitch correction), the policy's steering
angle and speed action, the lateral dynamics in sub-steps, the wrong-way recovery, the wall
push-out, the car's height (ground ray, falling back to the spline / road-mesh blend), and the
fan rays and proximity pins for the next observation. The plugin settings a policy file carries
switch the rest on: EnableRearThrustLimit, EnableRearBrakeLimit, MaxSteerRateDegS and
EnableRearFrictionCircle (see step_post).

The actions: steering angle as a fraction of full lock; either the target speed as a fraction of
the car's MaxSpeed (-1 stop, +1 MaxSpeed) or, with EnableRearFrictionCircle, a pedal (thrust above
zero, brakes below); and (a policy with three outputs, EnableRearFrictionCircle only) a brake that
works alongside the throttle - left-foot braking, on the front axle (see step_post).

Checked against the server with replay_live.py (one tick at a time from real telemetry) and
check_plugin_policy.py.
"""
import math
import sys
import numpy as np
import sim_target

_TARGET, njit = sim_target.get_njit(globals())     # CPU normally; CUDA when loaded as the GPU variant

# ----------------------------------------------------------------------------------------
# State vector layout (one float64 row per car). NaN = no history yet.
# ----------------------------------------------------------------------------------------
S_X, S_Y, S_Z = 0, 1, 2
S_HEADING = 3
S_YAW = 4
S_LATV = 5
S_OWN_SPEED = 6
S_CORR_ANGLE = 7        # NaN = first tick
S_PIN0 = 8              # 6 pin distances: 8..13
S_MAX_SPEED = 14
S_PREV_NEAREST = 15     # -1 = none (NearestPointCorrector)
S_NP_STREAK = 16
S_SPIN_STREAK = 17      # the wrong-way recovery's entry count and flag (named after the plugin's
S_SPIN_RECOVERING = 18  # older spin recovery, whose two slots and telemetry flag it took over)
S_RAY0 = 19             # 15 fan distances from the last cast: 19..33
S_FAN_CENTER = 34
S_PINSURF0 = 35         # 6 pin surface codes: 35..40
S_ARC = 41              # arc-length position along the spline, metres (for progress)
S_LAST_STEER = 42       # last tick's final steer command, rad (an observation; and model state with the rear thrust limit)
S_LAST_FSAT = 43        # last tick's front / rear tire saturation (observation only)
S_LAST_RSAT = 44
S_SMOOTH_H = 45         # the plugin's _smoothedHeight; NaN = none yet
S_REAR_SLIDE_GRIP = 46  # the plugin's RearSlidingGrip for this car: share of its peak sideways grip
                        # the rear axle keeps once it is sliding (1 = all of it). Per car, like
                        # S_MAX_SPEED, so training can vary it per episode.
S_LOAD_SHIFT = 48       # load moved onto the front axle by braking (- = onto the rear by thrust), N; EnableWeightTransfer
S_PACE = 47             # the pace this car is asked to drive, a fraction of the course's reference speed (0 = none): the
                        # plugin's bot-spacing dial, drawn per episode in training (drift_env)
STATE_SIZE = 49

# Bumped whenever the simulator changes in a way that makes policies trained before it unfit for
# the server. Trainers write it into policy.json ("simulator_revision"); check_plugin_policy.py
# warns about a policy that carries an older one (or none).
#   1  (implicit, no tag) - car height = spline height. WRONG away from the line: the plugin blends
#      in the road-mesh height, and the fan rays are horizontal, so a few cm decides whether a ray
#      hits the ground. runs/long1 was trained here and failed on the server.
#   2  EXT_HEIGHT_MODEL (the plugin's height blend, from the server's own
#      road-mesh file). One-tick replay of that live run: fan rays wrong on 27.4% of ticks -> 2.3%.
#      Later, EXT_GROUND_RAY_HEIGHT added WITHOUT bumping this: it follows a change to the
#      plugin (not a simulator error), and a revision-2 policy measured the same under it
#      (fix2_from_long1, 256 no-reset runs: drift 45.0% -> 44.0%, runs with an excursion 2.7% -> 0%).
#      Later still, the front friction circle under the foot brake too (it had only been applied
#      with the brake action), plugin and simulator together, WITHOUT bumping this: gen5's
#      direction changes measured the same (Drift Playground 18.4 -> 19.1 a minute at pace 0.9).
#      Later again (2026-10-18), the contact pins see only walls (a road triangle no longer hides a
#      wall behind it), plugin and simulator together, WITHOUT bumping this: at the pins' old height
#      (0.55 m above the road) the road is never within their 1.6 m; it is what lets a policy lower
#      them (ProximityPinHeightOffsetMeters) to feel a low wall.
SIMULATOR_REVISION = 2

# "mid" vector: what step_pre() works out (where the car is on the spline and how far off it)
# and step_post() then consumes. The split exists so a policy can be evaluated between the two,
# where the C# hook sits.
M_NEAREST, M_A, M_B, M_T, M_RADIUS, M_CORR_ANGLE, M_GLITCH, M_CTE, M_FIRST, M_NP_GLITCH, M_DT = range(11)
MID_SIZE = 11
N_RAYS = 15
N_PINS = 6
# How far the reward's wall-proximity signal (D_MinWallDistanceM) looks for the game's walls.
# Reward only - nothing the plugin does. RewardConfig.wall_proximity_distance_m must not exceed it.
WALL_PROXIMITY_RANGE_M = 3.0

# ----------------------------------------------------------------------------------------
# Parameter vector layout
# ----------------------------------------------------------------------------------------
_PNAMES = [
    # the car (the Miata's numbers; car_numbers.py reads them for other cars)
    'YAW_INERTIA_KGM2', 'DIST_CG_FRONT_M', 'DIST_CG_REAR_M', 'FRONT_STATIC_LOAD_N',
    'REAR_STATIC_LOAD_N', 'LAT_FRONT_PEAK_FRICTION', 'LAT_REAR_PEAK_FRICTION',
    'FRONT_PEAK_SLIP_RAD', 'REAR_PEAK_SLIP_RAD', 'MAX_STEER_RAD',
    # speed: the target-speed ramp, and the brakes the pedal and the decel cap use
    'AI_ACCELERATION_MS2', 'AI_DECELERATION_MS2', 'MAX_LONGITUDINAL_DECEL_MS2', 'ENABLE_LONGITUDINAL_DECEL_CAP',
    # lateral dynamics
    'MAX_SUBSTEP_S', 'MIN_SLIP_SPEED_MS', 'ENABLE_FRICTION_ELLIPSE',
    # the angle-glitch correction: the largest plausible change of spline direction per tick
    'MAX_LATERAL_ACCEL_MS2',
    # sensing
    'RAYCAST_FAN_ANGLE_DEGREES', 'RAYCAST_FAN_MAX_DISTANCE_M', 'RAYCAST_HEIGHT_OFFSET_M',
    'PROXIMITY_PIN_MAX_DISTANCE_M', 'PROXIMITY_PIN_HEIGHT_OFFSET_M',
    # plugin mechanisms (GradientDriftConfiguration.cs defaults)
    'EXT_NEAREST_POINT_CORRECTOR', 'EXT_COLLISION_RESPONSE',
    'COLLISION_CONTACT_DISTANCE_M', 'COLLISION_PUSHOUT_MARGIN_M',
    'EXT_HEIGHT_MODEL', 'HEIGHT_SMOOTHING_FACTOR', 'HEIGHT_SNAP_THRESHOLD_M', 'ROAD_MESH_HEIGHT_OFFSET_M',
    'CROSS_TRACK_BLEND_START_M', 'CROSS_TRACK_BLEND_END_M', 'MAX_TRUSTED_MESH_DISTANCE_M',
    'EXT_GROUND_RAY_HEIGHT', 'GROUND_RAY_START_ABOVE_M', 'GROUND_RAY_MAX_DISTANCE_M',
    'GROUND_RAY_HEIGHT_OFFSET_M', 'GROUND_RAY_SMOOTHING_FACTOR',
    'EXT_WRONG_WAY_RECOVERY', 'WRONG_WAY_ENTRY_DEG', 'WRONG_WAY_ENTRY_TICKS', 'WRONG_WAY_EXIT_SLIP_DEG',
    'WRONG_WAY_EXIT_YAW_DEGS', 'SPIN_RECOVERY_TIME_CONSTANT_S',
    # the plugin settings a policy file carries (off unless it sets them) - see step_post
    'EXT_REAR_THRUST_LIMIT', 'MAX_STEER_RATE_RAD_S', 'EXT_REAR_BRAKE_LIMIT', 'EXT_REAR_FRICTION_CIRCLE',
    'EXT_WEIGHT_TRANSFER', 'CG_HEIGHT_M', 'LOAD_TRANSFER_TIME_S',
    'EXT_TIRE_LOAD_SENSITIVITY', 'FRONT_LOAD_SENSITIVITY_PER_KN', 'REAR_LOAD_SENSITIVITY_PER_KN',
    'FRONT_TRACK_M', 'REAR_TRACK_M', 'ROLL_STIFFNESS_FRONT_SHARE',
    # the newer tyre model (tyres.ini VERSION 5 and up): grip at a tire's load F is
    # peak x (F / FZ0)^(LS_EXPY - 1), the peak (LAT_*_PEAK_FRICTION) being DY_REF. 0 = the older
    # model (DY0, DY1 per kN), as before.
    'FRONT_LS_EXPY', 'REAR_LS_EXPY', 'FRONT_FZ0_N', 'REAR_FZ0_N',
    # the engine (EnableEngine): the most force the drivetrain can put on the road, N, at
    # ENGINE_CURVE_STEP_MS steps of forward speed from 0 (cars/car_numbers.py drive_force_curve)
    'EXT_ENGINE',
    # this car's flat-out speed over the reference car's (a pace asks for pace x the course's
    # reference speed x this); 0 = 1
    'PACE_SCALE',
    # how good a driver the policy is asked to be, 0 - 1 (drift_env SKILL_*); 1 = as trained before
    'SKILL',
] + [f'ENGINE_FORCE_{k:02d}' for k in range(33)] + [
    # DisableBrakes (a plugin setting a policy file may carry): with EnableRearFrictionCircle the
    # pedal cannot go below zero and the brake action does nothing, so a car sheds speed only by
    # lifting and with its angle, the way the owner drives Drift Playground (step_post)
    'EXT_NO_BRAKES',
    # FrontBrakeOnly (the same kind of setting): the pedal cannot go below zero, so no braking on
    # all four wheels, but the brake action - the front axle, with the throttle - still works, so a
    # car can scrub speed and keep the rear loose (step_post)
    'EXT_FRONT_BRAKE_ONLY',
    # CarOutlineContact (the same kind of setting): walls are found along the car's own outline,
    # a rectangle turned with its heading (the axles' distances from the centre plus an overhang,
    # half the wider track plus a side margin), instead of a circle of COLLISION_CONTACT_DISTANCE_M
    # around its centre: a sliding car's tail reaches a wall long before its centre does
    # (Spirit Peaks' tire stacks, 2026-10-21). The reward's nearest wall reads the same outline.
    'EXT_CAR_OUTLINE_CONTACT', 'CAR_OUTLINE_OVERHANG_M', 'CAR_OUTLINE_SIDE_MARGIN_M',
    # RoundTireCurve (the same kind of setting): each axle's sideways force follows the brush
    # (Fiala) curve, steep at first and rounding over into full sliding, instead of a straight line
    # to its peak. Full sliding comes at the tyre's FRICTION_LIMIT_ANGLE plus FLEX x its static
    # load (tyres.ini: FLEX, "the bigger the added slip angle with load", degrees per N).
    # Assetto Corsa's E36 front gives 75 - 90% of its peak by 4 - 8 deg (the owner's laps).
    'EXT_ROUND_TIRE_CURVE', 'FRONT_TIRE_FLEX', 'REAR_TIRE_FLEX',
]
P = {name: i for i, name in enumerate(_PNAMES)}

# Whatever the server's extra_cfg.yml (or a newer plugin build) sets differently from the values
# make_params() fills in. Applied on top of them, so every environment, evaluation and replay in
# this process sees the same plugin. Keys are _PNAMES.
PLUGIN_CONFIG_OVERRIDES = {}
PARAM_SIZE = len(_PNAMES)
ENGINE_CURVE_STEP_MS = 2.5
ENGINE_CURVE_POINTS = 33
for _n, _i in P.items():
    globals()['P_' + _n] = _i

# ----------------------------------------------------------------------------------------
# Diagnostics vector layout (per tick outputs)
# ----------------------------------------------------------------------------------------
_DNAMES = [
    'SpeedKmh', 'BodyForwardSpeedMs', 'RawSplineAngle', 'GlitchCorrected', 'CrossTrackErrorM',
    'SteerCommandDeg', 'FrontSlipAngleDeg', 'RearSlipAngleDeg', 'FrontSaturation', 'RearSaturation',
    'SteerSaturated', 'NearestPointId', 'Radius', 'RealizedLongitudinalAccelMs2', 'DesiredSpeedMs',
    # plugin mechanisms and reward signals
    'NearestPointGlitchCorrected', 'UncontrolledSpinRecovering', 'CollisionPushOutM', 'ProgressM',
    'SurfaceUnderCar', 'MinWallDistanceM', 'Vx', 'Vz', 'RealSlipAngleRad',
]
D = {name: i for i, name in enumerate(_DNAMES)}
DIAG_SIZE = len(_DNAMES)
for _n, _i in D.items():
    globals()['D_' + _n] = _i

SURF_ROAD, SURF_WALL, SURF_GRASS, SURF_UNKNOWN, SURF_NONE = 0, 1, 2, 3, 4
SURFACE_CODES = {'Road': SURF_ROAD, 'Wall': SURF_WALL, 'Grass': SURF_GRASS,
                 'Unknown': SURF_UNKNOWN, 'None': SURF_NONE}

PIN_OFFSETS = np.array([-135.0 * math.pi / 180.0, -90.0 * math.pi / 180.0, -45.0 * math.pi / 180.0,
                        45.0 * math.pi / 180.0, 90.0 * math.pi / 180.0, 135.0 * math.pi / 180.0])

TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _pin_offset(i):
    """PIN_OFFSETS[i] as a function (same expressions, so bit-identical values): compiled code
    reads it this way so it needs no global array, which keeps it valid as GPU device code."""
    if i == 0:
        return -135.0 * math.pi / 180.0
    if i == 1:
        return -90.0 * math.pi / 180.0
    if i == 2:
        return -45.0 * math.pi / 180.0
    if i == 3:
        return 45.0 * math.pi / 180.0
    if i == 4:
        return 90.0 * math.pi / 180.0
    return 135.0 * math.pi / 180.0


def make_params():
    """The flat parameter vector: the Miata, and the plugin's compiled defaults
    (GradientDriftConfiguration.cs), with PLUGIN_CONFIG_OVERRIDES on top."""
    values = {
        'YAW_INERTIA_KGM2': 2199.6, 'DIST_CG_FRONT_M': 1.035, 'DIST_CG_REAR_M': 1.265,
        'FRONT_STATIC_LOAD_N': 6132.6, 'REAR_STATIC_LOAD_N': 5017.6,
        'LAT_FRONT_PEAK_FRICTION': 1.3745, 'LAT_REAR_PEAK_FRICTION': 1.3385,
        'FRONT_PEAK_SLIP_RAD': math.radians(11.7), 'REAR_PEAK_SLIP_RAD': math.radians(12.0),
        'MAX_STEER_RAD': math.radians(28.125),
        'AI_ACCELERATION_MS2': 2.5, 'AI_DECELERATION_MS2': 8.5, 'MAX_LONGITUDINAL_DECEL_MS2': 5.6,
        'ENABLE_LONGITUDINAL_DECEL_CAP': 1.0,
        'MAX_SUBSTEP_S': 0.05, 'MIN_SLIP_SPEED_MS': 3.0, 'ENABLE_FRICTION_ELLIPSE': 1.0,
        'MAX_LATERAL_ACCEL_MS2': 13.3,
        'RAYCAST_FAN_ANGLE_DEGREES': 120.0, 'RAYCAST_FAN_MAX_DISTANCE_M': 60.0, 'RAYCAST_HEIGHT_OFFSET_M': 0.5,
        'PROXIMITY_PIN_MAX_DISTANCE_M': 12.0, 'PROXIMITY_PIN_HEIGHT_OFFSET_M': 0.15,
        'EXT_NEAREST_POINT_CORRECTOR': 1.0, 'EXT_COLLISION_RESPONSE': 1.0,
        'COLLISION_CONTACT_DISTANCE_M': 1.6, 'COLLISION_PUSHOUT_MARGIN_M': 0.3,
        'CAR_OUTLINE_OVERHANG_M': 0.8, 'CAR_OUTLINE_SIDE_MARGIN_M': 0.1,
        'EXT_HEIGHT_MODEL': 1.0, 'HEIGHT_SMOOTHING_FACTOR': 0.15, 'HEIGHT_SNAP_THRESHOLD_M': 0.15,
        'ROAD_MESH_HEIGHT_OFFSET_M': 0.44, 'CROSS_TRACK_BLEND_START_M': 5.0, 'CROSS_TRACK_BLEND_END_M': 12.0,
        'MAX_TRUSTED_MESH_DISTANCE_M': 8.0,
        'EXT_GROUND_RAY_HEIGHT': 1.0, 'GROUND_RAY_START_ABOVE_M': 1.5, 'GROUND_RAY_MAX_DISTANCE_M': 6.0,
        'GROUND_RAY_HEIGHT_OFFSET_M': 0.40, 'GROUND_RAY_SMOOTHING_FACTOR': 1.0,
        'EXT_WRONG_WAY_RECOVERY': 1.0, 'WRONG_WAY_ENTRY_DEG': 67.0, 'WRONG_WAY_ENTRY_TICKS': 3.0,
        'WRONG_WAY_EXIT_SLIP_DEG': 30.0, 'WRONG_WAY_EXIT_YAW_DEGS': 20.0, 'SPIN_RECOVERY_TIME_CONSTANT_S': 0.3,
        # weight transfer (EnableWeightTransfer) and tire load sensitivity: the Miata's centre of
        # gravity (suspensions.ini BASEY, 0.14 m above the wheel centres, + the 0.26 m tyre
        # radius; was a guessed 0.50 at first), and how long the suspension takes to settle
        'CG_HEIGHT_M': 0.40, 'LOAD_TRANSFER_TIME_S': 0.15,
        # tire load sensitivity (EnableTireLoadSensitivity, off unless a policy file sets it): the
        # Miata's tyres.ini DY1 (grip lost per kN on a tire), suspensions.ini TRACK, and the share
        # of the roll stiffness at the front from its springs and anti-roll bars
        'FRONT_LOAD_SENSITIVITY_PER_KN': -0.044, 'REAR_LOAD_SENSITIVITY_PER_KN': -0.049,
        'FRONT_TRACK_M': 1.67, 'REAR_TRACK_M': 1.50, 'ROLL_STIFFNESS_FRONT_SHARE': 0.67,
        'SKILL': 1.0,
        # tyres.ini FLEX (RoundTireCurve only), degrees of slip angle per N on a tire
        'FRONT_TIRE_FLEX': 0.000852, 'REAR_TIRE_FLEX': 0.000852,
    }
    p = np.zeros(PARAM_SIZE)
    for k, v in values.items():
        p[P[k]] = float(v)
    for k, v in PLUGIN_CONFIG_OVERRIDES.items():
        p[P[k]] = float(v)
    return p


# ----------------------------------------------------------------------------------------
# World: spline arrays + triangle grid in CSR form
# ----------------------------------------------------------------------------------------
class World:
    """Plain container of numpy arrays (numba functions take the arrays, not this object)."""

    def __init__(self, spline_points, tri_v, tri_surf, collision_v, collision_surf, target_per_cell=8, max_cell_span=64,
                 fine_grid=True, fine_cell_size=1.0, fine_margin_m=100.0, road_mesh=None, rm_cell=2.0,
                 course_edges=None, course_markers=None, course_zones=None):
        n = len(spline_points)
        self.sp_pos = np.array([p.position for p in spline_points], dtype=np.float64)
        self.sp_radius = np.array([p.radius for p in spline_points], dtype=np.float64)
        self.sp_length = np.array([p.length for p in spline_points], dtype=np.float64)
        self.sp_next = np.array([p.next_id for p in spline_points], dtype=np.int64)
        self.sp_prev = np.array([p.previous_id for p in spline_points], dtype=np.int64)
        fwd = np.zeros((n, 3))
        for i in range(n):
            if self.sp_next[i] >= 0:
                fwd[i] = self.sp_pos[self.sp_next[i]] - self.sp_pos[i]
        self.sp_fwd = fwd
        cum = np.zeros(n)
        cum[1:] = np.cumsum(self.sp_length[:-1])
        self.sp_cum = cum
        self.lap_length = float(self.sp_length.sum())

        # Road-mesh points (<track>_road_mesh.csv - the same file the plugin builds its
        # MeshKdTree from), bucketed into 2 m cells for the nearest-point lookup. Absent file =
        # placeholder arrays with zero dims: the lookup then finds nothing and height falls back
        # to the spline, exactly as the plugin does when _roadMeshTree is null.
        self.rm_cell = rm_cell
        if road_mesh is not None and len(road_mesh) > 0:
            self.rm_pos = np.ascontiguousarray(road_mesh, dtype=np.float64)
            self.rm_min = self.rm_pos.min(axis=0) - 0.01
            ext = self.rm_pos.max(axis=0) + 0.01 - self.rm_min
            self.rm_dims = np.array([int(ext[a] / self.rm_cell) + 1 for a in range(3)], dtype=np.int64)
            self.rm_start, self.rm_items = _build_point_grid(self.rm_pos, self.rm_min, self.rm_cell, self.rm_dims)
        else:
            self.rm_pos = np.zeros((1, 3))
            self.rm_min = np.zeros(3)
            self.rm_dims = np.zeros(3, dtype=np.int64)
            self.rm_start = np.zeros(2, dtype=np.int64)
            self.rm_items = np.zeros(1, dtype=np.int64)
        self.has_road_mesh = bool(self.rm_dims[0] > 0)

        # The course layout (build_course.py): per spline point, how far the course extends to
        # the positive and negative cross-track side of the line and where those two edges are
        # [pos_m, neg_m, pos_x, pos_z, neg_x, neg_z, on_long_straight, zone_side, held_side]; and the markers [x, z, radius]. Absent =
        # one-row placeholders, which every consumer reads as "no course known" (co has to have
        # one row per spline point to count).
        # A 7th column marks the spline points on a LONG straight (recorded radius above 150 m for
        # at least 30 m on end) - only used for drift_env's "what does it do on the straights"
        # statistics. On Spirit Peaks, for example, that is the 92 m top straight, points 488-547.
        # An 8th says which edge has an outside zone at that point (+1 positive side, -1 negative,
        # 0 none; build_zones.py), a 9th the same for the stretch of a long corner the tail has to
        # stay out the whole way (build_zones.held_side_per_point), a 10th which way the line turns
        # over the next TURN_WINDOW_M (+1 / -1 as the slip angle's sign in a drift through it; 0 =
        # less than TURN_MIN_DEG: straight; drift_env's weave penalty), an 11th the line's signed
        # curvature (signed_curvature: + = turning toward + cross-track; drift_env path_progress).
        self.course_rows = 0 if course_edges is None else len(course_edges)
        if course_edges is not None and len(course_edges) == n:
            on_straight = np.zeros(n)
            wide = self.sp_radius > 150.0
            i = 0
            while i < n:
                if wide[i]:
                    j = i
                    while j + 1 < n and wide[j + 1]:
                        j += 1
                    if self.sp_length[i:j + 1].sum() >= 30.0:
                        on_straight[i:j + 1] = 1.0
                    i = j + 1
                else:
                    i += 1
            zone_side = np.zeros(n)
            for side, first, last in (course_zones if course_zones is not None else ()):
                zone_side[np.arange(int(first), int(first) + (int(last) - int(first)) % n + 1) % n] = side
            edges = np.asarray(course_edges, dtype=np.float64)[:, :6]
            from build_zones import held_side_per_point
            held_side = held_side_per_point(edges[:, 2:4], edges[:, 4:6])
            self.co = np.ascontiguousarray(np.c_[edges, on_straight, zone_side, held_side, self.turn_side(),
                                                    self.signed_curvature()])
        else:
            self.co = np.zeros((1, 11))
        if course_markers is not None and len(course_markers) > 0:
            self.mk = np.ascontiguousarray(course_markers, dtype=np.float64)
        else:
            self.mk = np.zeros((1, 3))
        self.has_course = bool(self.co.shape[0] == n)

        # One triangle list, the sensors' triangles first and the collision triangles (what the
        # game collides with: ground and walls) after them, in one grid whose cells hold both:
        # cell c's sensor triangles are cell_items[cell_start[2c] : cell_start[2c+1]], its
        # collision triangles cell_items[cell_start[2c+1] : cell_start[2c+2]] (raycast's last
        # argument picks which).
        self.n_sensor = len(tri_v)
        self.tri_v = np.ascontiguousarray(np.concatenate([np.asarray(tri_v, dtype=np.float64).reshape(-1, 3, 3),
                                                          np.asarray(collision_v, dtype=np.float64).reshape(-1, 3, 3)]))
        self.tri_surf = np.ascontiguousarray(np.concatenate([tri_surf, collision_surf]), dtype=np.int8)
        if fine_grid:
            # TRAINING GRID. The plugin's grids (fine_grid=False, roughly: the plugin sizes its
            # sensor and collision grids separately) size their cells from the whole mesh's
            # bounding volume, and Spirit Peaks' mesh includes ~2.6 km of scenery - so its
            # cells come out ~37 m wide and every ray tests thousands of triangles. A ray's
            # nearest hit does not depend on how the triangles are bucketed, so this grid just
            # buckets the triangles near the track into 1 m cells instead: same answers from
            # a small fraction of the triangle tests. Only valid while
            # the car stays within fine_margin_m - 60 m (the fan range) of the spline's bounding
            # box, which the training environment's own off-track termination guarantees.
            lo = self.sp_pos.min(axis=0) - fine_margin_m
            hi = self.sp_pos.max(axis=0) + fine_margin_m
            flat = self.tri_v.reshape(-1, 3)
            near = ((flat[:, 0] > lo[0]) & (flat[:, 0] < hi[0]) & (flat[:, 2] > lo[2]) & (flat[:, 2] < hi[2]))
            lo[1] = flat[near, 1].min() - 0.01
            hi[1] = flat[near, 1].max() + 0.01
            (self.grid_min, self.cell_size, self.dims, self.cell_start, self.cell_items,
             self.skipped) = _build_grid(self.tri_v, target_per_cell, 1 << 30, bounds=(lo, hi),
                                         cell_size=fine_cell_size)
        else:
            (self.grid_min, self.cell_size, self.dims, self.cell_start,
             self.cell_items, self.skipped) = _build_grid(self.tri_v, target_per_cell, max_cell_span)
        self.cell_start = _split_cells(self.cell_start, self.cell_items, self.n_sensor)

    def turn_side(self, window_m=15.0, min_deg=12.0):
        """Per spline point: +1 / -1 for the way the line turns over the next window_m, 0 where it
        turns less than min_deg (drift_env weave penalty, co[:, 9])."""
        n = len(self.sp_pos)
        heading = np.arctan2(self.sp_fwd[:, 2], self.sp_fwd[:, 0])
        out = np.zeros(n)
        for i in range(n):
            cur, acc = i, 0.0
            while acc < window_m and self.sp_next[cur] >= 0:
                acc += self.sp_length[cur]
                cur = self.sp_next[cur]
            turned = math.degrees(math.atan2(math.sin(heading[cur] - heading[i]), math.cos(heading[cur] - heading[i])))
            out[i] = 0.0 if abs(turned) < min_deg else math.copysign(1.0, turned)
        return out

    def signed_curvature(self, window_m=15.0):
        """Per spline point: the line's heading change per metre over window_m centred on it (round
        the loop), + where it turns toward + cross-track (the side M_CTE counts as +). A path
        n metres to that side of the line is (1 - k n) metres long per metre of the line."""
        xz = self.sp_pos[:, [0, 2]]
        step = np.roll(xz, -1, axis=0) - xz
        heading = np.arctan2(step[:, 1], step[:, 0])
        turn = np.diff(np.r_[heading, heading[0]])
        turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
        lengths = np.asarray(self.sp_length, dtype=np.float64)
        arc = np.r_[0.0, np.cumsum(lengths[:-1])]
        lap = float(lengths.sum())
        cum = np.r_[0.0, np.cumsum(turn)]
        knots = np.r_[arc, lap]

        def heading_at(a):
            laps = np.floor(a / lap)
            return np.interp(a - laps * lap, knots, cum) + laps * cum[-1]
        return (heading_at(arc + 0.5 * window_m) - heading_at(arc - 0.5 * window_m)) / window_m

    def sensor_triangles(self):
        """(vertices, surface codes) of the sensors' triangles."""
        return self.tri_v[:self.n_sensor], self.tri_surf[:self.n_sensor]

    def arrays(self):
        """Everything step()/step_batch() take after their own arguments, in order."""
        return (self.sp_pos, self.sp_radius, self.sp_length, self.sp_next, self.sp_prev, self.sp_fwd,
                self.sp_cum, self.tri_v, self.tri_surf, self.grid_min, self.cell_size, self.dims,
                self.cell_start, self.cell_items,
                self.rm_pos, self.rm_min, self.rm_cell, self.rm_dims, self.rm_start, self.rm_items,
                self.co, self.mk)

    def ray_arrays(self):
        """Just the triangle grid, as raycast() takes it."""
        return (self.tri_v, self.tri_surf, self.grid_min, self.cell_size, self.dims, self.cell_start,
                self.cell_items)


def load_world(tracks_dir, fine_grid=True, track=None, fine_margin_m=100.0, fine_cell_size=1.0, rm_cell=2.0,
               sensors='triangles'):
    """Loads a track's AI line, its two triangle files (the sensors' and the collision one) and what
    else of its data there is, from its folder in tracks_dir (track_files; track_files.TRACK when
    track is not given). sensors='visual' gives the sensors what the track's model shows instead
    (<track>_visual.csv.gz, for finding its markers)."""
    import os
    import pandas as pd
    from ai_spline import load_fast_lane
    import track_files as tf
    data = tracks_dir
    line = tf.path(data, 'fast_lane', track or tf.TRACK)
    if not os.path.exists(line):
        sys.exit(f"ERROR: {line} not found: copy the track's ai/fast_lane.ai there under that name")
    points = load_fast_lane(line)
    tris = []
    for kind in (sensors, 'collision'):
        path = tf.path(data, kind, track or tf.TRACK)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found. A track's data comes from its own files in the game: "
                     "python setup_track.py <game folder>/content/tracks/<track>")
        df = pd.read_csv(path)
        tris.append((df.iloc[:, :9].to_numpy(dtype=np.float64).reshape(-1, 3, 3),
                     df.iloc[:, 9].map(SURFACE_CODES).to_numpy(dtype=np.int8)))
    (v, surf), (collision_v, collision_surf) = tris
    road_mesh = None
    road_mesh_path = tf.path(data, 'road_mesh', track or tf.TRACK)
    if os.path.exists(road_mesh_path):
        road_mesh = pd.read_csv(road_mesh_path).iloc[:, :3].to_numpy(dtype=np.float64)
    else:
        print(f"WARNING: {road_mesh_path} not found - the simulator's car height will not match the "
              "plugin's away from the racing line (copy the file from the server folder).", file=sys.stderr)
    course_edges = course_markers = None
    edges_path = tf.path(data, 'course_edges', track or tf.TRACK)
    markers_path = tf.path(data, 'course_markers', track or tf.TRACK)
    if os.path.exists(edges_path) and os.path.exists(markers_path):
        course_edges = pd.read_csv(edges_path).iloc[:, 1:7].to_numpy(dtype=np.float64)
        course_markers = pd.read_csv(markers_path)[['X', 'Z', 'RadiusM']].to_numpy(dtype=np.float64)
    zones_path = tf.path(data, 'course_zones', track or tf.TRACK)
    course_zones = None
    if os.path.exists(zones_path):
        course_zones = pd.read_csv(zones_path)[['Side', 'FirstPointId', 'LastPointId']].to_numpy(dtype=np.float64)
    return World(points, v, surf, collision_v, collision_surf, fine_grid=fine_grid, fine_margin_m=fine_margin_m,
                 fine_cell_size=fine_cell_size, rm_cell=rm_cell, road_mesh=road_mesh,
                 course_edges=course_edges, course_markers=course_markers, course_zones=course_zones)


def require_course(world, tracks_dir, track=None):
    """Trainers call this: without the course layout "on the track" silently means "on the paved
    surface" again, which on this track is up to 15 m wider than the course on each side."""
    if not world.has_course:
        import os
        import track_files as tf
        if world.course_rows:
            sys.exit(f"\nERROR: the course file has {world.course_rows} points and the AI line {len(world.sp_pos)}: "
                     "they belong to different tracks (each track's files carry its name, fast_lane.ai too).")
        sys.exit("\nERROR: the course layout files are missing from "
                 + tf.folder(os.path.abspath(tracks_dir), track or tf.TRACK)
                 + f"\n({tf.name('course_edges', track or tf.TRACK)}, {tf.name('course_markers', track or tf.TRACK)})."
                 "\nRun  python build_course.py  to create them from the track mesh.")


def require_road_mesh(world, tracks_dir, track=None):
    """Trainers call this: training without the road-mesh file reproduces the first live run's failure
    (a policy that looks excellent here and leaves the track on the server), so it is refused."""
    if not world.has_road_mesh:
        import os
        import track_files as tf
        sys.exit("\nERROR: " + tf.path(os.path.abspath(tracks_dir), 'road_mesh', track or tf.TRACK)
                 + " is missing.\nIt is the server's own file (in the AssettoServer folder); "
                 "copy it there.\nWithout it the simulated car sits at the wrong height off the racing line, "
                 "its wall sensors read differently\nfrom the server's, and the trained policy will not transfer.")


@njit(cache=True)
def _cell_index(p, gmin, cell_size, dims, out):
    for a in range(3):
        c = int((p[a] - gmin[a]) / cell_size)
        if c < 0:
            c = 0
        if c > dims[a] - 1:
            c = dims[a] - 1
        out[a] = c


@njit(cache=True)
def _build_grid_csr(tri_v, gmin, gmax, bounded, cell_size, dims, max_cell_span):
    n = tri_v.shape[0]
    n_cells = dims[0] * dims[1] * dims[2]
    counts = np.zeros(n_cells + 1, dtype=np.int64)
    c0 = np.zeros(3, dtype=np.int64)
    c1 = np.zeros(3, dtype=np.int64)
    tmin = np.zeros(3)
    tmax = np.zeros(3)
    skipped = 0
    keep = np.ones(n, dtype=np.bool_)
    for idx in range(n):
        for a in range(3):
            tmin[a] = min(tri_v[idx, 0, a], tri_v[idx, 1, a], tri_v[idx, 2, a])
            tmax[a] = max(tri_v[idx, 0, a], tri_v[idx, 1, a], tri_v[idx, 2, a])
        if bounded:
            outside = False
            for a in range(3):
                if tmax[a] < gmin[a] or tmin[a] > gmax[a]:
                    outside = True
            if outside:
                keep[idx] = False
                continue
        _cell_index(tmin, gmin, cell_size, dims, c0)
        _cell_index(tmax, gmin, cell_size, dims, c1)
        span = max(c1[0] - c0[0] + 1, c1[1] - c0[1] + 1, c1[2] - c0[2] + 1)
        if span > max_cell_span:
            skipped += 1
            keep[idx] = False
            continue
        for ix in range(c0[0], c1[0] + 1):
            for iy in range(c0[1], c1[1] + 1):
                for iz in range(c0[2], c1[2] + 1):
                    counts[(ix * dims[1] + iy) * dims[2] + iz + 1] += 1
    cell_start = np.cumsum(counts)
    items = np.zeros(cell_start[-1], dtype=np.int64)
    fill = cell_start[:-1].copy()
    for idx in range(n):
        if not keep[idx]:
            continue
        for a in range(3):
            tmin[a] = min(tri_v[idx, 0, a], tri_v[idx, 1, a], tri_v[idx, 2, a])
            tmax[a] = max(tri_v[idx, 0, a], tri_v[idx, 1, a], tri_v[idx, 2, a])
        _cell_index(tmin, gmin, cell_size, dims, c0)
        _cell_index(tmax, gmin, cell_size, dims, c1)
        for ix in range(c0[0], c1[0] + 1):
            for iy in range(c0[1], c1[1] + 1):
                for iz in range(c0[2], c1[2] + 1):
                    c = (ix * dims[1] + iy) * dims[2] + iz
                    items[fill[c]] = idx
                    fill[c] += 1
    return cell_start, items, skipped


@njit(cache=True)
def _split_cells(cell_start, items, n_sensor):
    """cell_start with each cell split in two: its sensor triangles (index below n_sensor), then
    its collision triangles. Within a cell the items are in ascending index order, so the sensors'
    come first already."""
    n_cells = cell_start.shape[0] - 1
    out = np.zeros(2 * n_cells + 1, dtype=np.int64)
    for c in range(n_cells):
        k = cell_start[c]
        while k < cell_start[c + 1] and items[k] < n_sensor:
            k += 1
        out[2 * c] = cell_start[c]
        out[2 * c + 1] = k
    out[2 * n_cells] = cell_start[n_cells]
    return out


def _build_grid(tri_v, target_per_cell, max_cell_span, bounds=None, cell_size=None):
    """bounds=None: the plugin's sizing rule (TriangleGrid.cs), applied to the given triangles
    (the plugin sizes its sensor and collision grids each from its own set); per-cell triangle order
    ascending index, as there. bounds=(lo, hi) + cell_size: the
    training grid, see World.__init__."""
    if bounds is None:
        flat = tri_v.reshape(-1, 3)
        gmin = flat.min(axis=0) - 0.01
        gmax = flat.max(axis=0) + 0.01
    else:
        gmin = np.array(bounds[0], dtype=np.float64)
        gmax = np.array(bounds[1], dtype=np.float64)
    extent = gmax - gmin
    if cell_size is None:
        volume = extent[0] * extent[1] * extent[2]
        n = tri_v.shape[0]
        cell_volume = volume * target_per_cell / max(n, 1)
        cell_size = max(cell_volume ** (1.0 / 3.0), 0.5)
    dims = np.array([max(1, int(extent[a] / cell_size) + 1) for a in range(3)], dtype=np.int64)
    cell_start, items, skipped = _build_grid_csr(tri_v, gmin, gmax, bounds is not None, cell_size,
                                                 dims, max_cell_span)
    return gmin, float(cell_size), dims, cell_start, items, int(skipped)


@njit(cache=True)
def _build_point_grid(pos, gmin, cell, dims):
    n_cells = dims[0] * dims[1] * dims[2]
    counts = np.zeros(n_cells + 1, dtype=np.int64)
    idx = np.zeros(pos.shape[0], dtype=np.int64)
    for i in range(pos.shape[0]):
        cx = min(dims[0] - 1, max(0, int((pos[i, 0] - gmin[0]) / cell)))
        cy = min(dims[1] - 1, max(0, int((pos[i, 1] - gmin[1]) / cell)))
        cz = min(dims[2] - 1, max(0, int((pos[i, 2] - gmin[2]) / cell)))
        idx[i] = (cx * dims[1] + cy) * dims[2] + cz
        counts[idx[i] + 1] += 1
    start = np.cumsum(counts)
    items = np.zeros(pos.shape[0], dtype=np.int64)
    fill = start[:-1].copy()
    for i in range(pos.shape[0]):
        items[fill[idx[i]]] = i
        fill[idx[i]] += 1
    return start, items


@njit(cache=True)
def nearest_road_mesh_point(x, y, z, max_dist, rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items):
    """MeshKdTree.FindNearest, restricted to what the plugin actually uses it for: the nearest
    road-mesh point if it lies within max_dist (MaxTrustedMeshDistanceM). Returns (distance, Y of
    that point); distance < 0 means nothing within range (or no road mesh loaded)."""
    if rm_dims[0] <= 0:
        return -1.0, 0.0
    reach = int(max_dist / rm_cell) + 1
    cx = int(math.floor((x - rm_min[0]) / rm_cell))
    cy = int(math.floor((y - rm_min[1]) / rm_cell))
    cz = int(math.floor((z - rm_min[2]) / rm_cell))
    best = max_dist * max_dist
    best_y = 0.0
    found = False
    for ix in range(max(0, cx - reach), min(rm_dims[0] - 1, cx + reach) + 1):
        for iy in range(max(0, cy - reach), min(rm_dims[1] - 1, cy + reach) + 1):
            for iz in range(max(0, cz - reach), min(rm_dims[2] - 1, cz + reach) + 1):
                c = (ix * rm_dims[1] + iy) * rm_dims[2] + iz
                for k in range(rm_start[c], rm_start[c + 1]):
                    i = rm_items[k]
                    d = (rm_pos[i, 0] - x) ** 2 + (rm_pos[i, 1] - y) ** 2 + (rm_pos[i, 2] - z) ** 2
                    if d < best:
                        best = d
                        best_y = rm_pos[i, 1]
                        found = True
    if not found:
        return -1.0, 0.0
    return math.sqrt(best), best_y


# ----------------------------------------------------------------------------------------
# Raycasting (TriangleGrid.cs: a grid walk + Moller-Trumbore)
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def ground_below(x, oy, z, max_distance, tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items):
    """(distance down from oy, surface code) of the first collision triangle below (x, oy, z) that
    is not a wall; walls are stepped through, 12 casts at most (the plugin's TryFindGround,
    GroundRayMaxWallLayers). Distance < 0 = no ground within max_distance."""
    ray_y = oy
    left = max_distance
    for _ in range(12):
        d, surf = raycast(x, ray_y, z, 0.0, -1.0, 0.0, left, tri_v, tri_surf, gmin, cell_size, dims,
                          cell_start, cell_items, 1)
        if d < 0.0:
            break
        if surf != SURF_WALL:
            return (oy - ray_y) + d, surf
        ray_y -= d + 0.001
        left -= d + 0.001
        if left <= 0.0:
            break
    return -1.0, SURF_NONE


CRACK_PROBE_M = 0.15     # surface_below: how far around a miss to look for the ground (cracks <= 0.3 m)


@njit(cache=True)
def surface_below(x, oy, z, max_distance, tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items):
    """The surface code of the ground below (x, oy, z) (ground_below), or SURF_NONE when there is
    none. Collision meshes have cracks a few cm wide between triangles (Klutch Kickers: 0.3% of
    the road near the line), which a single vertical ray falls through; a miss is retried at four
    points CRACK_PROBE_M around, so only a real hole or the edge of the world reads as none."""
    d, surf = ground_below(x, oy, z, max_distance, tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items)
    if d >= 0.0:
        return surf
    for k in range(4):
        dx = CRACK_PROBE_M if k == 0 else (-CRACK_PROBE_M if k == 1 else 0.0)
        dz = CRACK_PROBE_M if k == 2 else (-CRACK_PROBE_M if k == 3 else 0.0)
        d, surf = ground_below(x + dx, oy, z + dz, max_distance, tri_v, tri_surf, gmin, cell_size, dims,
                               cell_start, cell_items)
        if d >= 0.0:
            return surf
    return SURF_NONE


N_OUTLINE = 8


@njit(cache=True)
def _outline_point(i, front, rear, half):
    """Point i of the car's outline (CarOutlineContact), metres (forward, left) of its centre:
    the front corners and middle, the sides' middles, the rear corners and middle. The plugin's
    CarOutline.Point is the same list."""
    if i == 0:
        return front, half
    if i == 1:
        return front, 0.0
    if i == 2:
        return front, -half
    if i == 3:
        return 0.0, half
    if i == 4:
        return 0.0, -half
    if i == 5:
        return -rear, half
    if i == 6:
        return -rear, 0.0
    return -rear, -half


@njit(cache=True)
def raycast(ox, oy, oz, dx, dy, dz, max_distance, tri_v, tri_surf, gmin, cell_size, dims,
            cell_start, cell_items, which):
    """Returns (distance, surface_code); distance < 0 means no hit within max_distance. which:
    0 = the sensors' triangles, 1 = the collision triangles (what the game collides with), 2 = only
    the walls among the collision triangles (the contact pins: a rise in the road just ahead must
    not hide a low wall behind it)."""
    walls_only = which == 2
    if walls_only:
        which = 1
    l = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= l
    dy /= l
    dz /= l
    # Per-axis DDA set-up, written out as scalars (no per-ray array allocation - this function
    # is the simulator's hot path).
    inf = np.inf
    cx = int((ox - gmin[0]) / cell_size)
    cx = 0 if cx < 0 else (dims[0] - 1 if cx > dims[0] - 1 else cx)
    cy = int((oy - gmin[1]) / cell_size)
    cy = 0 if cy < 0 else (dims[1] - 1 if cy > dims[1] - 1 else cy)
    cz = int((oz - gmin[2]) / cell_size)
    cz = 0 if cz < 0 else (dims[2] - 1 if cz > dims[2] - 1 else cz)
    stx = 0; sty = 0; stz = 0
    tmx = inf; tmy = inf; tmz = inf
    tdx = inf; tdy = inf; tdz = inf
    if dx > 1e-12:
        stx = 1
        tmx = ((gmin[0] + (cx + 1) * cell_size) - ox) / dx
        tdx = cell_size / dx
    elif dx < -1e-12:
        stx = -1
        tmx = ((gmin[0] + cx * cell_size) - ox) / dx
        tdx = cell_size / -dx
    if dy > 1e-12:
        sty = 1
        tmy = ((gmin[1] + (cy + 1) * cell_size) - oy) / dy
        tdy = cell_size / dy
    elif dy < -1e-12:
        sty = -1
        tmy = ((gmin[1] + cy * cell_size) - oy) / dy
        tdy = cell_size / -dy
    if dz > 1e-12:
        stz = 1
        tmz = ((gmin[2] + (cz + 1) * cell_size) - oz) / dz
        tdz = cell_size / dz
    elif dz < -1e-12:
        stz = -1
        tmz = ((gmin[2] + cz * cell_size) - oz) / dz
        tdz = cell_size / -dz
    d1 = dims[1]
    d2 = dims[2]

    best_t = -1.0
    best_surface = SURF_NONE
    eps = 1e-6
    guard = 0
    while guard < 100000:
        guard += 1
        c = (cx * d1 + cy) * d2 + cz
        for k in range(cell_start[2 * c + which], cell_start[2 * c + which + 1]):
            idx = cell_items[k]
            if walls_only and tri_surf[idx] != SURF_WALL:
                continue
            v1x = tri_v[idx, 0, 0]; v1y = tri_v[idx, 0, 1]; v1z = tri_v[idx, 0, 2]
            e1x = tri_v[idx, 1, 0] - v1x; e1y = tri_v[idx, 1, 1] - v1y; e1z = tri_v[idx, 1, 2] - v1z
            e2x = tri_v[idx, 2, 0] - v1x; e2y = tri_v[idx, 2, 1] - v1y; e2z = tri_v[idx, 2, 2] - v1z
            hx = dy * e2z - dz * e2y
            hy = dz * e2x - dx * e2z
            hz = dx * e2y - dy * e2x
            aa = e1x * hx + e1y * hy + e1z * hz
            if -eps < aa < eps:
                continue
            f = 1.0 / aa
            sx = ox - v1x; sy = oy - v1y; sz = oz - v1z
            u = f * (sx * hx + sy * hy + sz * hz)
            if u < 0.0 or u > 1.0:
                continue
            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x
            v = f * (dx * qx + dy * qy + dz * qz)
            if v < 0.0 or u + v > 1.0:
                continue
            t = f * (e2x * qx + e2y * qy + e2z * qz)
            if t > eps and t <= max_distance and (best_t < 0.0 or t < best_t):
                best_t = t
                best_surface = tri_surf[idx]

        next_boundary = min(tmx, tmy, tmz)
        if best_t >= 0.0 and best_t <= next_boundary:
            return best_t, best_surface
        if next_boundary > max_distance:
            return best_t, best_surface
        # first axis holding the minimum (x, then y, then z) - same tie-break as list.index(min)
        if tmx <= tmy and tmx <= tmz:
            cx += stx
            if not (0 <= cx < dims[0]):
                return best_t, best_surface
            tmx += tdx
        elif tmy <= tmz:
            cy += sty
            if not (0 <= cy < dims[1]):
                return best_t, best_surface
            tmy += tdy
        else:
            cz += stz
            if not (0 <= cz < dims[2]):
                return best_t, best_surface
            tmz += tdz
    return best_t, best_surface


@njit(cache=True)
def wrap_pi(x):
    return (x + math.pi) % TWO_PI - math.pi


@njit(cache=True)
def rear_sliding_grip(s):
    """S_REAR_SLIDE_GRIP, read safely: anything outside (0, 1] (a state row built without it) is 1."""
    g = s[S_REAR_SLIDE_GRIP]
    if g > 0.0 and g <= 1.0:
        return g
    return 1.0


@njit(cache=True)
def _tire_curve(slip_mag, peak_slip_angle_rad, round_curve):
    """How much of its peak force an axle gives at slip angle slip_mag (0 - 1): a straight line up
    to peak_slip_angle_rad, or with round_curve (RoundTireCurve) the brush (Fiala) curve, which
    starts three times as steep and reaches full sliding at peak_slip_angle_rad:
    3x - 3x^2 + x^3, x = tan(slip) / tan(full sliding)."""
    if slip_mag >= peak_slip_angle_rad:
        return 1.0
    if round_curve:
        x = math.tan(slip_mag) / math.tan(peak_slip_angle_rad)
        return x * (3.0 - 3.0 * x + x * x)
    return slip_mag / peak_slip_angle_rad


@njit(cache=True)
def tire_curve_angles(p):
    """(front, rear) slip angle where each axle's force reaches its peak: the tyres' friction limit
    angle, plus with RoundTireCurve FLEX x the static load on one of its tires."""
    front = p[P_FRONT_PEAK_SLIP_RAD]
    rear = p[P_REAR_PEAK_SLIP_RAD]
    if p[P_EXT_ROUND_TIRE_CURVE] > 0.5:
        front += math.radians(p[P_FRONT_TIRE_FLEX] * 0.5 * p[P_FRONT_STATIC_LOAD_N])
        rear += math.radians(p[P_REAR_TIRE_FLEX] * 0.5 * p[P_REAR_STATIC_LOAD_N])
    return front, rear


@njit(cache=True)
def _lateral_force_within(slip_angle, peak_force, peak_slip_angle_rad, capacity, round_curve):
    """Sideways force of an axle that rises with slip angle as _lateral_tire_force does and stops at
    `capacity`: what the friction circle leaves after the force along the car. Opposes the slip."""
    force_mag = min(peak_force * _tire_curve(abs(slip_angle), peak_slip_angle_rad, round_curve), capacity)
    if slip_angle > 0.0:
        return -force_mag
    if slip_angle < 0.0:
        return force_mag
    return 0.0


@njit(cache=True)
def _lateral_tire_force(slip_angle, peak_friction, static_load_n, peak_slip_angle_rad, sliding_grip,
                        round_curve):
    """Sideways force of one axle: rises to its peak at peak_slip_angle_rad (linearly, or along the
    brush curve with round_curve: _tire_curve), then falls
    linearly to sliding_grip x peak at twice that angle and stays there. sliding_grip = 1 is the
    flat-topped curve the model had at first (bit for bit: the force is peak_force itself).

    Why it exists: with a flat top the rear axle of a car in a held drift always pushes with its
    full peak grip, and a held drift then has ONE speed per corner radius (sideways acceleration
    about 13 m/s2, so about 73 km/h through this course's 30 m corners). A human-driven car on the
    same course slid at a median 42 km/h with 6.6 m/s2 (ninth server test): driven
    wheels that are spinning have much less sideways grip than their peak."""
    peak_force = peak_friction * static_load_n
    mag = abs(slip_angle)
    if mag < peak_slip_angle_rad:
        force_mag = peak_force * _tire_curve(mag, peak_slip_angle_rad, round_curve)
    elif sliding_grip >= 1.0:
        force_mag = peak_force
    else:
        past_peak = (mag - peak_slip_angle_rad) / peak_slip_angle_rad
        if past_peak >= 1.0:
            force_mag = peak_force * sliding_grip
        else:
            force_mag = peak_force * (1.0 - (1.0 - sliding_grip) * past_peak)
    if slip_angle > 0:
        return -force_mag
    if slip_angle < 0:
        return force_mag
    return 0.0


# ----------------------------------------------------------------------------------------
# Spline queries (ai_spline.py)
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def world_to_spline(px, py, pz, sp_pos):
    best_id = -1
    best = np.inf
    for i in range(sp_pos.shape[0]):
        d = (sp_pos[i, 0] - px) ** 2 + (sp_pos[i, 1] - py) ** 2 + (sp_pos[i, 2] - pz) ** 2
        if d < best:
            best = d
            best_id = i
    return best_id


@njit(cache=True)
def find_interpolation_segment(px, py, pz, nearest, sp_pos, sp_next, sp_prev):
    n = sp_pos.shape[0]
    if nearest < 0 or nearest >= n:
        return nearest, nearest, 0.0
    best_d = np.inf
    best_a, best_b, best_t = nearest, nearest, 0.0
    nx_ = sp_next[nearest]
    if nx_ >= 0:
        sx = sp_pos[nx_, 0] - sp_pos[nearest, 0]
        sy = sp_pos[nx_, 1] - sp_pos[nearest, 1]
        sz = sp_pos[nx_, 2] - sp_pos[nearest, 2]
        seg_len_sq = sx * sx + sy * sy + sz * sz
        if seg_len_sq > 0.0001:
            t = ((px - sp_pos[nearest, 0]) * sx + (py - sp_pos[nearest, 1]) * sy
                 + (pz - sp_pos[nearest, 2]) * sz) / seg_len_sq
            t = max(0.0, min(1.0, t))
            qx = sp_pos[nearest, 0] + sx * t
            qy = sp_pos[nearest, 1] + sy * t
            qz = sp_pos[nearest, 2] + sz * t
            d = (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2
            if d < best_d:
                best_d, best_a, best_b, best_t = d, nearest, nx_, t
    pv = sp_prev[nearest]
    if pv >= 0:
        sx = sp_pos[nearest, 0] - sp_pos[pv, 0]
        sy = sp_pos[nearest, 1] - sp_pos[pv, 1]
        sz = sp_pos[nearest, 2] - sp_pos[pv, 2]
        seg_len_sq = sx * sx + sy * sy + sz * sz
        if seg_len_sq > 0.0001:
            t = ((px - sp_pos[pv, 0]) * sx + (py - sp_pos[pv, 1]) * sy
                 + (pz - sp_pos[pv, 2]) * sz) / seg_len_sq
            t = max(0.0, min(1.0, t))
            qx = sp_pos[pv, 0] + sx * t
            qy = sp_pos[pv, 1] + sy * t
            qz = sp_pos[pv, 2] + sz * t
            d = (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2
            if d < best_d:
                best_d, best_a, best_b, best_t = d, pv, nearest, t
    return best_a, best_b, best_t


@njit(cache=True)
def _np_reachable(start, target, max_d, forward, sp_length, sp_next, sp_prev):
    n = sp_length.shape[0]
    cur = start
    traveled = 0.0
    for _ in range(n):
        nb = sp_next[cur] if forward else sp_prev[cur]
        if nb < 0 or nb >= n:
            return False
        traveled += sp_length[cur] if forward else sp_length[nb]
        if traveled > max_d:
            return False
        if nb == target:
            return True
        cur = nb
    return False


@njit(cache=True)
def _np_closest_within(start, max_d, px, py, pz, sp_pos, sp_length, sp_next, sp_prev):
    """The point nearest (px, py, pz) among those within max_d of start along the line, either way."""
    n = sp_length.shape[0]
    best_id = start
    best = (sp_pos[start, 0] - px) ** 2 + (sp_pos[start, 1] - py) ** 2 + (sp_pos[start, 2] - pz) ** 2
    for direction in range(2):
        forward = direction == 0
        cur = start
        traveled = 0.0
        for _ in range(n):
            nb = sp_next[cur] if forward else sp_prev[cur]
            if nb < 0 or nb >= n:
                break
            traveled += sp_length[cur] if forward else sp_length[nb]
            if traveled > max_d:
                break
            d = (sp_pos[nb, 0] - px) ** 2 + (sp_pos[nb, 1] - py) ** 2 + (sp_pos[nb, 2] - pz) ** 2
            if d < best:
                best = d
                best_id = nb
            cur = nb
    return best_id


@njit(cache=True)
def nearest_point_correct(raw_id, prev_id, prev_streak, own_speed, dt, px, py, pz, sp_pos, sp_length, sp_next,
                          sp_prev):
    """Returns (corrected_id, glitch_corrected, new_streak). prev_id < 0 = no history."""
    n = sp_length.shape[0]
    if prev_id < 0:
        return raw_id, False, 0
    if prev_id >= n or raw_id < 0 or raw_id >= n:
        return raw_id, False, 0
    if raw_id == prev_id:
        return raw_id, False, 0
    if prev_streak >= 60:               # MaxConsecutiveCorrectionTicks
        return raw_id, False, 0
    max_d = max(own_speed * dt * 2.0, 10.0)   # PlausibilityMarginFactor, MinimumPlausibilityFloorM
    if (_np_reachable(prev_id, raw_id, max_d, True, sp_length, sp_next, sp_prev)
            or _np_reachable(prev_id, raw_id, max_d, False, sp_length, sp_next, sp_prev)):
        return raw_id, False, 0
    # out of reach (another part of the track passing close): the nearest point that is in reach
    return _np_closest_within(prev_id, max_d, px, py, pz, sp_pos, sp_length, sp_next, sp_prev), True, prev_streak + 1


# ----------------------------------------------------------------------------------------
# One tick
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def step(s, act_steer, act_speed, act_brake, dt_s, p, diag, mid,
         sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
         tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
         rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk):
    """One whole tick = step_pre + step_post. Advances state row `s` in place and fills `diag`.
    act_steer/act_speed are the policy's first two actions, in [-1, 1]; act_brake its third, in
    [0, 1] (0 for a policy with two)."""
    step_pre(s, dt_s, p, mid, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd)
    step_post(s, act_steer, act_speed, act_brake, p, diag, mid,
              sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
              tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
              rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk)


@njit(cache=True)
def step_pre(s, dt_s, p, mid, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd):
    """Everything the plugin knows before the policy acts: the nearest spline point, the spline's
    direction there and the cross-track error. Writes `mid`; touches only the
    NearestPointCorrector history in `s`."""
    px, py, pz = s[S_X], s[S_Y], s[S_Z]
    heading = s[S_HEADING]
    is_first_tick = math.isnan(s[S_CORR_ANGLE])
    max_lat_acc = p[P_MAX_LATERAL_ACCEL_MS2]

    # --- nearest point + raw spline angle ---
    raw_nearest = world_to_spline(px, py, pz, sp_pos)
    nearest = raw_nearest
    np_glitch = False
    if p[P_EXT_NEAREST_POINT_CORRECTOR] > 0.5:
        own_for_check = 0.5 if is_first_tick else s[S_OWN_SPEED]
        nearest, np_glitch, np_streak = nearest_point_correct(
            raw_nearest, int(s[S_PREV_NEAREST]), int(s[S_NP_STREAK]), own_for_check, dt_s,
            px, py, pz, sp_pos, sp_length, sp_next, sp_prev)
        s[S_PREV_NEAREST] = nearest
        s[S_NP_STREAK] = np_streak
    a, b, t = find_interpolation_segment(px, py, pz, nearest, sp_pos, sp_next, sp_prev)
    radius = sp_radius[nearest]
    fx = sp_fwd[a, 0] + (sp_fwd[b, 0] - sp_fwd[a, 0]) * t
    fz = sp_fwd[a, 2] + (sp_fwd[b, 2] - sp_fwd[a, 2]) * t
    norm = math.sqrt(fx * fx + fz * fz)
    if norm < 0.01:
        raw_angle = heading if is_first_tick else s[S_CORR_ANGLE]
    else:
        raw_angle = math.atan2(fz / norm, fx / norm)

    # --- glitch correction (own speed = LAST tick's) ---
    glitch = False
    corrected_angle = raw_angle
    if not is_first_tick:
        raw_step = wrap_pi(raw_angle - s[S_CORR_ANGLE])
        max_plausible_step = (2.0 * max_lat_acc) / max(s[S_OWN_SPEED], 0.5) * dt_s
        if abs(raw_step) > max_plausible_step:
            corrected_angle = s[S_CORR_ANGLE] + math.copysign(max_plausible_step, raw_step)
            glitch = True

    # --- cross-track error ---
    ix = sp_pos[a, 0] + (sp_pos[b, 0] - sp_pos[a, 0]) * t
    iz = sp_pos[a, 2] + (sp_pos[b, 2] - sp_pos[a, 2]) * t
    cdx = math.cos(corrected_angle)
    cdz = math.sin(corrected_angle)
    cte = (px - ix) * (-cdz) + (pz - iz) * cdx

    mid[M_NEAREST] = nearest
    mid[M_A] = a
    mid[M_B] = b
    mid[M_T] = t
    mid[M_RADIUS] = radius
    mid[M_CORR_ANGLE] = corrected_angle
    mid[M_GLITCH] = 1.0 if glitch else 0.0
    mid[M_CTE] = cte
    mid[M_FIRST] = 1.0 if is_first_tick else 0.0
    mid[M_NP_GLITCH] = 1.0 if np_glitch else 0.0
    mid[M_DT] = dt_s


@njit(cache=True)
def axle_friction(p, s):
    """(front, rear) peak friction of each axle this tick. The tyre file's peak (DY0) is the grip
    of a tire with no load on it; each kN on a tire takes DY1 off (EnableTireLoadSensitivity).
    An axle's two tires share its load evenly at rest; cornering moves
    m * a_y * CG height / track of it onto the outside tire, split between the axles by their
    roll stiffness. With grip falling as load rises, the heavier outside tire gains less than the
    inner one loses: per axle mu = DY0 + DY1 * (F/2 + 2 dF^2 / F), loads in kN. The sideways
    acceleration is the steady-state one, forward speed x yaw rate, from the tick before. Off:
    DY0, as before."""
    mu_f = p[P_LAT_FRONT_PEAK_FRICTION]
    mu_r = p[P_LAT_REAR_PEAK_FRICTION]
    if p[P_EXT_TIRE_LOAD_SENSITIVITY] < 0.5:
        return mu_f, mu_r
    f_load = p[P_FRONT_STATIC_LOAD_N]
    r_load = p[P_REAR_STATIC_LOAD_N]
    mass = (f_load + r_load) / 9.80665
    a_y = min(abs(s[S_OWN_SPEED] * s[S_YAW]), 2.0 * 9.80665)
    moved = mass * a_y * p[P_CG_HEIGHT_M]
    df = min(0.5 * f_load, p[P_ROLL_STIFFNESS_FRONT_SHARE] * moved / max(p[P_FRONT_TRACK_M], 0.5))
    dr = min(0.5 * r_load, (1.0 - p[P_ROLL_STIFFNESS_FRONT_SHARE]) * moved / max(p[P_REAR_TRACK_M], 0.5))
    if p[P_FRONT_LS_EXPY] > 0.0:
        mu_f = _power_law_axle(mu_f, f_load, df, p[P_FRONT_FZ0_N], p[P_FRONT_LS_EXPY])
    else:
        mu_f += p[P_FRONT_LOAD_SENSITIVITY_PER_KN] * (0.5 * f_load + 2.0 * df * df / f_load) / 1000.0
    if p[P_REAR_LS_EXPY] > 0.0:
        mu_r = _power_law_axle(mu_r, r_load, dr, p[P_REAR_FZ0_N], p[P_REAR_LS_EXPY])
    else:
        mu_r += p[P_REAR_LOAD_SENSITIVITY_PER_KN] * (0.5 * r_load + 2.0 * dr * dr / r_load) / 1000.0
    return max(mu_f, 0.1), max(mu_r, 0.1)


@njit(cache=True)
def _power_law_axle(peak, axle_load, moved, fz0, ls_expy):
    """An axle's friction under the newer tyre model: each tire's grip is
    peak x (load / FZ0)^(LS_EXPY - 1); the axle's is the two tires' forces over its load."""
    outer = 0.5 * axle_load + moved
    inner = max(0.5 * axle_load - moved, 1.0)
    fz0 = max(fz0, 1.0)
    force = outer * peak * (outer / fz0) ** (ls_expy - 1.0) + inner * peak * (inner / fz0) ** (ls_expy - 1.0)
    return force / max(outer + inner, 1.0)


@njit(cache=True)
def engine_force(p, speed_ms):
    """The most force the drivetrain can put on the road at this forward speed (EnableEngine),
    from the car's curve, linearly between its points; 0 past its last one."""
    x = max(speed_ms, 0.0) / ENGINE_CURVE_STEP_MS
    k = int(x)
    if k >= ENGINE_CURVE_POINTS - 1:
        return 0.0
    w = x - k
    return p[P_ENGINE_FORCE_00 + k] * (1.0 - w) + p[P_ENGINE_FORCE_00 + k + 1] * w


@njit(cache=True)
def step_post(s, act_steer, act_speed, act_brake, p, diag, mid,
              sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
              tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
              rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk):
    """The rest of the tick: forward speed, lateral dynamics, recovery, integration, sensing."""
    px, py, pz = s[S_X], s[S_Y], s[S_Z]
    heading = s[S_HEADING]
    nearest = int(mid[M_NEAREST])
    a = int(mid[M_A])
    b = int(mid[M_B])
    t = mid[M_T]
    radius = mid[M_RADIUS]
    corrected_angle = mid[M_CORR_ANGLE]
    glitch = mid[M_GLITCH] > 0.5
    cte = mid[M_CTE]
    is_first_tick = mid[M_FIRST] > 0.5
    np_glitch = mid[M_NP_GLITCH] > 0.5
    dt_s = mid[M_DT]
    mu_f, mu_r = axle_friction(p, s)     # each axle's peak friction this tick (load sensitivity)
    curve_f, curve_r = tire_curve_angles(p)
    round_curve = p[P_EXT_ROUND_TIRE_CURVE] > 0.5

    # --- forward speed ---
    # The speed action is the target speed as a fraction of MaxSpeed (-1 => stop, +1 =>
    # MaxSpeed); the accel/decel ramp below limits how fast own-speed gets there. With
    # EnableRearFrictionCircle it is a pedal instead (next block).
    desired_speed = s[S_MAX_SPEED] * 0.5 * (1.0 + act_speed)

    prev_own = s[S_OWN_SPEED]
    # --- rear friction circle (EnableRearFrictionCircle) ---
    # The speed action is a pedal, not a target speed: above zero it is the share of the thrust
    # the rear tires are driven with, below zero the share of the brakes. The forward speed is
    # then whatever the forces along the car make it: dv_x/dt = v_y * r + F_x / m. And the rear
    # tires have one friction circle: what goes into thrust (or braking) is not there for
    # sideways grip, sqrt(peak^2 - F_x^2) is. Lift, and a sliding rear axle grips at its peak;
    # full throttle leaves it RearSlidingGrip x the peak. A human's car on this course, rear
    # sliding at 20 - 60 deg: 1.26 x its load sideways with no wheelspin, 0.54 at full throttle.
    #
    # The brake action (a third output): braking WITH the throttle, the way a drifter
    # scrubs off speed to hold a gap without giving up the angle. It works the front axle only -
    # the throttle keeps the rear wheels turning, so the rear stays loose and the angle holds - at
    # up to the front's share of the car's braking, never more than the front tires can take. The
    # front tires have one friction circle too: what goes into braking is not there for steering,
    # sqrt(peak^2 - F_x^2) is. A brake action of 0 is exactly the model without it.
    #
    # Weight transfer (EnableWeightTransfer): slowing the car moves load onto the front
    # axle and speeding it up onto the rear, m x a_x x CG height / wheelbase, the suspension
    # settling over LoadTransferTimeConstantS; each axle's grip and friction circle scale with its
    # load. Braking with the throttle then lightens the rear as well as slowing the car - the rear
    # lets go more easily and the front bites harder - which is what lets a drifter scrub speed
    # and keep the angle. Without it braking only took the front's grip, and a car held to 85% of
    # its pace slid 30 - 57% of the time against 64 - 72% (README, "Pace results"). The load
    # shift used is the one left by the tick before (S_LOAD_SHIFT); this tick's forces set the next.
    pedal_mode = p[P_EXT_REAR_FRICTION_CIRCLE] > 0.5
    if pedal_mode and p[P_EXT_NO_BRAKES] > 0.5:
        act_speed = max(0.0, act_speed)     # DisableBrakes: lift-off at most, no brake action
        act_brake = 0.0
    if pedal_mode and p[P_EXT_FRONT_BRAKE_ONLY] > 0.5:
        act_speed = max(0.0, act_speed)     # FrontBrakeOnly: no foot brake; the brake action stays
    front_load = p[P_FRONT_STATIC_LOAD_N]    # each axle's load this tick (static unless weight transfer)
    rear_load = p[P_REAR_STATIC_LOAD_N]
    rear_capacity = -1.0            # sideways force the rear axle has left this tick; < 0 = not in use
    front_capacity = -1.0           # the same for the front axle, while it brakes
    if pedal_mode and not is_first_tick and dt_s > 1e-4:
        t_mass = (p[P_FRONT_STATIC_LOAD_N] + p[P_REAR_STATIC_LOAD_N]) / 9.80665
        if p[P_EXT_WEIGHT_TRANSFER] > 0.5:
            shift = max(-0.8 * p[P_FRONT_STATIC_LOAD_N], min(0.8 * p[P_REAR_STATIC_LOAD_N], s[S_LOAD_SHIFT]))
            front_load = p[P_FRONT_STATIC_LOAD_N] + shift
            rear_load = p[P_REAR_STATIC_LOAD_N] - shift
        t_vx = max(prev_own, p[P_MIN_SLIP_SPEED_MS])
        t_steer = s[S_LAST_STEER]
        t_front_slip = math.atan2(s[S_LATV] + p[P_DIST_CG_FRONT_M] * s[S_YAW], t_vx) - t_steer
        t_fy_f = _lateral_tire_force(t_front_slip, mu_f, front_load, curve_f, 1.0, round_curve)
        t_rear_peak = mu_r * rear_load
        front_along = -t_fy_f * math.sin(t_steer)
        slide_loss = s[S_LATV] * s[S_YAW]
        grip_at_full_throttle = rear_sliding_grip(s)
        rear_share = p[P_REAR_STATIC_LOAD_N] / (p[P_FRONT_STATIC_LOAD_N] + p[P_REAR_STATIC_LOAD_N])
        t_front_peak = mu_f * front_load
        brake = max(0.0, min(1.0, act_brake))
        front_brake_fx = -brake * min((1.0 - rear_share) * p[P_MAX_LONGITUDINAL_DECEL_MS2] * t_mass, t_front_peak)
        if act_speed >= 0.0:
            rear_fx = act_speed * t_rear_peak * math.sqrt(max(0.0, 1.0 - grip_at_full_throttle * grip_at_full_throttle))
            # EnableEngine: no more than the engine can put down at this speed. A car without the
            # power to spin its rear tires keeps their sideways grip at full throttle.
            if p[P_EXT_ENGINE] > 0.5:
                rear_fx = min(rear_fx, act_speed * engine_force(p, prev_own))
            # no thrust past the car's speed limit
            to_limit = t_mass * ((s[S_MAX_SPEED] - prev_own) / dt_s - slide_loss) - front_along - front_brake_fx
            rear_fx = max(0.0, min(rear_fx, to_limit))
            front_fx = front_brake_fx
        else:
            brake_force = -act_speed * p[P_MAX_LONGITUDINAL_DECEL_MS2] * t_mass
            rear_fx = -min(rear_share * brake_force, t_rear_peak)
            front_fx = -(1.0 - rear_share) * brake_force + front_brake_fx
        if front_fx < 0.0:          # braking, by the pedal or the brake action
            front_fx = max(front_fx, -t_front_peak)
            front_capacity = math.sqrt(max(0.0, t_front_peak * t_front_peak - front_fx * front_fx))
        new_own = max(0.0, prev_own + (slide_loss + (rear_fx + front_fx + front_along) / t_mass) * dt_s)
        rear_capacity = math.sqrt(max(0.0, t_rear_peak * t_rear_peak - rear_fx * rear_fx))
        if p[P_EXT_WEIGHT_TRANSFER] > 0.5:
            # the forces along the car this tick set the load shift the next tick starts from
            accel_along = (rear_fx + front_fx + front_along) / t_mass
            target_shift = -t_mass * accel_along * p[P_CG_HEIGHT_M] / (p[P_DIST_CG_FRONT_M] + p[P_DIST_CG_REAR_M])
            s[S_LOAD_SHIFT] += (target_shift - s[S_LOAD_SHIFT]) * min(1.0, dt_s / max(p[P_LOAD_TRANSFER_TIME_S], 1e-3))
    elif is_first_tick:
        new_own = desired_speed
    else:
        if desired_speed > prev_own:
            ramp = p[P_AI_ACCELERATION_MS2]
        else:
            ramp = p[P_AI_DECELERATION_MS2]
            if p[P_ENABLE_LONGITUDINAL_DECEL_CAP] > 0.5:
                ramp = min(ramp, p[P_MAX_LONGITUDINAL_DECEL_MS2])
        max_delta = ramp * dt_s
        new_own = prev_own + max(-max_delta, min(max_delta, desired_speed - prev_own))
    # --- rear thrust limit (EnableRearThrustLimit) ---
    # Without it the forward speed simply follows the ramp above, whatever the car is doing. But a
    # car that is sliding sideways (lateral velocity v_y) while it yaws (r) loses forward speed at
    # v_y * r unless something pushes it: dv_x/dt = v_y * r + F_x / m. Holding the speed in a slide
    # therefore takes thrust, more of it the larger the slip angle - and a free, unlimited supply
    # of it is what let a 60-degree-lock policy (angle1) slide at a median 84 km/h with 1.4 g of
    # sideways acceleration, 2.5 g at the 90th percentile; a human on this course does 0.76 g.
    # The thrust the rear tires can actually give is what the friction circle leaves after their
    # sideways force: sqrt(peak^2 - F_yr^2), i.e. sqrt(1 - s^2) x peak for a rear axle sliding at
    # share s (RearSlidingGrip). The front tires' sideways force also has a part along the car:
    # -F_yf * sin(steer), which pulls a countersteered car forward. Forces are evaluated from the
    # state the tick starts in (last tick's lateral velocity, yaw rate and steering angle).
    # The human recording shows that push being used: forward acceleration rises with slip angle
    # the way v_y * r does (0.55 g against 0.55 g at 35-50 deg of rear slip, 0.65 against 0.78 at
    # 50-65, 0.72 against 0.82 beyond, where the car is losing speed).
    if p[P_EXT_REAR_THRUST_LIMIT] > 0.5 and not pedal_mode and not is_first_tick and dt_s > 1e-4:
        t_mass = (p[P_FRONT_STATIC_LOAD_N] + p[P_REAR_STATIC_LOAD_N]) / 9.80665
        t_vx = max(prev_own, p[P_MIN_SLIP_SPEED_MS])
        t_steer = s[S_LAST_STEER]
        t_front_slip = math.atan2(s[S_LATV] + p[P_DIST_CG_FRONT_M] * s[S_YAW], t_vx) - t_steer
        t_rear_slip = math.atan2(s[S_LATV] - p[P_DIST_CG_REAR_M] * s[S_YAW], t_vx)
        t_fy_f = _lateral_tire_force(t_front_slip, mu_f, p[P_FRONT_STATIC_LOAD_N], curve_f, 1.0, round_curve)
        t_fy_r = _lateral_tire_force(t_rear_slip, mu_r, p[P_REAR_STATIC_LOAD_N], curve_r, rear_sliding_grip(s),
                                     round_curve)
        t_rear_peak = mu_r * p[P_REAR_STATIC_LOAD_N]
        thrust_max = math.sqrt(max(0.0, t_rear_peak * t_rear_peak - t_fy_r * t_fy_r))
        front_along = -t_fy_f * math.sin(t_steer)
        slide_loss = s[S_LATV] * s[S_YAW]                 # dv_x/dt with no force along the car
        thrust_needed = t_mass * ((new_own - prev_own) / dt_s - slide_loss) - front_along
        if thrust_needed > thrust_max:
            new_own = max(0.0, prev_own + (slide_loss + (thrust_max + front_along) / t_mass) * dt_s)
        # EnableRearBrakeLimit: the same limit the other way. When the car swings back toward
        # its direction of travel v_y * r is positive and forward speed rises by itself; holding
        # it down takes a force pulling back, and the rear tires have no more of that than of
        # thrust. Without this the policy `angle4` held the car back with more than they have
        # on 6.9% of the ticks of its server run, by up to 1.9 g.
        elif p[P_EXT_REAR_BRAKE_LIMIT] > 0.5 and thrust_needed < -thrust_max:
            new_own = max(0.0, prev_own + (slide_loss + (front_along - thrust_max) / t_mass) * dt_s)
    realized_long_acc = 0.0 if (is_first_tick or dt_s <= 1e-4) else (new_own - prev_own) / dt_s
    body_fwd = max(new_own, 0.5)

    # --- lateral dynamics ---
    speed_total_ms = math.sqrt(body_fwd ** 2 + s[S_LATV] ** 2)
    dt = (dt_s * 1000.0) / 1000.0
    speed_total = (speed_total_ms * 3.6) / 3.6
    body_speed = max(math.sqrt(max(speed_total ** 2 - s[S_LATV] ** 2, 0.0)), 0.5)

    sub_n = max(1, int(math.ceil(dt / p[P_MAX_SUBSTEP_S])))
    sub_dt = dt / sub_n
    run_latv = s[S_LATV]
    run_yaw = math.radians(math.degrees(s[S_YAW]))
    run_heading = math.radians(math.degrees(heading))

    mass = (p[P_FRONT_STATIC_LOAD_N] + p[P_REAR_STATIC_LOAD_N]) / 9.80665
    rear_peak_force = mu_r * rear_load
    # With the thrust limit the rear's sideways force comes first and thrust takes what is left
    # (above), so the older coupling - forward acceleration eating into sideways grip - is off.
    rear_drive = 0.0 if (p[P_EXT_REAR_THRUST_LIMIT] > 0.5 or pedal_mode) else max(0.0, mass * realized_long_acc)
    rear_avail = math.sqrt(max(0.0, rear_peak_force ** 2 - rear_drive ** 2))
    eff_rear_fric = rear_avail / rear_load if rear_load > 1e-4 else 0.0
    max_steer = p[P_MAX_STEER_RAD]

    # The steering action is the steering angle, as a fraction of full lock, held for every
    # sub-step of the tick. MaxSteerRateDegS: the wheels move toward it no faster than a steering
    # wheel can be turned (from where they were last tick; not on a car's first tick).
    steer_cmd = max_steer * act_steer
    if p[P_MAX_STEER_RATE_RAD_S] > 0.0 and not is_first_tick:
        steer_step = p[P_MAX_STEER_RATE_RAD_S] * dt_s
        steer_cmd = max(s[S_LAST_STEER] - steer_step, min(s[S_LAST_STEER] + steer_step, steer_cmd))
    cos_steer = math.cos(steer_cmd)

    front_slip = 0.0
    rear_slip = 0.0
    fy_f = 0.0
    fy_r = 0.0
    rear_peak_used = 0.0
    slide_grip = rear_sliding_grip(s)
    for k in range(sub_n):
        slip_speed = max(body_speed, p[P_MIN_SLIP_SPEED_MS])
        front_slip = math.atan2(run_latv + p[P_DIST_CG_FRONT_M] * run_yaw, slip_speed) - steer_cmd
        rear_slip = math.atan2(run_latv - p[P_DIST_CG_REAR_M] * run_yaw, slip_speed)
        fy_f = _lateral_tire_force(front_slip, mu_f, front_load, curve_f, 1.0, round_curve)
        if front_capacity >= 0.0:
            fy_f = _lateral_force_within(front_slip, mu_f * front_load, curve_f, front_capacity, round_curve)
        rear_fric_used = eff_rear_fric if p[P_ENABLE_FRICTION_ELLIPSE] > 0.5 else mu_r
        rear_peak_used = rear_fric_used * rear_load
        fy_r = _lateral_tire_force(rear_slip, rear_fric_used, rear_load, curve_r, slide_grip, round_curve)
        if rear_capacity >= 0.0:
            fy_r = _lateral_force_within(rear_slip, rear_peak_force, curve_r, rear_capacity, round_curve)
            rear_peak_used = rear_capacity

        yaw_moment = p[P_DIST_CG_FRONT_M] * fy_f * cos_steer - p[P_DIST_CG_REAR_M] * fy_r
        new_yaw = run_yaw + (yaw_moment / p[P_YAW_INERTIA_KGM2]) * sub_dt
        lat_acc = (fy_f * cos_steer + fy_r) / mass - body_speed * run_yaw
        new_latv = run_latv + lat_acc * sub_dt
        run_heading = wrap_pi(run_heading + new_yaw * sub_dt)
        run_yaw = new_yaw
        run_latv = new_latv

    front_peak_force = mu_f * front_load
    front_sat = abs(fy_f) / front_peak_force if front_peak_force > 1e-6 else 0.0
    rear_sat = abs(fy_r) / rear_peak_used if rear_peak_used > 1e-6 else 0.0

    new_heading = wrap_pi(math.radians(math.degrees(run_heading)))
    new_yaw_rate = math.radians(math.degrees(run_yaw))
    new_latv = run_latv

    spin_recovering = False

    # --- wrong-way recovery (C# only) ---
    # This vehicle model holds the car's FORWARD speed at whatever the speed ramp says while
    # lateral velocity is free (lateral accel = tire force / mass - forward speed x yaw rate), so
    # once the car is rotating faster than the tires can stop, nothing bleeds the energy off:
    # measured with "direct2" and no net at all, 13 of 256 three-minute runs had a spin-out the
    # car took 10-65 s to come back from, at up to 1,089 km/h. The trigger is the one thing a held
    # drift never does and a spin always does: the car POINTS more than WRONG_WAY_ENTRY_DEG away
    # from the racing line's direction (the largest value in the 3,322-tick manji3 server run was
    # 58 deg). Then yaw rate and lateral velocity decay until the car is neither sliding nor
    # rotating.
    if p[P_EXT_WRONG_WAY_RECOVERY] > 0.5:
        already = s[S_SPIN_RECOVERING] > 0.5
        if not already:
            wrong_way_deg = abs(wrap_pi(new_heading - corrected_angle)) * (180.0 / math.pi)
            if wrong_way_deg > p[P_WRONG_WAY_ENTRY_DEG]:
                streak = s[S_SPIN_STREAK] + 1
                s[S_SPIN_STREAK] = streak
                if streak >= p[P_WRONG_WAY_ENTRY_TICKS]:
                    s[S_SPIN_RECOVERING] = 1.0
                    already = True
            else:
                s[S_SPIN_STREAK] = 0.0
        if already:
            decay = math.exp(-dt_s / p[P_SPIN_RECOVERY_TIME_CONSTANT_S])
            new_yaw_rate *= decay
            new_latv *= decay
            spin_recovering = True
            slide_deg = math.atan2(abs(new_latv), body_fwd) * (180.0 / math.pi)
            if (slide_deg < p[P_WRONG_WAY_EXIT_SLIP_DEG]
                    and abs(new_yaw_rate) * (180.0 / math.pi) < p[P_WRONG_WAY_EXIT_YAW_DEGS]):
                s[S_SPIN_RECOVERING] = 0.0
                s[S_SPIN_STREAK] = 0.0

    # --- position integration ---
    ch = math.cos(new_heading)
    sh = math.sin(new_heading)
    vx = body_fwd * ch - new_latv * sh
    vz = body_fwd * sh + new_latv * ch
    new_x = px + vx * dt_s
    new_z = pz + vz * dt_s

    # --- collision response (C# only): fresh pins around the travel direction, against the
    # walls the game collides with ---
    pushout_m = 0.0
    outline = p[P_EXT_CAR_OUTLINE_CONTACT] > 0.5
    ol_front = p[P_DIST_CG_FRONT_M] + p[P_CAR_OUTLINE_OVERHANG_M]
    ol_rear = p[P_DIST_CG_REAR_M] + p[P_CAR_OUTLINE_OVERHANG_M]
    ol_half = 0.5 * max(p[P_FRONT_TRACK_M], p[P_REAR_TRACK_M]) + p[P_CAR_OUTLINE_SIDE_MARGIN_M]
    if p[P_EXT_COLLISION_RESPONSE] > 0.5:
        base = math.atan2(vz, vx) if (vx * vx + vz * vz) > 0.01 else new_heading
        pox = 0.0
        poz = 0.0
        oy = py + p[P_PROXIMITY_PIN_HEIGHT_OFFSET_M]
        contact = p[P_COLLISION_CONTACT_DISTANCE_M]
        if outline:
            # CarOutlineContact: a ray from the centre to each outline point (the car's heading);
            # a wall nearer than the point is inside the car, pushed out along that ray
            for i in range(N_OUTLINE):
                fwd, left = _outline_point(i, ol_front, ol_rear, ol_half)
                reach = math.sqrt(fwd * fwd + left * left)
                ang = new_heading + math.atan2(left, fwd)
                dist, surf = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), reach,
                                     tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 2)
                if dist < 0.0 or surf != SURF_WALL or dist >= reach:
                    continue
                pen = (reach + p[P_COLLISION_PUSHOUT_MARGIN_M]) - dist
                pox += -math.cos(ang) * pen
                poz += -math.sin(ang) * pen
        for i in range(N_PINS if not outline else 0):
            ang = base + _pin_offset(i)
            dist, surf = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), contact,
                                 tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 2)
            if dist < 0.0 or surf != SURF_WALL or dist >= contact:
                continue
            pen = (contact + p[P_COLLISION_PUSHOUT_MARGIN_M]) - dist
            pox += -math.cos(ang) * pen
            poz += -math.sin(ang) * pen
        if pox * pox + poz * poz > 0.0001:
            new_x += pox
            new_z += poz
            pushout_m = math.sqrt(pox * pox + poz * poz)
            nx_ = pox / pushout_m
            nz_ = poz / pushout_m
            along = vx * nx_ + vz * nz_
            if along < 0.0:
                vx -= nx_ * along
                vz -= nz_ * along

    if p[P_EXT_HEIGHT_MODEL] > 0.5:
        # The plugin's "VERTICAL FIX" block, in its own order: everything below is evaluated at
        # the NEW X/Z (after any push-out) but still the OLD Y, exactly as the C# reads `position`
        # before it overwrites position.Y.
        vp = world_to_spline(new_x, py, new_z, sp_pos)
        ha, hb, ht = find_interpolation_segment(new_x, py, new_z, vp, sp_pos, sp_next, sp_prev)
        spline_h = sp_pos[ha, 1] + (sp_pos[hb, 1] - sp_pos[ha, 1]) * ht
        raw_h = spline_h
        mesh_d, mesh_y = nearest_road_mesh_point(new_x, py, new_z, p[P_MAX_TRUSTED_MESH_DISTANCE_M],
                                                 rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items)
        if mesh_d >= 0.0:
            blend = max(0.0, min(1.0, (abs(cte) - p[P_CROSS_TRACK_BLEND_START_M])
                                 / (p[P_CROSS_TRACK_BLEND_END_M] - p[P_CROSS_TRACK_BLEND_START_M])))
            raw_h = spline_h + blend * ((mesh_y + p[P_ROAD_MESH_HEIGHT_OFFSET_M]) - spline_h)
        smoothing = p[P_HEIGHT_SMOOTHING_FACTOR]
        if p[P_EXT_GROUND_RAY_HEIGHT] > 0.5:
            # The plugin's GROUND RAY HEIGHT block: the ground below the old Y (ground_below). No
            # ground found leaves the spline/mesh blend above, and its smoothing factor, as the
            # fallback.
            ray_y = py + p[P_GROUND_RAY_START_ABOVE_M]
            gdist, gsurf = ground_below(new_x, ray_y, new_z, p[P_GROUND_RAY_MAX_DISTANCE_M], tri_v, tri_surf,
                                        gmin, cell_size, dims, cell_start, cell_items)
            if gdist >= 0.0:
                raw_h = (ray_y - gdist) + p[P_GROUND_RAY_HEIGHT_OFFSET_M]
                smoothing = p[P_GROUND_RAY_SMOOTHING_FACTOR]
        if math.isnan(s[S_SMOOTH_H]):
            new_y = raw_h
        else:
            h_gap = raw_h - s[S_SMOOTH_H]
            if abs(h_gap) > p[P_HEIGHT_SNAP_THRESHOLD_M]:
                new_y = raw_h
            else:
                new_y = s[S_SMOOTH_H] + smoothing * h_gap
        s[S_SMOOTH_H] = new_y
    else:
        new_y = sp_pos[a, 1] + (sp_pos[b, 1] - sp_pos[a, 1]) * t

    # --- cast THIS tick's fan + pins from the new pose ---
    fan_center = math.atan2(vz, vx) if (vx * vx + vz * vz) > 0.01 else new_heading
    fan_rad = p[P_RAYCAST_FAN_ANGLE_DEGREES] * (math.pi / 180.0)
    start_angle = fan_center - fan_rad / 2.0
    ray_step = fan_rad / (N_RAYS - 1)
    fan_max = p[P_RAYCAST_FAN_MAX_DISTANCE_M]
    oy = new_y + p[P_RAYCAST_HEIGHT_OFFSET_M]
    for i in range(N_RAYS):
        ang = start_angle + ray_step * i
        dist, surf = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), fan_max,
                             tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 0)
        s[S_RAY0 + i] = fan_max if dist < 0.0 else dist

    oy = new_y + p[P_PROXIMITY_PIN_HEIGHT_OFFSET_M]
    pin_max = p[P_PROXIMITY_PIN_MAX_DISTANCE_M]
    min_wall = pin_max
    for i in range(N_PINS):
        ang = new_heading + _pin_offset(i)
        dist, surf = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), pin_max,
                             tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 0)
        if dist < 0.0:
            s[S_PIN0 + i] = pin_max
            s[S_PINSURF0 + i] = SURF_NONE
        else:
            s[S_PIN0 + i] = dist
            s[S_PINSURF0 + i] = surf
        # the reward's nearest wall: the same pin, against the walls the game collides with
        wd, ws = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), WALL_PROXIMITY_RANGE_M,
                         tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 2)
        if wd >= 0.0 and ws == SURF_WALL and wd < min_wall:
            min_wall = wd
    if outline:
        # CarOutlineContact: the gap between the outline and a wall, plus the half width, so a
        # wall beside the car reads about as before and a wall near the nose or tail now counts
        for i in range(N_OUTLINE):
            fwd, left = _outline_point(i, ol_front, ol_rear, ol_half)
            reach = math.sqrt(fwd * fwd + left * left)
            ang = new_heading + math.atan2(left, fwd)
            wd, ws = raycast(new_x, oy, new_z, math.cos(ang), 0.0, math.sin(ang), reach + WALL_PROXIMITY_RANGE_M,
                             tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items, 2)
            if wd >= 0.0 and ws == SURF_WALL:
                gap = max(0.0, wd - reach) + ol_half
                if gap < min_wall:
                    min_wall = gap

    # --- reward signals (not part of the plugin's control law) ---
    # progress: arc-length of the segment projection, unwrapped across the lap seam.
    na = world_to_spline(new_x, new_y, new_z, sp_pos)
    a2, b2, t2 = find_interpolation_segment(new_x, new_y, new_z, na, sp_pos, sp_next, sp_prev)
    arc_new = sp_cum[a2] + t2 * sp_length[a2]
    lap = sp_cum[-1] + sp_length[-1]
    arc_prev = s[S_ARC]
    progress = 0.0
    if not math.isnan(arc_prev):
        progress = arc_new - arc_prev
        if progress > lap / 2:
            progress -= lap
        elif progress < -lap / 2:
            progress += lap
    s[S_ARC] = arc_new
    # the ground directly under the car (from 3 m above, walls stepped through, cracks bridged).
    surface_under = surface_below(new_x, new_y + 3.0, new_z, 12.0, tri_v, tri_surf, gmin, cell_size, dims,
                                  cell_start, cell_items)

    # --- write state back ---
    s[S_X], s[S_Y], s[S_Z] = new_x, new_y, new_z
    s[S_HEADING] = new_heading
    s[S_YAW] = new_yaw_rate
    s[S_LATV] = new_latv
    s[S_OWN_SPEED] = new_own
    s[S_CORR_ANGLE] = corrected_angle
    s[S_FAN_CENTER] = fan_center
    s[S_LAST_STEER] = steer_cmd
    s[S_LAST_FSAT] = front_sat
    s[S_LAST_RSAT] = rear_sat

    diag[D_SpeedKmh] = speed_total_ms * 3.6
    diag[D_BodyForwardSpeedMs] = body_fwd
    diag[D_RawSplineAngle] = corrected_angle
    diag[D_GlitchCorrected] = 1.0 if glitch else 0.0
    diag[D_CrossTrackErrorM] = cte
    diag[D_SteerCommandDeg] = math.degrees(steer_cmd)
    diag[D_FrontSlipAngleDeg] = math.degrees(front_slip)
    diag[D_RearSlipAngleDeg] = math.degrees(rear_slip)
    diag[D_FrontSaturation] = front_sat
    diag[D_RearSaturation] = rear_sat
    diag[D_SteerSaturated] = 1.0 if abs(abs(steer_cmd) - max_steer) < 1e-6 else 0.0
    diag[D_NearestPointId] = nearest
    diag[D_Radius] = radius
    diag[D_RealizedLongitudinalAccelMs2] = realized_long_acc
    diag[D_DesiredSpeedMs] = desired_speed
    diag[D_NearestPointGlitchCorrected] = 1.0 if np_glitch else 0.0
    diag[D_UncontrolledSpinRecovering] = 1.0 if spin_recovering else 0.0
    diag[D_CollisionPushOutM] = pushout_m
    diag[D_ProgressM] = progress
    diag[D_SurfaceUnderCar] = surface_under
    diag[D_MinWallDistanceM] = min_wall
    diag[D_Vx] = vx
    diag[D_Vz] = vz
    # heading minus direction of travel (the C#'s own realSlipAngle definition)
    vel_dir = math.atan2(vz, vx) if (vx * vx + vz * vz) > 0.0001 else new_heading
    diag[D_RealSlipAngleRad] = wrap_pi(new_heading - vel_dir)


def initial_state(position, heading_rad, max_speed_ms):
    s = np.zeros(STATE_SIZE)
    s[S_X], s[S_Y], s[S_Z] = position
    s[S_HEADING] = heading_rad
    s[S_CORR_ANGLE] = np.nan
    s[S_MAX_SPEED] = max_speed_ms
    s[S_PREV_NEAREST] = -1
    s[S_ARC] = np.nan
    s[S_SMOOTH_H] = np.nan
    s[S_REAR_SLIDE_GRIP] = 1.0
    s[S_RAY0:S_RAY0 + N_RAYS] = 60.0
    s[S_PIN0:S_PIN0 + N_PINS] = 12.0
    s[S_PINSURF0:S_PINSURF0 + N_PINS] = SURF_NONE
    return s
