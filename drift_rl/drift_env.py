"""
drift_env.py - the reinforcement-learning environment around fast_sim.py.

WHAT THE POLICY CONTROLS (three numbers, evaluated once per server tick):
    action[0]  the steering angle, as a fraction of full lock (the policy file's MaxSteerAngleDeg)
    action[1]  the target speed, as a fraction of the car's MaxSpeed (-1 stop, +1 MaxSpeed), or,
               with EnableRearFrictionCircle, a pedal: thrust above zero, brakes below
    action[2]  the brake WITH the throttle, [0, 1]: left-foot braking on the front
               axle (fast_sim.step_post); EnableRearFrictionCircle only. Older policies have two.
    The wrong-way recovery, wall push-out and the corrections to the nearest spline point keep
    running underneath, as in the plugin.

WHAT IT SEES (54 numbers; older policies read the first 46 or 52 - see OBS_NAMES and build_obs; each is a value the plugin has in hand
where its policy hook sits).

WHAT IT IS REWARDED FOR: see RewardConfig - progress along the spline paid more while the car
is sideways, the drift line (outside zones), linked drifts, and costs for walls, leaving the
course, the recovery net, a sawing wheel, front tires off the car's path and a tail on the wrong
side of a long corner.

DriftVecEnv steps N cars together in one compiled loop; train_ppo_gpu.py uses it (or gpu_env.py,
the same code compiled for the GPU).
"""
import math
import os
from dataclasses import dataclass, asdict, replace, field

import numpy as np
from numba import prange

import sim_target

# CPU normally. When this file is loaded as the GPU variant (sim_target.load_variant), `njit`
# compiles CUDA device functions and `fs` is the GPU variant of fast_sim - same source, see
# sim_target.py.
_TARGET, njit = sim_target.get_njit(globals())
fs = sim_target.load_variant('fast_sim', _TARGET)
if _TARGET == 'cpu':
    # numba's on-disk cache only notices edits to THIS file; the compiled loops below also
    # contain fast_sim's code, so throw the cache away whenever fast_sim.py is newer than it.
    sim_target.drop_stale_cache(__file__, [fs.__file__, sim_target.__file__])
from fast_sim import (S_X, S_Y, S_Z, S_HEADING, S_YAW, S_LATV, S_OWN_SPEED, S_MAX_SPEED, S_RAY0,
                      S_FAN_CENTER, S_PIN0, S_SPIN_RECOVERING, S_LAST_STEER, S_LAST_FSAT,
                      S_LAST_RSAT, S_CORR_ANGLE, S_PREV_NEAREST, S_ARC, S_PINSURF0,
                      M_NEAREST, M_CORR_ANGLE, M_CTE, M_FIRST, M_DT, M_RADIUS, N_RAYS, N_PINS, STATE_SIZE, MID_SIZE, DIAG_SIZE,
                      D_ProgressM, D_RealSlipAngleRad, D_CollisionPushOutM, D_MinWallDistanceM,
                      D_SurfaceUnderCar, D_UncontrolledSpinRecovering, D_SpeedKmh,
                      D_CrossTrackErrorM, D_SteerSaturated, SURF_ROAD, SURF_NONE, SURF_WALL)

LOOKAHEAD_M = (5.0, 15.0, 30.0, 50.0, 80.0)
# Room to the course's edges (build_course.py: tire stacks, walls or the paving's edge), either side,
# at the car and at two of the look-ahead points: where the road ends, which no sensor sees on a
# grass-edged road (the fan rays are level and pass over grass). The car's own offset from the
# line is carried ahead unchanged. 0 m is the car's nearest point; the others are LOOKAHEAD_M[1:3].
EDGE_AHEAD_M = (0.0, 15.0, 30.0)
EDGE_ROOM_MAX_M = 20.0
N_LOOK = len(LOOKAHEAD_M)
MIN_RADIUS_WINDOW_M = 40.0

OBS_NAMES = (
    ['own_speed/25', 'lateral_velocity/10', 'yaw_rate/2', 'body_slip/(pi/2)', 'angle_diff/pi',
     'cross_track_error/15', '20/radius']
    + [f'spline_turn_ahead_{int(d)}m/pi' for d in LOOKAHEAD_M]
    + ['max_speed/25']
    + [f'fan_ray_{i}/60' for i in range(N_RAYS)]
    + ['fan_center_minus_heading/pi']
    + [f'pin_{i}/12' for i in range(N_PINS)]
    + ['gap_correction/0.5', 'pin_correction/0.35', 'prev_action_steer', 'prev_action_speed',
       'dt/0.1', 'spin_recovering', 'last_steer/max_steer', 'last_front_saturation',
       'last_rear_saturation', '20/min_radius_next_40m', 'first_tick']
    + [f'course_room_{side}_{int(d)}m/10' for d in EDGE_AHEAD_M for side in ('pos', 'neg')]
    + ['pace_fraction', 'prev_action_brake', 'pace_over']
    + ['car_mass', 'car_inertia', 'car_wheelbase', 'car_front_weight', 'car_steer_lock', 'car_rear_grip',
       'car_grip_balance', 'car_rear_peak_slip', 'car_power_50kmh', 'car_power_90kmh', 'car_cg_height']
    + ['skill']
)
OBS_SIZE_V1 = 46                 # policies from before the course-room inputs read only the first 46
OBS_SIZE_V2 = 52                 # ... and from before the pace and brake inputs, the first 52
OBS_SIZE_V3 = 54                 # ... and from before the pace error input, the first 54
OBS_SIZE_V4 = 55                 # ... and from before the car inputs, the first 55
OBS_SIZE = len(OBS_NAMES)
# The car (car_inputs()): what tells one car from another, each scaled to sit near 0 for a typical
# drift car, from the same numbers the vehicle model drives with - so the plugin works them out
# the same way from the car's files.
CAR_OBS_SLOT = OBS_SIZE_V4
N_CAR_OBS = 11
SKILL_OBS_SLOT = CAR_OBS_SLOT + N_CAR_OBS
# The skill input (fast_sim P_SKILL, 0 - 1, drawn per episode like the pace): how good a driver to
# be. From what decides drift-day run groups and what coaches and judges pick out (README, "A
# skill setting"), the range runs from a driver who is allowed out with others - holds a drift,
# rarely spins - up to the best this policy can do. Below 1, by the share d = 1 - skill:
#  * its hands are worse: smooth noise on the steering and pedal (SKILL_STEER_NOISE and
#    SKILL_PEDAL_NOISE x d) and a slower reaction (the action applied is the one asked
#    SKILL_DELAY_TICKS x d ticks earlier), so it is late into slides and over-corrects;
#  * it is asked for less: the angle that counts as full drift falls by SKILL_ANGLE x d, linking
#    pays SKILL_CHAIN x d less (straightening up between corners is fine), the outside zones pay
#    nothing at skill 0 (a mid-track line), and busy steering costs SKILL_STEER x d less.
#  * not relaxed: leaving the course, walls, markers, spins and weaving cost the same at every
#    level - a bot that loses it among people would not be let out with them.
SKILL_STEER_NOISE = 0.10
SKILL_PEDAL_NOISE = 0.20
SKILL_NOISE_KEEP = 0.9          # per tick: the noise wanders, it does not jitter
SKILL_DELAY_TICKS = 1.5         # x d, rounded: 1 tick (about 55 ms) below skill 0.67, 2 at skill 0
SKILL_ANGLE = 0.3
SKILL_CHAIN = 0.7
SKILL_STEER = 0.6
# The pace this car is asked to drive (fast_sim.S_PACE), as a fraction of what the course allows:
# of its reference speed profile (course_pace_reference(): pace_model's speed at each point, a
# little faster than a quick human), so "0.9" means the same on a tight course as on a fast one and the plugin needs
# no number per track. gen4 read km/h / 50 here instead ("pace_kmh_range").
PACE_OBS_SLOT = OBS_SIZE_V2
# How much faster than asked the car has been driving lately: its speed along the line over the
# speed asked of it (pace x the reference speed at each point it passed), both averaged over about
# RewardConfig.pace_average_s, minus 1; 0 with no pace. The reward's measure, shown to the policy:
# without it the network sees "0.85" and its own speed but not what 0.85 is here.
PACE_OVER_SLOT = OBS_SIZE_V3
PACE_OVER_CLIP = 1.0
# Where a new input's normalisation starts when a policy without it is widened to read it
# (train_ppo_gpu.py; frozen for the run). Others start at mean 0.6, std 0.5.
NEW_INPUT_NORM = {'pace_fraction': (0.88, 0.1), 'pace/50': (1.0, 0.12), 'prev_action_brake': (0.0, 0.3),
                  'pace_over': (0.0, 0.15),
                  **{name: (0.0, 0.3) for name in OBS_NAMES[OBS_SIZE_V4:SKILL_OBS_SLOT]},
                  'skill': (0.6, 0.3)}
# steer, speed (pedal), and brake WITH the throttle (fast_sim.step_post): the third is used as
# max(0, output), so an output at or below zero is no brake at all
ACT_SIZE = 3

# per-episode statistics columns
(E_RETURN, E_LENGTH, E_PROGRESS, E_ABS_SLIP_DEG, E_DRIFT_TICKS, E_PUSHOUT_TICKS, E_PUSHOUT_M,
 E_OFFROAD_TICKS, E_SPIN_TICKS, E_SPEED_KMH, E_ABS_CTE, E_NEARWALL_TICKS, E_STEER_SAT_TICKS,
 E_TIME_S, E_TERMINATED, E_REASON, E_OFFCOURSE_TICKS, E_MARKER_HITS, E_STRAIGHT_TICKS,
 E_STRAIGHT_DRIFT_TICKS, E_STRAIGHT_TRANSITIONS, E_STRAIGHT_M, E_CHAIN_S, E_SLIDE_TICKS,
 E_HIGH_ANGLE_TICKS, E_ZONE_TICKS, E_ZONE_FILLED_TICKS, E_STEER_REVERSALS, E_FRONT_SLIDING_TICKS,
 E_HELD_TICKS, E_WRONG_SIDE_TICKS, E_PACE_TARGET, E_PACE_ERROR, E_BRAKE_TICKS, E_BRAKE_SUM,
 E_PACE_ABS_ERROR, E_WEAVES, E_SWINGS) = range(38)
EP_SIZE = 38
# a tick counts as braking with the brake action (brake_fraction) from this much of it
BRAKE_STAT_MIN = 0.05
# Slip-angle bands of the statistics, degrees. "drift" (20-50) is the band every table up to
# slide2 reports and stays as it was. A car with the real steering lock can hold more than 50,
# which that band would count as not drifting, so two more: any slide short of a spin, and the
# part of it at a high angle.
SLIDE_MIN_DEG, HIGH_ANGLE_MIN_DEG, SLIDE_MAX_DEG = 20.0, 40.0, 80.0
# per-car counters columns: [ticks this episode, off-surface streak, no-progress streak,
# 1 + index of the course marker the body was touching last tick (0 = none),
# seconds the current drift chain has lasted, seconds since the car was last sliding,
# which way it was last sliding (+1 / -1, 0 = not yet), last tick's change of the steering action,
# the car's speed along the line averaged over the last pace_average_s seconds (m/s), and the
# speed its pace asked for (pace x the course's reference speed where the car was), averaged the
# same way (m/s), seconds since its last transition, and which way the course was turning there
# (fast_sim co[:, 9]; 2 = no transition yet)]
N_COUNTERS = 23       # ... [12, 13] the steering and pedal noise of a lower skill (SKILL_*), [14 - 19] its last two asked actions,
# [20 - 22] the way the car last rotated at swing_rate_degs or more (+1 / -1, 0 = not yet), seconds since, and which
# way the course was turning there
# a steering reversal for the statistics: the steering action turning back by more than this
# (fraction of full lock) in one tick
STEER_REVERSAL_MIN = 0.05
# "a straight" for the statistics: the long straights fast_sim.World marks in co[:, 6] (radius
# above 150 m for 30 m or more - on Spirit Peaks, for example, the 92 m top straight, points 488-547). Without
# a course file, any spline point with a recorded radius above this.
STRAIGHT_RADIUS_M = 150.0

# The car's body as a rectangle around its centre, for the marker-contact test. The plugin's car
# has DistanceCgToFrontAxleM 1.035 and DistanceCgToRearAxleM 1.265 (2.30 m wheelbase); the
# overhangs and the width are NOT in the plugin's configuration and are assumed here for a car
# of that wheelbase (about 3.9 m long, 1.7 m wide).
CAR_FRONT_M = 1.80
CAR_REAR_M = 2.10
CAR_HALF_WIDTH_M = 0.85
# why an episode ended early (E_REASON)
REASONS = {0: 'time limit (completed)', 1: 'too far from the line', 2: 'spun round (travelling backwards)',
           3: 'off the track surface too long', 4: 'stuck (no progress)', 5: 'left the mesh'}
EP_NAMES = ['return', 'length', 'progress_m', 'mean_abs_slip_deg', 'drift_fraction',
            'pushout_tick_fraction', 'pushout_m_per_100m', 'offroad_fraction', 'spin_recovery_fraction',
            'mean_speed_kmh', 'mean_abs_cte_m', 'near_wall_fraction', 'steer_saturated_fraction',
            'sim_seconds', 'terminated_early', 'end_reason', 'offcourse_fraction', 'marker_hits_per_min',
            'straight_drift_fraction', 'straight_transitions_per_100m', 'mean_chain_s',
            'slide_fraction', 'high_angle_fraction', 'zone_fill_fraction', 'steer_reversals_per_s',
            'front_sliding_fraction', 'wrong_side_fraction', 'pace', 'pace_error_pct', 'pace_abs_error_pct',
            'brake_fraction', 'mean_brake', 'weaves_per_min', 'swings_per_min']


@dataclass
class RewardConfig:
    progress_weight: float = 0.5          # reward per metre of spline progress with no slide at all...
    grip_progress_fraction: float = 0.25  # ...is this fraction of the reward per metre while drifting
    slip_ramp_start_deg: float = 10.0     # drift quality 0 below this,
    slip_full_start_deg: float = 25.0     # 1 from here...
    slip_full_end_deg: float = 45.0       # ...to here (MaxSlipAngleRadians is 45 deg),
    slip_ramp_end_deg: float = 70.0       # back to 0 here (past this it is a spin, not a drift)
    pushout_penalty_per_m: float = 2.0    # wall push-out, per metre pushed (capped at 3 m/tick)
    pushout_penalty_flat: float = 0.5     # plus this per tick of contact
    wall_proximity_penalty: float = 0.3   # scaled 0..1 as the nearest game wall (along a pin) goes 2.5 m -> 0 m
    wall_proximity_distance_m: float = 2.5
    offroad_penalty: float = 0.3          # per tick with anything but the track surface under the car
    spin_recovery_penalty: float = 0.5    # per tick the wrong-way recovery net is active
    action_rate_penalty: float = 0.05     # * sum((a - a_prev)^2)
    terminal_penalty: float = 10.0        # on early termination
    max_abs_cte_m: float = 35.0           # terminate: this far from the spline
    max_abs_slip_deg: float = 120.0       # terminate: travelling backwards
    offroad_ticks_limit: float = 60.0     # terminate: this many consecutive ticks off the tarmac
    stall_ticks_limit: float = 60.0       # terminate: this many consecutive ticks with < 0.1 m progress
    # Which collision surface counts as "the track". -1 = work it out: whatever
    # surface lies under most of the spline's own points (Road: what the track's surfaces.ini
    # marks IS_VALID_TRACK). Anything else under the car counts as off the track.
    track_surface_code: float = -1.0
    # THE COURSE: the paved surface is wider than the course, which is the lane
    # between the tire stacks, cones and barrels (build_course.py). "Off the course" = the car's
    # centre less than course_margin_m inside the course edge (so the body, not just the centre,
    # stays between the markers). Charged per tick, plus per metre beyond that.
    offcourse_penalty: float = 0.3
    offcourse_penalty_per_m: float = 0.15
    course_margin_m: float = 1.0
    # each time the car's body starts overlapping a marker. About what 10 ticks of good drifting
    # earn, so clipping a stack is never worth the line it buys.
    marker_hit_penalty: float = 5.0
    # LINKED DRIFT ("manji the straight"): everything above pays for being sideways
    # one tick at a time, so on a straight the best move was to straighten up and wait for the
    # next corner (the 92 m top straight: 2-3 deg mean slip, 3-5% of ticks at 20 deg or more).
    # This pays for KEEPING a drift going. A chain is alive while |slip| is at least
    # chain_slip_deg, and survives dips below it shorter than chain_grace_s - the car has to pass
    # through zero slip to change direction, and at the 50-60 deg/s the car yaws in a transition
    # going from +15 to -15 deg takes about half a second. The bonus is per metre of progress
    # (so parking sideways earns nothing), ramps up over the chain's first chain_full_s seconds,
    # and is paid during the dips too, so a transition is not a hole in the reward. Letting a
    # chain die costs the whole ramp again. 0 switches it off.
    chain_bonus_per_m: float = 0.25
    chain_slip_deg: float = 15.0
    chain_grace_s: float = 0.7
    chain_full_s: float = 4.0
    # SPEED IS NOT THE POINT: everything above that pays, pays per metre, so a faster
    # lap earned more and the policy drove for one. With this set, no tick is paid for more
    # distance than this speed covers: above it the pay is by time, below it still by distance
    # (so parking sideways earns nothing). 0 = off.
    paid_speed_kmh: float = 0.0
    # THE DRIFT LINE: outside zones along the outer edge of each corner
    # (build_zones.py), filled the way Formula Drift scores them - by the rear of the car, while
    # sliding. Paid per metre like the rest, times the drift quality, times how well the zone is
    # filled: 1 with the rear axle within zone_depth_m inside the edge, falling to 0 over the
    # next zone_reach_m (so there is a slope to climb from the racing line), 0 past the edge
    # (No-Go; the off-course penalty is already there). 0 = off.
    zone_bonus_per_m: float = 0.0
    zone_depth_m: float = 3.0
    zone_reach_m: float = 12.0
    # each time the steering action turns back on itself, times how far (fraction of lock): the
    # smaller of the two opposite moves. A flip is one reversal; sawing at the wheel is many.
    steer_reversal_penalty: float = 0.0
    # FRONT TIRES THAT FOLLOW THE CAR: in a flip the policy threw the front wheels
    # to full lock, 14 deg past where the front axle was travelling (the tire's peak is at 11.7),
    # and let the scrubbing front tires spin the car; a human's front tires stay within 2 - 5 deg
    # of the front axle's path all the way through. Charged per tick, per degree of front slip
    # angle beyond front_slip_free_deg. 0 = off.
    front_slip_penalty_per_deg: float = 0.0
    front_slip_free_deg: float = 8.0
    # THE TAIL ON THE OUTSIDE: pedal3 swung the tail across to the inside twice in
    # the middle of the 209 deg corner, braking each time; a human holds one drift through it.
    # Charged per second of not sliding (chain_slip_deg or more) with the tail toward the outside,
    # over the stretch of a long corner where it has to stay out (fast_sim co[:, 8]); straightening
    # up mid-corner counts too, or the policy only swaps the flip for one. 0 = off.
    wrong_side_penalty_per_s: float = 0.0
    # ONE DRIFT THROUGH A LONG CORNER: pedal4 and pedal5 still flick the tail across
    # once in the 209 deg corner and back, paying ~0.5 s of the cost above for it. The human eases
    # the angle to 20 - 30 deg there instead and never changes side. Charged once per side change
    # (a transition: sliding chain_slip_deg+ the other way, the chain still alive) inside a long
    # corner's held stretch, so going across and back costs it twice; easing off is free. 0 = off.
    side_change_penalty: float = 0.0
    # PACE: the plugin spaces its bots round the lap by asking each for a pace instead
    # of changing its grip, which changed how it drifted. A car with a pace (fast_sim.S_PACE, a
    # fraction drawn per episode from EnvConfig.pace_low/high) is asked for pace x the course's
    # reference speed at each point (DriftVecEnv pace_ref). It is paid for no more distance than
    # that speed covers (as paid_speed_kmh, which it takes the place of), and while its speed,
    # averaged over the last pace_average_s seconds, is off the asked speed averaged the same way,
    # everything it is paid for driving (progress, linked drift, zones) shrinks, down to
    # pace_min_pay of it at pace_over_full_cut too fast or pace_under_full_cut too slow (shares:
    # 0.1 = 10%) - so it has to slow down, not just stop earning more, and it is not paid to
    # crawl either. Too slow used to cost only the distance not covered, and gen4 (277M
    # ticks) settled on braking a good deal at every pace: asked 40 -> 60 km/h it drove 49 -> 51
    # on Spirit Peaks. A cut in pay with a floor, not a charge: anything
    # that left a tick worth less than nothing made ending the episode the best move (trials:
    # a charge, and a cut to zero, both had the training cars ending early 40% ->
    # 60-70% within minutes). Asking for pace x the speed at each point, not one speed for the
    # lap, keeps a hairpin from being judged by the straight's speed (gen4 asked 40 - 60 km/h
    # everywhere; flat out the courses run 42 - 73 km/h).
    pace_over_full_cut: float = 0.1
    pace_min_pay: float = 0.4
    pace_average_s: float = 1.0
    pace_under_full_cut: float = 0.2
    # WEAVING: gen5, asked to drive slower than its brake can take it, flicks from one
    # side to the other to bleed speed - 25 transitions a minute on Spirit Peaks at pace 0.8, 10 at
    # 1.0, the human 14.6 (counted the same way) - and the owner saw the rotation as impossible.
    # Charged once per transition the course did not ask for: one into a slide against the way the
    # course turns there (fast_sim co[:, 9]), or one within weave_gap_s of the last with the
    # course turning the same way (or straight) at both. On a human's laps that charges
    # 0.2 a minute, gen3 0.9, gen5 4.4 at pace 1.0 and 11.3 at 0.8. 0 = off.
    weave_penalty: float = 0.0
    weave_gap_s: float = 1.5
    # SWINGS: gen10 at pace 0.85 on Drift Playground (server test 25) reverses its rotation 15 times
    # a minute, twice the human's 7.2, and the weave penalty sees little of it: most are on one side,
    # the angle swung open and shut with the front brake to shed speed it is asked to lose. Charged
    # once per reversal of the yaw rate (swing_rate_degs one way, then the other within
    # swing_window_s: drive_report's direction changes) the course did not ask for - unless the
    # course turns at the end, the car now rotates the way it turns, and it turned differently at the
    # start. 0 = off.
    swing_penalty: float = 0.0
    swing_rate_degs: float = 60.0
    swing_window_s: float = 2.0
    # TYRE ENERGY (Djeumou et al., "Reference-Free Formula Drift with Reinforcement Learning",
    # Toyota Research Institute, 2024, https://arxiv.org/abs/2410.20990): pays the work the rear
    # tyres do sliding - their sideways force (a share of the axle's load) times the rear axle's
    # sideways speed over the car's speed - per metre of paid progress, instead of a slip-angle
    # band. An angle reward has to pass through zero at every transition, so it fights them; the
    # rear tyres' work does not care which way they slide. 0 = off. slip_band_pay scales the
    # slip-band share of the progress pay (1 = as always, 0 = the band is gone, only the grip
    # share is paid).
    tyre_energy_per_m: float = 0.0
    slip_band_pay: float = 1.0

    def vector(self):
        return np.array(list(asdict(self).values()), dtype=np.float64)


(R_PROGRESS_W, R_GRIP_FRAC, R_SLIP0, R_SLIP1, R_SLIP2, R_SLIP3, R_PUSH_M, R_PUSH_FLAT, R_WALL,
 R_WALL_DIST, R_OFFROAD, R_SPIN, R_ACT_RATE, R_TERMINAL, R_MAX_CTE, R_MAX_SLIP,
 R_OFFROAD_LIMIT, R_STALL_LIMIT, R_TRACK_SURF, R_OFFCOURSE, R_OFFCOURSE_M, R_COURSE_MARGIN,
 R_MARKER_HIT, R_CHAIN_BONUS, R_CHAIN_SLIP, R_CHAIN_GRACE, R_CHAIN_FULL, R_PAID_SPEED,
 R_ZONE_BONUS, R_ZONE_DEPTH, R_ZONE_REACH, R_STEER_REVERSAL, R_FRONT_SLIP_PEN, R_FRONT_SLIP_FREE,
 R_WRONG_SIDE, R_SIDE_CHANGE, R_PACE_OVER, R_PACE_MIN_PAY, R_PACE_AVG, R_PACE_UNDER, R_WEAVE,
 R_WEAVE_GAP, R_SWING, R_SWING_RATE, R_SWING_WINDOW, R_TYRE_ENERGY, R_SLIP_BAND_PAY) = range(47)


@dataclass
class EnvConfig:
    episode_ticks: int = 800                  # ~40 s of driving, roughly one 836 m lap
    # tick length: modelled on the server's measured RealDtMs (first live policy run,
    # 3046 ticks): 48-49 ms and 64-65 ms ticks in a "sLsLss" rhythm - a long tick followed 72.5%
    # of short ticks but only 1.1% of long ones (42.3% long overall) - plus 0.23% hitches at the
    # 200 ms MaxRealDtSeconds ceiling. randomise_dt=False fixes it at 0.049.
    randomise_dt: bool = True
    hitch_probability: float = 0.0025
    long_after_short_probability: float = 0.725
    long_after_long_probability: float = 0.011
    # spawn: a random point on the spline, nudged sideways and rotated a little
    spawn_lateral_m: float = 2.0
    spawn_heading_rad: float = 0.15
    # recovery starts: this share of episodes instead begins well off the line and pointing the
    # wrong way. The live server never resets a lost car, so the policy has to have seen being
    # lost; with only on-line starts (and episodes that end the moment the car is lost) it never
    # does. evaluate() always uses on-line starts, so its numbers stay comparable.
    recovery_start_fraction: float = 0.15
    recovery_lateral_m: float = 20.0
    recovery_heading_rad: float = 0.8
    # per-car MaxSpeed (CeilingKmh): real runs show 74.8 - 85.7 km/h
    max_speed_kmh_low: float = 72.0
    max_speed_kmh_high: float = 88.0
    # optional +-fraction on both axles' peak friction, per episode (0 = the plugin's own values)
    grip_randomisation: float = 0.0
    # Several cars (car_param_values() of each): every episode draws one, then varies it - its
    # weight by +-car_mass_randomisation (loads and yaw inertia together) and its engine's force by
    # +-car_power_randomisation - so a policy meets cars between and around the ones it was given.
    # Empty = the one car the plugin settings describe, as before.
    car_pool: tuple = ()
    # the skill input (SKILL_*), drawn per episode from this range; (1, 1) = full skill, as before
    skill_low: float = 1.0
    skill_high: float = 1.0
    car_mass_randomisation: float = 0.0
    car_power_randomisation: float = 0.0
    # the plugin's RearSlidingGrip (share of its peak sideways grip the rear axle keeps once it is
    # sliding), drawn per episode from this range and shown to the policy in slot 35. 1.0 - 1.0 is
    # the model every policy up to direct3 was trained in. A policy trained over a range can be
    # run at any value inside it, which makes the setting a speed dial: less grip = slower drift.
    rear_sliding_grip_low: float = 1.0
    rear_sliding_grip_high: float = 1.0
    # the pace each car is asked to drive (fast_sim.S_PACE), a fraction of the course's reference
    # speed, drawn per episode from this range and shown to the policy (PACE_OBS_SLOT). 0 - 0 =
    # no pace. Needs the environment's pace_ref (DriftVecEnv / course_pace_reference()).
    pace_low: float = 0.0
    pace_high: float = 0.0
    # ... and drawn again every this many seconds on average during an episode (0 = once per
    # episode): the plugin moves a car's pace as it spaces the bots, so the policy has to follow
    # a change, not only hold one
    pace_change_s: float = 15.0
    # Plugin settings (the names in GradientDriftConfiguration.cs / extra_cfg.yml) this car is
    # simulated with where they differ from the plugin's defaults - see PLUGIN_SETTING_PARAMS.
    # A policy file carries the ones it was trained with ("plugin_settings") and the plugin
    # applies them, so a policy cannot be run with a different steering lock than it learned.
    plugin_settings: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# compiled kernels
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def marker_contact(x, z, heading, mk):
    """Index of the first course marker the car's body overlaps, or -1. mk rows: [x, z, radius]."""
    c = math.cos(heading)
    sn = math.sin(heading)
    for j in range(mk.shape[0]):
        r = mk[j, 2]
        if r <= 0.0:
            continue
        dx = mk[j, 0] - x
        dz = mk[j, 1] - z
        if dx * dx + dz * dz > 9.0:        # 3 m: further than any body corner plus a marker radius
            continue
        fwd = dx * c + dz * sn
        lat = dz * c - dx * sn
        if -CAR_REAR_M - r < fwd < CAR_FRONT_M + r and abs(lat) < CAR_HALF_WIDTH_M + r:
            return j
    return -1


@njit(cache=True)
def course_overrun_m(cte, nearest, margin, co, n_points):
    """How far (m) the car's centre is past the course edge less `margin`; 0 = inside.
    co rows: [positive-side half width, negative-side half width, ...] per spline point; a co
    without one row per spline point means no course is known and this is always 0."""
    if co.shape[0] != n_points or nearest < 0 or nearest >= n_points:
        return 0.0
    over = cte - (co[nearest, 0] - margin)
    other = -cte - (co[nearest, 1] - margin)
    if other > over:
        over = other
    return over if over > 0.0 else 0.0


@njit(cache=True)
def _drift_quality(abs_slip_deg, rw):
    if abs_slip_deg <= rw[R_SLIP0]:
        return 0.0
    if abs_slip_deg < rw[R_SLIP1]:
        return (abs_slip_deg - rw[R_SLIP0]) / (rw[R_SLIP1] - rw[R_SLIP0])
    if abs_slip_deg <= rw[R_SLIP2]:
        return 1.0
    if abs_slip_deg < rw[R_SLIP3]:
        return 1.0 - (abs_slip_deg - rw[R_SLIP2]) / (rw[R_SLIP3] - rw[R_SLIP2])
    return 0.0


@njit(cache=True)
def build_obs(s, mid, prev_act, p, obs, sp_dir, la_idx, la_min_radius, co, pace_over):
    """The inputs (OBS_NAMES; older policies read the first 46 or 52). Slot 35 is the share of rear grip lost when sliding,
    (1 - RearSlidingGrip) / 0.5, and slot 36 reads 0: both carried the hand-written controller's
    steering corrections until it was retired, and the network shape stayed."""
    own = s[S_OWN_SPEED]
    nearest = int(mid[M_NEAREST])
    first = mid[M_FIRST] > 0.5
    obs[0] = own / 25.0
    obs[1] = s[S_LATV] / 10.0
    obs[2] = s[S_YAW] / 2.0
    obs[3] = math.atan2(s[S_LATV], max(own, 3.0)) / (math.pi / 2)
    obs[4] = fs.wrap_pi(mid[M_CORR_ANGLE] - s[S_HEADING]) / math.pi
    obs[5] = max(-3.0, min(3.0, mid[M_CTE] / 15.0))
    obs[6] = min(2.0, 20.0 / max(mid[M_RADIUS], 1.0))
    for j in range(N_LOOK):
        obs[7 + j] = fs.wrap_pi(sp_dir[la_idx[nearest, j]] - mid[M_CORR_ANGLE]) / math.pi
    k = 7 + N_LOOK
    obs[k] = s[S_MAX_SPEED] / 25.0
    k += 1
    for i in range(N_RAYS):
        obs[k + i] = s[S_RAY0 + i] / 60.0
    k += N_RAYS
    obs[k] = 0.0 if first else fs.wrap_pi(s[S_FAN_CENTER] - s[S_HEADING]) / math.pi
    k += 1
    for i in range(N_PINS):
        obs[k + i] = s[S_PIN0 + i] / 12.0
    k += N_PINS
    obs[k] = (1.0 - fs.rear_sliding_grip(s)) / 0.5
    obs[k + 1] = 0.0
    obs[k + 2] = prev_act[0]
    obs[k + 3] = prev_act[1]
    obs[k + 4] = mid[M_DT] / 0.1
    obs[k + 5] = s[S_SPIN_RECOVERING]
    obs[k + 6] = s[S_LAST_STEER] / p[fs.P_MAX_STEER_RAD]
    obs[k + 7] = s[S_LAST_FSAT]
    obs[k + 8] = s[S_LAST_RSAT]
    obs[k + 9] = min(2.0, 20.0 / max(la_min_radius[nearest], 1.0))
    obs[k + 10] = 1.0 if first else 0.0
    k += 11
    cte = mid[M_CTE]
    has_course = co.shape[0] == la_idx.shape[0]
    for j in range(3):
        point = nearest if j == 0 else la_idx[nearest, j]
        pos_room = EDGE_ROOM_MAX_M
        neg_room = EDGE_ROOM_MAX_M
        if has_course:
            pos_room = max(-10.0, min(EDGE_ROOM_MAX_M, co[point, 0] - cte))
            neg_room = max(-10.0, min(EDGE_ROOM_MAX_M, co[point, 1] + cte))
        obs[k + 2 * j] = pos_room / 10.0
        obs[k + 2 * j + 1] = neg_room / 10.0
    k += 6
    obs[k] = s[fs.S_PACE]
    obs[k + 1] = prev_act[2]
    obs[k + 2] = max(-PACE_OVER_CLIP, min(PACE_OVER_CLIP, pace_over))
    car_inputs(p, obs, k + 3)
    obs[k + 3 + N_CAR_OBS] = p[fs.P_SKILL]


@njit(cache=True)
def car_inputs(p, out, k):
    """The car inputs (OBS_NAMES 'car_*') into out[k:k + N_CAR_OBS]: mass, yaw inertia per kg,
    wheelbase, the front's share of the weight, steering lock, the rear tires' grip at their static
    load and the front's against it, the rear's peak slip angle, how much more force the engine can
    put down than the rear tires can take at 50 and at 90 km/h (log2, so 1 = twice; +1.5 = no engine
    limit), and the CG height."""
    f_load = p[fs.P_FRONT_STATIC_LOAD_N]
    r_load = p[fs.P_REAR_STATIC_LOAD_N]
    mass = (f_load + r_load) / 9.80665
    mu_f = p[fs.P_LAT_FRONT_PEAK_FRICTION]
    mu_r = p[fs.P_LAT_REAR_PEAK_FRICTION]
    if p[fs.P_FRONT_LS_EXPY] > 0.0:
        mu_f *= (0.5 * f_load / max(p[fs.P_FRONT_FZ0_N], 1.0)) ** (p[fs.P_FRONT_LS_EXPY] - 1.0)
    elif p[fs.P_EXT_TIRE_LOAD_SENSITIVITY] > 0.5:
        mu_f += p[fs.P_FRONT_LOAD_SENSITIVITY_PER_KN] * 0.5 * f_load / 1000.0
    if p[fs.P_REAR_LS_EXPY] > 0.0:
        mu_r *= (0.5 * r_load / max(p[fs.P_REAR_FZ0_N], 1.0)) ** (p[fs.P_REAR_LS_EXPY] - 1.0)
    elif p[fs.P_EXT_TIRE_LOAD_SENSITIVITY] > 0.5:
        mu_r += p[fs.P_REAR_LOAD_SENSITIVITY_PER_KN] * 0.5 * r_load / 1000.0
    out[k] = mass / 1300.0 - 1.0
    out[k + 1] = p[fs.P_YAW_INERTIA_KGM2] / max(mass, 1.0) / 1.95 - 1.0
    out[k + 2] = (p[fs.P_DIST_CG_FRONT_M] + p[fs.P_DIST_CG_REAR_M]) / 2.5 - 1.0
    out[k + 3] = (f_load / max(f_load + r_load, 1.0) - 0.53) * 10.0
    out[k + 4] = p[fs.P_MAX_STEER_RAD] * (180.0 / math.pi) / 40.0 - 1.0
    out[k + 5] = mu_r / 1.3 - 1.0
    out[k + 6] = (mu_f / max(mu_r, 0.1) - 1.0) * 5.0
    out[k + 7] = p[fs.P_REAR_PEAK_SLIP_RAD] * (180.0 / math.pi) / 10.0 - 1.0
    rear_grip = max(mu_r * r_load, 1.0)
    for j, speed in ((8, 50.0 / 3.6), (9, 90.0 / 3.6)):
        if p[fs.P_EXT_ENGINE] > 0.5:
            out[k + j] = max(-1.5, min(1.5, math.log2(max(fs.engine_force(p, speed), 1.0) / rear_grip)))
        else:
            out[k + j] = 1.5
    out[k + 10] = p[fs.P_CG_HEIGHT_M] / 0.45 - 1.0


@njit(cache=True)
def _sample_dt(u1, u2, cfg, prev_dt):
    """cfg: the env_cfg vector (see env_cfg_vector). prev_dt: the tick length before this one."""
    if cfg[4] < 0.5:
        return 0.049
    if u1 < cfg[5]:
        return 0.080 + 0.120 * u2
    long_p = cfg[7] if prev_dt > 0.056 else cfg[6]
    if u1 < cfg[5] + long_p:
        return 0.064 + 0.001 * u2
    return 0.048 + 0.001 * u2


@njit(cache=True)
def _reset_one(i, states, counters, prev_act, ep, r0, r1, r2, r3, r4, r5, r6, cfg, sp_pos, sp_dir):
    """r0..r6: 7 uniforms. cfg: the env_cfg vector (see env_cfg_vector)."""
    n = sp_pos.shape[0]
    k = min(n - 1, int(r0 * n))
    ang = sp_dir[k]
    recovery = r4 < cfg[8]
    lat = (r1 * 2.0 - 1.0) * (cfg[9] if recovery else cfg[0])
    heading_range = cfg[10] if recovery else cfg[1]
    s = states[i]
    for j in range(STATE_SIZE):
        s[j] = 0.0
    s[S_X] = sp_pos[k, 0] + lat * (-math.sin(ang))
    s[S_Y] = sp_pos[k, 1]
    s[S_Z] = sp_pos[k, 2] + lat * math.cos(ang)
    s[S_HEADING] = fs.wrap_pi(ang + (r2 * 2.0 - 1.0) * heading_range)
    s[S_CORR_ANGLE] = np.nan
    s[S_MAX_SPEED] = (cfg[2] + r3 * (cfg[3] - cfg[2])) / 3.6
    s[fs.S_REAR_SLIDE_GRIP] = cfg[11] + r5 * (cfg[12] - cfg[11])
    s[fs.S_PACE] = cfg[13] + r6 * (cfg[14] - cfg[13])
    s[S_PREV_NEAREST] = -1
    s[S_ARC] = np.nan
    s[fs.S_SMOOTH_H] = np.nan
    for j in range(N_RAYS):
        s[S_RAY0 + j] = 60.0
    for j in range(N_PINS):
        s[S_PIN0 + j] = 12.0
        s[S_PINSURF0 + j] = SURF_NONE
    for j in range(counters.shape[1]):
        counters[i, j] = 0.0
    for j in range(prev_act.shape[1]):
        prev_act[i, j] = 0.0
    for j in range(ep.shape[1]):
        ep[i, j] = 0.0


@njit(cache=True)
def _step_one(i, states, mids, diags, actions, prev_act, counters, ep, params, rw, obs, rewards,
              terminated, truncated, final_obs, ep_out, rnd, env_cfg, episode_ticks,
              sp_dir, la_idx, la_min_radius, pace_ref,
              sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
              tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk):
    """One environment step for car i: finish the current tick with the given action
    (step_post), score it, then start the next tick (step_pre) and build the next observation.
    A finished episode is reset in place; its last observation is kept in final_obs[i] and its
    statistics in ep_out[i]. counters: see N_COUNTERS.
    co, mk: the course edges and markers (see fast_sim.World). pace_ref: the course's reference
    speed per spline point, km/h (zeros = none; see course_pace_reference())."""
    s = states[i]
    p = params[i]
    d = diags[i]
    a0 = max(-1.0, min(1.0, actions[i, 0]))
    a1 = max(-1.0, min(1.0, actions[i, 1]))
    a2 = max(0.0, min(1.0, actions[i, 2]))
    lack = 1.0 - max(0.0, min(1.0, p[fs.P_SKILL]))     # d in SKILL_*: 0 at full skill
    if lack > 0.0 and rnd.shape[1] >= RND_COLS + 2:
        # worse hands: noise that wanders (unit variance) on a delayed reaction
        for j in range(2):
            z = (rnd[i, RND_COLS + j] - 0.5) * 3.4641016
            counters[i, 12 + j] = SKILL_NOISE_KEEP * counters[i, 12 + j] + math.sqrt(1.0 - SKILL_NOISE_KEEP ** 2) * z
        # a slower reaction: what is applied is what was asked SKILL_DELAY_TICKS x d ticks ago
        # (rounded), so the car is already further on when an input lands - corrections overshoot
        # and have to be corrected again, the busy hands of a weaker driver
        delay = int(SKILL_DELAY_TICKS * lack + 0.5)
        c0, c1, c2 = a0, a1, a2
        if delay >= 1 and counters[i, 0] >= delay:
            k = 14 if delay == 1 else 17
            a0, a1, a2 = counters[i, k], counters[i, k + 1], counters[i, k + 2]
        counters[i, 17], counters[i, 18], counters[i, 19] = counters[i, 14], counters[i, 15], counters[i, 16]
        counters[i, 14], counters[i, 15], counters[i, 16] = c0, c1, c2
        a0 = max(-1.0, min(1.0, a0 + SKILL_STEER_NOISE * lack * counters[i, 12]))
        a1 = max(-1.0, min(1.0, a1 + SKILL_PEDAL_NOISE * lack * counters[i, 13]))
    dt = mids[i, M_DT]
    fs.step_post(s, a0, a1, a2, p, d, mids[i], sp_pos, sp_radius, sp_length, sp_next, sp_prev,
                 sp_fwd, sp_cum, tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk)

    # ---- reward ----
    progress = max(-2.0, min(5.0, d[D_ProgressM]))
    abs_slip = abs(d[D_RealSlipAngleRad]) * (180.0 / math.pi)
    q = _drift_quality(abs_slip / (1.0 - SKILL_ANGLE * lack), rw)
    pace = s[fs.S_PACE]
    ref_point = max(0, min(pace_ref.shape[0] - 1, int(d[fs.D_NearestPointId])))
    ref_kmh = pace_ref[ref_point] if pace_ref.shape[0] == sp_pos.shape[0] else 0.0
    if p[fs.P_PACE_SCALE] > 0.0:
        ref_kmh *= p[fs.P_PACE_SCALE]
    # never more than the car's speed limit: on a long straight the course's reference runs up to
    # pace_model's 216 km/h ceiling, which no car limited to 72 - 88 km/h reaches, and the pace
    # error it sees there read 27% slow (Black Cat County, 2026-10-19). The plugin's UpdatePaceOver
    # asks the same.
    asked_kmh = pace * min(ref_kmh, s[fs.S_MAX_SPEED] * 3.6)
    paid_speed = asked_kmh if asked_kmh > 0.0 else rw[R_PAID_SPEED]
    paid = progress
    if paid_speed > 0.0:
        paid = min(progress, paid_speed / 3.6 * dt)
    # the car's speed along the line and the speed asked of it, averaged over about the last
    # pace_average_s seconds
    speed_now = progress / max(dt, 1e-3)
    if counters[i, 0] == 0.0:
        counters[i, 8] = speed_now
        counters[i, 9] = asked_kmh / 3.6
        counters[i, 11] = 2.0
    else:
        k_avg = min(1.0, dt / max(rw[R_PACE_AVG], 1e-3))
        counters[i, 8] += (speed_now - counters[i, 8]) * k_avg
        counters[i, 9] += (asked_kmh / 3.6 - counters[i, 9]) * k_avg
    pace_over = counters[i, 8] / counters[i, 9] - 1.0 if counters[i, 9] > 0.5 else 0.0    # share too fast
    pay = 1.0           # share of the driving pay this tick earns
    if asked_kmh > 0.0 and pace_over != 0.0:
        cut = rw[R_PACE_OVER] if pace_over > 0.0 else rw[R_PACE_UNDER]
        pay = max(rw[R_PACE_MIN_PAY], 1.0 - (1.0 - rw[R_PACE_MIN_PAY]) * abs(pace_over) / max(cut, 1e-3))
    r = pay * rw[R_PROGRESS_W] * paid * (rw[R_GRIP_FRAC] + (1.0 - rw[R_GRIP_FRAC]) * q * rw[R_SLIP_BAND_PAY])
    if rw[R_TYRE_ENERGY] > 0.0 and paid > 0.0:
        rear_slide = abs(s[S_LATV] - p[fs.P_DIST_CG_REAR_M] * s[S_YAW])
        speed = math.sqrt(s[S_OWN_SPEED] ** 2 + s[S_LATV] ** 2)
        r += pay * rw[R_TYRE_ENERGY] * paid * d[fs.D_RearForceShare] * rear_slide / max(speed, 3.0)
    push = d[D_CollisionPushOutM]
    if push > 0.0:
        r -= rw[R_PUSH_FLAT] + rw[R_PUSH_M] * min(push, 3.0)
    near = d[D_MinWallDistanceM] < rw[R_WALL_DIST]
    if near:
        r -= rw[R_WALL] * (1.0 - d[D_MinWallDistanceM] / rw[R_WALL_DIST])
    offroad = d[D_SurfaceUnderCar] != rw[R_TRACK_SURF]
    if offroad:
        r -= rw[R_OFFROAD]
    spinning = d[D_UncontrolledSpinRecovering] > 0.5
    if spinning:
        r -= rw[R_SPIN]
    overrun = course_overrun_m(d[D_CrossTrackErrorM], int(d[fs.D_NearestPointId]), rw[R_COURSE_MARGIN], co,
                               sp_pos.shape[0])
    offcourse = overrun > 0.0
    if offcourse:
        r -= rw[R_OFFCOURSE] + rw[R_OFFCOURSE_M] * min(overrun, 10.0)
    touching = marker_contact(s[S_X], s[S_Z], s[S_HEADING], mk) + 1      # 0 = none
    new_hit = touching > 0 and float(touching) != counters[i, 3]
    if new_hit:
        r -= rw[R_MARKER_HIT]
    counters[i, 3] = float(touching)
    # linked drift: counters[4] = seconds the chain has lasted, [5] = seconds since last sliding,
    # [6] = the side it was last sliding to
    slip_signed = d[D_RealSlipAngleRad] * (180.0 / math.pi)
    sliding = abs_slip >= rw[R_CHAIN_SLIP]
    transition = False
    side = 0.0
    if sliding:
        side = 1.0 if slip_signed > 0.0 else -1.0
        # a transition: sliding again, the other way, with the chain still alive
        transition = counters[i, 6] != 0.0 and side != counters[i, 6] and counters[i, 4] > 0.0
        counters[i, 6] = side
        counters[i, 5] = 0.0
        counters[i, 4] += dt
    else:
        counters[i, 5] += dt
        if counters[i, 5] > rw[R_CHAIN_GRACE]:
            counters[i, 4] = 0.0
        elif counters[i, 4] > 0.0:
            counters[i, 4] += dt
    if counters[i, 4] > 0.0 and progress > 0.0:
        r += pay * rw[R_CHAIN_BONUS] * (1.0 - SKILL_CHAIN * lack) * paid * min(1.0, counters[i, 4] / rw[R_CHAIN_FULL])
    # the drift line: the rear axle's distance inside the zone edge at this point of the course
    in_zone = False
    zone_filled = False
    point = int(d[fs.D_NearestPointId])
    if co.shape[0] == sp_pos.shape[0] and 0 <= point < co.shape[0] and co[point, 7] != 0.0:
        in_zone = True
        j = 2 if co[point, 7] > 0.0 else 4
        k = 4 if co[point, 7] > 0.0 else 2
        in_x = co[point, k] - co[point, j]
        in_z = co[point, k + 1] - co[point, j + 1]
        width = math.sqrt(in_x * in_x + in_z * in_z)
        rear_x = s[S_X] - p[fs.P_DIST_CG_REAR_M] * math.cos(s[S_HEADING])
        rear_z = s[S_Z] - p[fs.P_DIST_CG_REAR_M] * math.sin(s[S_HEADING])
        depth = ((rear_x - co[point, j]) * in_x + (rear_z - co[point, j + 1]) * in_z) / max(width, 1e-6)
        fill = 0.0
        if 0.0 <= depth <= rw[R_ZONE_DEPTH]:
            fill = 1.0
            zone_filled = abs_slip >= SLIDE_MIN_DEG
        elif depth > rw[R_ZONE_DEPTH] and rw[R_ZONE_REACH] > 0.0:
            fill = max(0.0, 1.0 - (depth - rw[R_ZONE_DEPTH]) / rw[R_ZONE_REACH])
        if paid > 0.0:
            r += pay * rw[R_ZONE_BONUS] * (1.0 - lack) * paid * q * fill
    held = co.shape[0] == sp_pos.shape[0] and 0 <= point < co.shape[0] and co[point, 8] != 0.0
    wrong_side = held and not (sliding and slip_signed * co[point, 8] < 0.0)
    if wrong_side:
        r -= rw[R_WRONG_SIDE] * dt
    if held and transition:
        r -= rw[R_SIDE_CHANGE]
    counters[i, 10] += dt
    weave = False
    if transition:
        turn = co[point, 9] if co.shape[0] == sp_pos.shape[0] and 0 <= point < co.shape[0] else 0.0
        weave = (turn != 0.0 and side != turn) or (counters[i, 10] < rw[R_WEAVE_GAP] and turn == counters[i, 11])
        counters[i, 10] = 0.0
        counters[i, 11] = turn
    if weave:
        r -= rw[R_WEAVE]
    counters[i, 21] += dt
    swing = False
    yaw_degs = s[S_YAW] * (180.0 / math.pi)
    if abs(yaw_degs) >= rw[R_SWING_RATE]:
        way = 1.0 if yaw_degs > 0.0 else -1.0
        turn = co[point, 9] if co.shape[0] == sp_pos.shape[0] and 0 <= point < co.shape[0] else 0.0
        if counters[i, 20] != 0.0 and way != counters[i, 20] and counters[i, 21] < rw[R_SWING_WINDOW]:
            swing = not (turn == way and turn != counters[i, 22])
        counters[i, 20] = way
        counters[i, 21] = 0.0
        counters[i, 22] = turn
    if swing:
        r -= rw[R_SWING]
    front_over = abs(d[fs.D_FrontSlipAngleDeg]) - rw[R_FRONT_SLIP_FREE]
    if front_over > 0.0:
        r -= rw[R_FRONT_SLIP_PEN] * front_over
    da0 = a0 - prev_act[i, 0]
    da1 = a1 - prev_act[i, 1]
    turned_back = 0.0
    if da0 * counters[i, 7] < 0.0:
        turned_back = min(abs(da0), abs(counters[i, 7]))
        r -= rw[R_STEER_REVERSAL] * (1.0 - SKILL_STEER * lack) * turned_back
    if da0 != 0.0:
        counters[i, 7] = da0
    da2 = a2 - prev_act[i, 2]
    r -= rw[R_ACT_RATE] * (1.0 - SKILL_STEER * lack) * (da0 * da0 + da1 * da1 + da2 * da2)
    prev_act[i, 0] = a0
    prev_act[i, 1] = a1
    prev_act[i, 2] = a2

    # ---- termination ----
    counters[i, 0] += 1.0
    counters[i, 1] = counters[i, 1] + 1.0 if offroad else 0.0
    counters[i, 2] = counters[i, 2] + 1.0 if d[D_ProgressM] < 0.1 else 0.0
    reason = 0.0
    if d[D_SurfaceUnderCar] == SURF_NONE:
        reason = 5.0
    elif abs(d[D_CrossTrackErrorM]) > rw[R_MAX_CTE]:
        reason = 1.0
    elif abs_slip > rw[R_MAX_SLIP]:
        reason = 2.0
    elif counters[i, 1] >= rw[R_OFFROAD_LIMIT]:
        reason = 3.0
    elif counters[i, 2] >= rw[R_STALL_LIMIT]:
        reason = 4.0
    term = reason > 0.0
    if term:
        r -= rw[R_TERMINAL]
    trunc = (not term) and counters[i, 0] >= episode_ticks
    rewards[i] = r
    terminated[i] = term
    truncated[i] = trunc

    # ---- episode statistics ----
    e = ep[i]
    e[E_RETURN] += r
    e[E_LENGTH] += 1.0
    e[E_PROGRESS] += d[D_ProgressM]
    e[E_ABS_SLIP_DEG] += abs_slip
    if 20.0 <= abs_slip <= 50.0:
        e[E_DRIFT_TICKS] += 1.0
    if SLIDE_MIN_DEG <= abs_slip <= SLIDE_MAX_DEG:
        e[E_SLIDE_TICKS] += 1.0
        if abs_slip >= HIGH_ANGLE_MIN_DEG:
            e[E_HIGH_ANGLE_TICKS] += 1.0
    if push > 0.0:
        e[E_PUSHOUT_TICKS] += 1.0
        e[E_PUSHOUT_M] += push
    if offroad:
        e[E_OFFROAD_TICKS] += 1.0
    if offcourse:
        e[E_OFFCOURSE_TICKS] += 1.0
    if new_hit:
        e[E_MARKER_HITS] += 1.0
    e[E_CHAIN_S] += counters[i, 4]
    if in_zone:
        e[E_ZONE_TICKS] += 1.0
        if zone_filled:
            e[E_ZONE_FILLED_TICKS] += 1.0
    if turned_back > STEER_REVERSAL_MIN:
        e[E_STEER_REVERSALS] += 1.0
    if front_over > 0.0:
        e[E_FRONT_SLIDING_TICKS] += 1.0
    if held:
        e[E_HELD_TICKS] += 1.0
        if wrong_side:
            e[E_WRONG_SIDE_TICKS] += 1.0
    point = max(0, min(sp_pos.shape[0] - 1, int(d[fs.D_NearestPointId])))
    if co.shape[0] == sp_pos.shape[0]:
        on_straight = co[point, 6] > 0.5
    else:
        on_straight = sp_radius[point] > STRAIGHT_RADIUS_M
    if on_straight:
        e[E_STRAIGHT_TICKS] += 1.0
        e[E_STRAIGHT_M] += max(0.0, d[D_ProgressM])
        if abs_slip >= 20.0:
            e[E_STRAIGHT_DRIFT_TICKS] += 1.0
        if transition:
            e[E_STRAIGHT_TRANSITIONS] += 1.0
    if spinning:
        e[E_SPIN_TICKS] += 1.0
    e[E_SPEED_KMH] += d[D_SpeedKmh]
    e[E_ABS_CTE] += abs(d[D_CrossTrackErrorM])
    if near:
        e[E_NEARWALL_TICKS] += 1.0
    if d[D_SteerSaturated] > 0.5:
        e[E_STEER_SAT_TICKS] += 1.0
    e[E_TIME_S] += dt
    e[E_PACE_TARGET] = pace
    # a new pace part-way through (EnvConfig.pace_change_s), from the next tick on
    if pace > 0.0 and env_cfg[15] > 0.0 and rnd[i, 11] < dt / env_cfg[15]:
        s[fs.S_PACE] = env_cfg[13] + rnd[i, 12] * (env_cfg[14] - env_cfg[13])
    if asked_kmh > 0.0:
        e[E_PACE_ERROR] += pace_over
        e[E_PACE_ABS_ERROR] += abs(pace_over)
    if weave:
        e[E_WEAVES] += 1.0
    if swing:
        e[E_SWINGS] += 1.0
    # the brake applied, not asked for: DisableBrakes (pedal mode) ignores the brake action
    a2_applied = 0.0 if (p[fs.P_EXT_REAR_FRICTION_CIRCLE] > 0.5 and p[fs.P_EXT_NO_BRAKES] > 0.5) else a2
    if a2_applied >= BRAKE_STAT_MIN:
        e[E_BRAKE_TICKS] += 1.0
    e[E_BRAKE_SUM] += a2_applied

    # ---- start the next tick, build its observation ----
    ndt = _sample_dt(rnd[i, 0], rnd[i, 1], env_cfg, dt)
    fs.step_pre(s, ndt, p, mids[i], sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd)
    build_obs(s, mids[i], prev_act[i], p, obs[i], sp_dir, la_idx, la_min_radius, co,
              pace_over if asked_kmh > 0.0 else 0.0)

    if term or trunc:
        for j in range(obs.shape[1]):
            final_obs[i, j] = obs[i, j]
        e[E_TERMINATED] = 1.0 if term else 0.0
        e[E_REASON] = reason
        for j in range(ep.shape[1]):
            ep_out[i, j] = e[j]
        _reset_one(i, states, counters, prev_act, ep, rnd[i, 2], rnd[i, 3], rnd[i, 4], rnd[i, 5], rnd[i, 8], rnd[i, 9], rnd[i, 10],
                   env_cfg, sp_pos, sp_dir)
        ndt = _sample_dt(rnd[i, 6], rnd[i, 7], env_cfg, 0.049)
        fs.step_pre(s, ndt, p, mids[i], sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd)
        build_obs(s, mids[i], prev_act[i], p, obs[i], sp_dir, la_idx, la_min_radius, co, 0.0)


@njit(cache=True, parallel=True)
def step_batch(states, mids, diags, actions, prev_act, counters, ep, params, rw, obs, rewards,
               terminated, truncated, final_obs, ep_out, rnd, env_cfg, episode_ticks,
               sp_dir, la_idx, la_min_radius, pace_ref,
               sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
               tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk):
    """_step_one for every car. Cars are independent, so the loop runs across CPU cores."""
    for i in prange(states.shape[0]):
        _step_one(i, states, mids, diags, actions, prev_act, counters, ep, params, rw, obs, rewards,
                  terminated, truncated, final_obs, ep_out, rnd, env_cfg, episode_ticks,
                  sp_dir, la_idx, la_min_radius, pace_ref,
                  sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
                  tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk)


@njit(cache=True)
def _reset_and_observe(i, states, mids, prev_act, counters, ep, params, obs, rnd, env_cfg, sp_dir, la_idx,
                       la_min_radius, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, co):
    _reset_one(i, states, counters, prev_act, ep, rnd[i, 2], rnd[i, 3], rnd[i, 4], rnd[i, 5], rnd[i, 8], rnd[i, 9], rnd[i, 10],
               env_cfg, sp_pos, sp_dir)
    ndt = _sample_dt(rnd[i, 6], rnd[i, 7], env_cfg, 0.049)
    fs.step_pre(states[i], ndt, params[i], mids[i], sp_pos, sp_radius, sp_length, sp_next,
                sp_prev, sp_fwd)
    build_obs(states[i], mids[i], prev_act[i], params[i], obs[i], sp_dir,
              la_idx, la_min_radius, co, 0.0)


@njit(cache=True)
def reset_batch(states, mids, prev_act, counters, ep, params, obs, rnd, env_cfg, sp_dir, la_idx,
                la_min_radius, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, co):
    for i in range(states.shape[0]):
        _reset_and_observe(i, states, mids, prev_act, counters, ep, params, obs, rnd, env_cfg, sp_dir, la_idx,
                           la_min_radius, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, co)


if _TARGET == 'cuda':
    # GPU entry points: one thread per car, each running the same _step_one / _reset_and_observe
    # device functions the CPU loops above call.
    from numba import cuda as _cuda

    @_cuda.jit
    def step_kernel(states, mids, diags, actions, prev_act, counters, ep, params, rw, obs, rewards,
                    terminated, truncated, final_obs, ep_out, rnd, env_cfg, episode_ticks,
                    sp_dir, la_idx, la_min_radius, pace_ref,
                    sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
                    tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk):
        i = _cuda.grid(1)
        if i < states.shape[0]:
            _step_one(i, states, mids, diags, actions, prev_act, counters, ep, params, rw, obs, rewards,
                      terminated, truncated, final_obs, ep_out, rnd, env_cfg, episode_ticks,
                      sp_dir, la_idx, la_min_radius, pace_ref,
                      sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, sp_cum,
                      tri_v, tri_surf, gmin, cell_size, dims, cell_start, cell_items,
                   rm_pos, rm_min, rm_cell, rm_dims, rm_start, rm_items, co, mk)

    @_cuda.jit
    def reset_kernel(states, mids, prev_act, counters, ep, params, obs, rnd, env_cfg, sp_dir, la_idx,
                     la_min_radius, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, co):
        i = _cuda.grid(1)
        if i < states.shape[0]:
            _reset_and_observe(i, states, mids, prev_act, counters, ep, params, obs, rnd, env_cfg, sp_dir, la_idx,
                               la_min_radius, sp_pos, sp_radius, sp_length, sp_next, sp_prev, sp_fwd, co)


def finalize_episode_stats(raw):
    """Raw per-episode accumulators (n, EP_SIZE) -> dict of per-episode arrays in EP_NAMES units."""
    raw = np.atleast_2d(raw)
    L = np.maximum(raw[:, E_LENGTH], 1.0)
    prog = raw[:, E_PROGRESS]
    return {
        'return': raw[:, E_RETURN],
        'length': raw[:, E_LENGTH],
        'progress_m': prog,
        'mean_abs_slip_deg': raw[:, E_ABS_SLIP_DEG] / L,
        'drift_fraction': raw[:, E_DRIFT_TICKS] / L,
        'pushout_tick_fraction': raw[:, E_PUSHOUT_TICKS] / L,
        'pushout_m_per_100m': 100.0 * raw[:, E_PUSHOUT_M] / np.maximum(prog, 1.0),
        'pushout_m': raw[:, E_PUSHOUT_M],
        'offroad_fraction': raw[:, E_OFFROAD_TICKS] / L,
        'spin_recovery_fraction': raw[:, E_SPIN_TICKS] / L,
        'mean_speed_kmh': raw[:, E_SPEED_KMH] / L,
        'mean_abs_cte_m': raw[:, E_ABS_CTE] / L,
        'near_wall_fraction': raw[:, E_NEARWALL_TICKS] / L,
        'steer_saturated_fraction': raw[:, E_STEER_SAT_TICKS] / L,
        'sim_seconds': raw[:, E_TIME_S],
        'terminated_early': raw[:, E_TERMINATED],
        'end_reason': raw[:, E_REASON],
        'offcourse_fraction': raw[:, E_OFFCOURSE_TICKS] / L,
        'marker_hits_per_min': 60.0 * raw[:, E_MARKER_HITS] / np.maximum(raw[:, E_TIME_S], 1e-9),
        'marker_hits': raw[:, E_MARKER_HITS],
        'straight_drift_fraction': raw[:, E_STRAIGHT_DRIFT_TICKS] / np.maximum(raw[:, E_STRAIGHT_TICKS], 1.0),
        'straight_transitions_per_100m': 100.0 * raw[:, E_STRAIGHT_TRANSITIONS] / np.maximum(raw[:, E_STRAIGHT_M], 1.0),
        'mean_chain_s': raw[:, E_CHAIN_S] / L,
        'slide_fraction': raw[:, E_SLIDE_TICKS] / L,
        'high_angle_fraction': raw[:, E_HIGH_ANGLE_TICKS] / L,
        'zone_fill_fraction': raw[:, E_ZONE_FILLED_TICKS] / np.maximum(raw[:, E_ZONE_TICKS], 1.0),
        'steer_reversals_per_s': raw[:, E_STEER_REVERSALS] / np.maximum(raw[:, E_TIME_S], 1e-9),
        'front_sliding_fraction': raw[:, E_FRONT_SLIDING_TICKS] / L,
        'wrong_side_fraction': raw[:, E_WRONG_SIDE_TICKS] / np.maximum(raw[:, E_HELD_TICKS], 1.0),
        'pace': raw[:, E_PACE_TARGET],                  # the pace asked (a fraction of the reference), last tick
        'pace_error_pct': 100.0 * raw[:, E_PACE_ERROR] / L,   # mean % the averaged speed was above the asked; 0 with no pace
        # mean % off the asked speed either way: the one to watch over a range of paces, where the
        # signed mean can sit near 0 with every pace missed (too fast when slow is asked, too slow
        # when fast is)
        'pace_abs_error_pct': 100.0 * raw[:, E_PACE_ABS_ERROR] / L,
        'brake_fraction': raw[:, E_BRAKE_TICKS] / L,
        'mean_brake': raw[:, E_BRAKE_SUM] / L,
        'weaves_per_min': 60.0 * raw[:, E_WEAVES] / np.maximum(raw[:, E_TIME_S], 1e-9),
        'swings_per_min': 60.0 * raw[:, E_SWINGS] / np.maximum(raw[:, E_TIME_S], 1e-9),
    }


# ----------------------------------------------------------------------------------------
# Python-facing environment
# ----------------------------------------------------------------------------------------
def spline_lookahead_tables(world):
    """Per spline point: its direction angle, the point index LOOKAHEAD_M metres further on, and
    the tightest radius within the next MIN_RADIUS_WINDOW_M. Cheap to rebuild identically in C#
    at plugin start-up (walk NextId, accumulate Length)."""
    n = len(world.sp_pos)
    sp_dir = np.arctan2(world.sp_fwd[:, 2], world.sp_fwd[:, 0])
    la_idx = np.zeros((n, N_LOOK), dtype=np.int64)
    la_min_radius = np.zeros(n)
    for i in range(n):
        for j, dist in enumerate(LOOKAHEAD_M):
            cur, acc = i, 0.0
            while acc + world.sp_length[cur] <= dist and world.sp_next[cur] >= 0:
                acc += world.sp_length[cur]
                cur = world.sp_next[cur]
            la_idx[i, j] = cur
        cur, acc, mr = i, 0.0, world.sp_radius[i]
        while acc < MIN_RADIUS_WINDOW_M and world.sp_next[cur] >= 0:
            acc += world.sp_length[cur]
            cur = world.sp_next[cur]
            mr = min(mr, world.sp_radius[cur])
        la_min_radius[i] = mr
    return sp_dir, la_idx, la_min_radius


RND_COLS = 13      # uniforms per car per step: [0,1] next tick length, [2..5,8,9,10] reset, [6,7] tick length after a reset,
                   # [11,12] a new pace mid-episode


ENV_CFG_SLIDING_GRIP_LOW = 11      # indices into env_cfg_vector()
ENV_CFG_PACE_LOW = 13


def env_cfg_vector(cfg):
    """EnvConfig -> the flat vector the compiled kernels read."""
    return np.array([cfg.spawn_lateral_m, cfg.spawn_heading_rad, cfg.max_speed_kmh_low, cfg.max_speed_kmh_high,
                     1.0 if cfg.randomise_dt else 0.0, cfg.hitch_probability, cfg.long_after_short_probability,
                     cfg.long_after_long_probability, cfg.recovery_start_fraction, cfg.recovery_lateral_m,
                     cfg.recovery_heading_rad, cfg.rear_sliding_grip_low, cfg.rear_sliding_grip_high,
                     cfg.pace_low, cfg.pace_high, cfg.pace_change_s])


def surface_under(world, x, y, z):
    """The ground's surface code directly under (x, z), from 3 m above y (fast_sim.surface_below) -
    the same probe fast_sim.step_post uses for D_SurfaceUnderCar."""
    return int(fs.surface_below(x, y + 3.0, z, 12.0, *world.ray_arrays()))


def track_surface_code(world):
    codes = [surface_under(world, p[0], p[1], p[2]) for p in world.sp_pos]
    return float(np.bincount(codes, minlength=5).argmax())


class DriftVecEnv:
    def __init__(self, world, n_envs=64, env_config=None, reward_config=None, seed=0):
        self.world = world
        self.W = world.arrays()
        self.n = n_envs
        self.cfg = env_config or EnvConfig()
        self.rcfg = reward_config or RewardConfig()
        if self.rcfg.wall_proximity_distance_m > fs.WALL_PROXIMITY_RANGE_M:
            raise ValueError(f"wall_proximity_distance_m above fast_sim.WALL_PROXIMITY_RANGE_M ({fs.WALL_PROXIMITY_RANGE_M} m)")
        self.rw = self.rcfg.vector()
        if self.rw[R_TRACK_SURF] < 0:
            self.rw[R_TRACK_SURF] = track_surface_code(world)
        self.rng = np.random.default_rng(seed)
        self.sp_dir, self.la_idx, self.la_min_radius = spline_lookahead_tables(world)
        # the course's reference speed per point for paces (course_pace_reference(); set on the world by
        # the trainer); a car with a pace on a world without one is asked for nothing
        ref = getattr(world, 'pace_ref', None)
        self.pace_ref = np.ascontiguousarray(ref if ref is not None else np.zeros(len(world.sp_pos)), dtype=np.float64)
        self.base_params = fs.make_params()
        apply_plugin_settings(self.base_params, self.cfg.plugin_settings)
        self.params = np.tile(self.base_params, (n_envs, 1))
        self.car_index = np.zeros(n_envs, dtype=np.int64)       # which car_pool entry each drives
        self.env_cfg = env_cfg_vector(self.cfg)
        self.states = np.zeros((n_envs, STATE_SIZE))
        self.mids = np.zeros((n_envs, MID_SIZE))
        self.diags = np.zeros((n_envs, DIAG_SIZE))
        self.prev_act = np.zeros((n_envs, ACT_SIZE))
        self.counters = np.zeros((n_envs, N_COUNTERS))
        self.ep = np.zeros((n_envs, EP_SIZE))
        self.obs = np.zeros((n_envs, OBS_SIZE))
        self.rewards = np.zeros(n_envs)
        self.terminated = np.zeros(n_envs, dtype=np.bool_)
        self.truncated = np.zeros(n_envs, dtype=np.bool_)
        self.final_obs = np.zeros((n_envs, OBS_SIZE))
        self.ep_out = np.zeros((n_envs, EP_SIZE))

    def _randomise_grip(self, idx):
        g = self.cfg.grip_randomisation
        if g <= 0.0:
            return
        scale = 1.0 + self.rng.uniform(-g, g, size=len(idx))
        for name in ('LAT_FRONT_PEAK_FRICTION', 'LAT_REAR_PEAK_FRICTION'):
            if self.cfg.car_pool:
                base = np.array([self.cfg.car_pool[k].get(name, self.base_params[fs.P[name]]) for k in self.car_index[idx]])
            else:
                base = self.base_params[fs.P[name]]
            self.params[idx, fs.P[name]] = base * scale

    def _draw_cars(self, idx):
        """A car from the pool for each of these environments (car_pool), varied."""
        pool = self.cfg.car_pool
        if not pool:
            return
        picks = self.rng.integers(0, len(pool), size=len(idx))
        mass_r, power_r = self.cfg.car_mass_randomisation, self.cfg.car_power_randomisation
        for i, k in zip(idx, picks):
            row = self.params[i]
            for name, value in pool[k].items():
                row[fs.P[name]] = value
            if mass_r > 0.0:
                m = 1.0 + self.rng.uniform(-mass_r, mass_r)
                for name in ('FRONT_STATIC_LOAD_N', 'REAR_STATIC_LOAD_N', 'YAW_INERTIA_KGM2'):
                    row[fs.P[name]] *= m
            if power_r > 0.0:
                row[fs.P_ENGINE_FORCE_00:fs.P_ENGINE_FORCE_00 + fs.ENGINE_CURVE_POINTS] *= \
                    1.0 + self.rng.uniform(-power_r, power_r)
            self.car_index[i] = k

    def _draw_skill(self, idx):
        if self.cfg.skill_low >= 1.0 and self.cfg.skill_high >= 1.0:
            return
        self.params[idx, fs.P_SKILL] = self.rng.uniform(self.cfg.skill_low, self.cfg.skill_high, size=len(idx))

    def _rnd(self):
        """The step's uniforms; two more per car (the noise of a lower skill) only when skill is
        drawn, so a run without it draws exactly the numbers it always did."""
        rnd = self.rng.random((self.n, RND_COLS))
        if self.cfg.skill_low < 1.0 or self.cfg.skill_high < 1.0:
            rnd = np.concatenate([rnd, self.rng.random((self.n, 2))], axis=1)
        return rnd

    def reset(self):
        self._draw_skill(np.arange(self.n))
        self._draw_cars(np.arange(self.n))
        self._randomise_grip(np.arange(self.n))
        rnd = self.rng.random((self.n, RND_COLS))
        reset_batch(self.states, self.mids, self.prev_act, self.counters, self.ep, self.params,
                    self.obs, rnd, self.env_cfg, self.sp_dir, self.la_idx, self.la_min_radius,
                    *self.W[:6], self.W[20])
        return self.obs.copy()

    def step(self, actions):
        """Returns obs, reward, terminated, truncated, info. Auto-resets finished cars: for those,
        obs is the new episode's first observation and info['final_obs'] holds the last one of
        the episode that just ended (needed to bootstrap the value on a time-limit truncation).
        info['episodes'] is a dict of per-episode statistics for episodes that ended this step."""
        rnd = self._rnd()
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape[1] < ACT_SIZE:
            # a policy with fewer outputs (two: no brake action) - the rest are 0
            actions = np.concatenate([actions, np.zeros((actions.shape[0], ACT_SIZE - actions.shape[1]))], axis=1)
        step_batch(self.states, self.mids, self.diags, np.ascontiguousarray(actions),
                   self.prev_act, self.counters, self.ep, self.params, self.rw, self.obs, self.rewards,
                   self.terminated, self.truncated, self.final_obs, self.ep_out, rnd,
                   self.env_cfg, float(self.cfg.episode_ticks), self.sp_dir, self.la_idx,
                   self.la_min_radius, self.pace_ref, *self.W)
        info = {'final_obs': self.final_obs}
        done_idx = np.nonzero(self.terminated | self.truncated)[0]
        if len(done_idx):
            info['episodes'] = finalize_episode_stats(self.ep_out[done_idx])
            info['episode_cars'] = done_idx
            self._draw_skill(done_idx)
            self._draw_cars(done_idx)
            self._randomise_grip(done_idx)
        return self.obs.copy(), self.rewards.copy(), self.terminated.copy(), self.truncated.copy(), info


# car_param_values(): a car's simulator parameters from its cars/car_numbers.py entry (it lives there,
# free of the simulator, so setup can write the plugin's car files with it)
from cars.car_numbers import DRIFT_STEER_LOCK_DEG, car_param_values  # noqa: E402,F401


def load_car_pool(path, names=None, exclude=(), steer_lock_deg=None):
    """Cars from a car_numbers.py --json file as car_param_values() dicts: the named ones (all
    by default), less exclude, and never one that drives all four wheels."""
    import json
    with open(path) as f:
        cars = json.load(f)
    out = []
    for c in cars:
        if (names and c['car'] not in names) or c['car'] in exclude:
            continue
        if str(c.get('driven_wheels', 'RWD')).upper() != 'RWD':
            continue
        out.append((c['car'], car_param_values(c, steer_lock_deg)))
    return out


def car_pace_scale(worlds, car):
    """How fast this car (a car_param_values() dict) goes against the default car the courses'
    pace references are for: pace_model's speed round each world for it over the default car's,
    averaged. A pace then asks it for pace x reference x this, so a slow car is not asked for a
    fast one's speed. (Measured until 2026-10-18 by a reference policy driving each car.)"""
    import pace_model
    car = {k: v for k, v in car.items() if not k.startswith('_')}
    return pace_model.car_pace_scale(worlds, car)


def policy_pace_scale(world, policy_meta, car, base_dir=None):
    """car_pace_scale() on this one world: what a server on this track should give the car
    (PACE_SCALE in the plugin's car file). The policy file and base_dir are no longer needed (a
    reference policy measured it before pace_model); kept so older callers still work."""
    return car_pace_scale([world], car)


def policy_course_pace_reference(data_dir, course, world, policy_meta=None, base_dir=None):
    """course_pace_reference() for a policy file: lets setup give a track the policy never trained
    on its <course>_pace_reference.csv. The same for every policy now (pace_model)."""
    return course_pace_reference(data_dir, course, world)


def evaluate(world, policy_fn, n_episodes=64, seed=12345, env_config=None, reward_config=None):
    """Runs n_episodes complete episodes (one per car, no auto-reset carry-over) under
    policy_fn(obs)->actions and returns the dict of per-episode statistics. The same seed gives
    the same spawns and the same tick-length sequence for every policy, so two calls with the
    same seed are a like-for-like comparison."""
    # always from ordinary on-line starts, whatever mix of recovery starts training used
    eval_config = replace(env_config or EnvConfig(), recovery_start_fraction=0.0)
    env = DriftVecEnv(world, n_envs=n_episodes, env_config=eval_config, reward_config=reward_config,
                      seed=seed)
    obs = env.reset()
    finished = np.zeros(n_episodes, dtype=bool)
    stats = np.zeros((n_episodes, EP_SIZE))
    while not finished.all():
        act = policy_fn(obs)
        obs, r, term, trunc, info = env.step(act)
        done = (term | trunc) & ~finished      # only each car's FIRST episode counts
        stats[done] = env.ep_out[done]
        finished |= done
    return finalize_episode_stats(stats)


def env_config_for(policy_meta=None, **overrides):
    """The EnvConfig a policy file was trained for. Every tool that runs a policy file builds its
    environment through this."""
    meta = policy_meta or {}
    require_direct(meta)
    grip_low, grip_high = sliding_grip_range(meta)
    pace_low, pace_high = pace_range(meta)
    cfg = EnvConfig(rear_sliding_grip_low=grip_low, rear_sliding_grip_high=grip_high,
                    pace_low=pace_low, pace_high=pace_high,
                    plugin_settings=dict(meta.get('plugin_settings') or {}))
    return replace(cfg, **overrides) if overrides else cfg


SLIDING_GRIP_OBS_SLOT = 35      # see build_obs

# plugin setting name -> (simulator parameter, conversion). The settings a policy file may carry.
PLUGIN_SETTING_PARAMS = {
    'MaxSteerAngleDeg': ('MAX_STEER_RAD', math.radians),
    'WrongWayEntryThresholdDeg': ('WRONG_WAY_ENTRY_DEG', float),
    'WrongWayEntryTicks': ('WRONG_WAY_ENTRY_TICKS', float),
    'WrongWayExitSlipDeg': ('WRONG_WAY_EXIT_SLIP_DEG', float),
    'WrongWayExitYawRateDegS': ('WRONG_WAY_EXIT_YAW_DEGS', float),
    'EnableRearThrustLimit': ('EXT_REAR_THRUST_LIMIT', float),      # 1 = on
    'MaxSteerRateDegS': ('MAX_STEER_RATE_RAD_S', math.radians),
    'EnableRearBrakeLimit': ('EXT_REAR_BRAKE_LIMIT', float),        # 1 = on
    'EnableRearFrictionCircle': ('EXT_REAR_FRICTION_CIRCLE', float),  # 1 = on
    'EnableWeightTransfer': ('EXT_WEIGHT_TRANSFER', float),          # 1 = on
    'CgHeightMeters': ('CG_HEIGHT_M', float),
    'LoadTransferTimeConstantS': ('LOAD_TRANSFER_TIME_S', float),
    'EnableTireLoadSensitivity': ('EXT_TIRE_LOAD_SENSITIVITY', float),  # 1 = on
    'FrontTireLoadSensitivityPerKn': ('FRONT_LOAD_SENSITIVITY_PER_KN', float),
    'RearTireLoadSensitivityPerKn': ('REAR_LOAD_SENSITIVITY_PER_KN', float),
    'FrontTrackMeters': ('FRONT_TRACK_M', float),
    'RearTrackMeters': ('REAR_TRACK_M', float),
    'RollStiffnessFrontShare': ('ROLL_STIFFNESS_FRONT_SHARE', float),
    # the proximity and contact pins' height over the car's position (0.40 m above the road):
    # 0.15 = 0.55 m up, over the top of a low tire wall (Spirit Peaks' straight, 15 - 27 cm)
    'ProximityPinHeightOffsetMeters': ('PROXIMITY_PIN_HEIGHT_OFFSET_M', float),
    'DisableBrakes': ('EXT_NO_BRAKES', float),                      # 1 = no pedal brake, no brake action
    'FrontBrakeOnly': ('EXT_FRONT_BRAKE_ONLY', float),              # 1 = no pedal brake; the brake action stays
    'CarOutlineContact': ('EXT_CAR_OUTLINE_CONTACT', float),        # 1 = walls touch the car's outline, not a circle
}
DEFAULT_MAX_STEER_DEG = 28.125      # the plugin's MaxSteerAngleRadians default


def apply_plugin_settings(params, settings):
    """Writes a plugin_settings dict into a parameter vector (or every row of a matrix of them)."""
    for name, value in (settings or {}).items():
        if name not in PLUGIN_SETTING_PARAMS:
            raise KeyError(f"plugin setting '{name}' is not one the simulator can take from a policy file "
                           f"({', '.join(PLUGIN_SETTING_PARAMS)})")
        param, convert = PLUGIN_SETTING_PARAMS[name]
        params[..., fs.P[param]] = convert(float(value))
    return params


def max_steer_deg(policy_meta):
    return float(((policy_meta or {}).get('plugin_settings') or {}).get('MaxSteerAngleDeg', DEFAULT_MAX_STEER_DEG))


def sliding_grip_range(policy_meta):
    """(low, high) of the RearSlidingGrip values a policy file was trained over; (1, 1) for every
    policy without the key, i.e. everything up to direct3."""
    r = (policy_meta or {}).get('rear_sliding_grip_range')
    if not r:
        return 1.0, 1.0
    return float(r[0]), float(r[1])


def pace_range(policy_meta):
    """(low, high), fractions of the reference speed, of the paces a policy file was trained to
    drive; (0, 0) for a policy without them (gen4's km/h paces, "pace_kmh_range", are not run by
    this simulator: it asks for fractions)."""
    r = (policy_meta or {}).get('pace_fraction_range')
    if not r:
        return 0.0, 0.0
    return float(r[0]), float(r[1])


def pace_reference(world, policy, n_cars=64, seed=3, sliding_grip=None, smooth_m=20.0, lost_s=3.0, car=None):
    """How fast `policy` drives each spline point flat out, km/h (a measurement for reports, e.g.
    drive_report --pace-reference; courses' pace references are pace_model's since 2026-10-18,
    course_pace_reference()): no pace; at sliding_grip, else the middle of its range; the middle of
    the speed ceiling range; smoothed along about smooth_m of the line. Per point: the median over cars of
    each car's mean speed there, leaving out each car's last lost_s before it was lost (a car
    about to leave the course is not driving flat out; on Drift Playground the
    plain mean over every tick asked 16 km/h of the hairpin's approach instead of ~45). Points
    fewer than 3 cars passed take the nearest measured value."""
    meta = policy.meta
    grip = sliding_grip if sliding_grip is not None else 0.5 * sum(sliding_grip_range(meta))
    cfg = env_config_for(meta, rear_sliding_grip_low=grip, rear_sliding_grip_high=grip, recovery_start_fraction=0.0,
                         pace_low=0.0, pace_high=0.0)
    if car is not None:                 # one car_param_values() dict: that car instead of the policy's
        cfg = replace(cfg, car_pool=(car,))
    env = DriftVecEnv(world, n_envs=n_cars, env_config=cfg, seed=seed)
    obs = env.reset()
    n = len(world.sp_pos)
    if len(policy.obs_mean) > PACE_OBS_SLOT:
        print("  (pace_reference: the reference policy reads a pace; it is given none, which it may never have seen)")
    total = np.zeros((n_cars, n))
    count = np.zeros((n_cars, n))
    recent = []                  # each tick's (point, speed, cars counted), kept until it is lost_s old
    lost_ticks = 0.0
    finished = np.zeros(n_cars, dtype=bool)
    cars = np.arange(n_cars)

    def commit(entry):
        point, speed, ok = entry
        np.add.at(total, (cars[ok], point[ok]), speed[ok])
        np.add.at(count, (cars[ok], point[ok]), 1.0)

    while not finished.all():
        obs, r, term, trunc, info = env.step(policy(obs))
        dt = np.maximum(env.mids[:, M_DT], 1e-3)
        lost_ticks = lost_s / float(np.median(dt))
        point = env.states[:, fs.S_PREV_NEAREST].astype(np.int64)
        ok = ~finished & (point >= 0) & (point < n)
        recent.append((point, env.diags[:, fs.D_ProgressM] / dt * 3.6, ok))
        for entry in recent:         # a car lost this tick: drop its last lost_s
            entry[2][term & ~finished] = False
        while len(recent) > lost_ticks:
            commit(recent.pop(0))
        finished |= term | trunc
    for entry in recent:             # the cars still running at the end were not lost
        commit(entry)
    passed = count > 0
    per_car = np.where(passed, total / np.maximum(count, 1.0), np.nan)
    measured = passed.sum(axis=0) >= 3
    if not measured.any():
        raise RuntimeError("pace_reference: no run got anywhere on this course")
    v = np.full(n, np.nan)
    v[measured] = np.nanmedian(per_car[:, measured], axis=0)
    idx = np.arange(n)
    v = np.interp(idx, idx[measured], v[measured], period=n)          # fill the gaps round the loop
    # smooth along the line: the mean over the points within smooth_m either way
    arc = np.r_[0.0, np.cumsum(world.sp_length[:-1])]
    lap = float(world.sp_length.sum())
    out = np.empty(n)
    for i in range(n):
        gap = np.abs(arc - arc[i])
        gap = np.minimum(gap, lap - gap)
        out[i] = v[gap <= smooth_m].mean()
    return np.maximum(out, 5.0)


# Plugin settings that are the car's physics rather than how a policy drives: drive_report measures
# a policy's flat-out speed with its own settings plus the run's values of these.
PHYSICS_SETTINGS = ('EnableTireLoadSensitivity', 'FrontTireLoadSensitivityPerKn', 'RearTireLoadSensitivityPerKn',
                    'FrontTrackMeters', 'RearTrackMeters', 'RollStiffnessFrontShare', 'CgHeightMeters')


def read_pace_reference(path):
    """(km/h per point, source) from a <course>_pace_reference.csv; source is the file's Source
    column (pace_model.MODEL for pace_model's), or None for an older file measured by a policy."""
    import csv
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'PointId' not in rows[0] or 'Kmh' not in rows[0]:
        raise RuntimeError(f"{path} lacks the PointId / Kmh columns")
    ids = np.array([int(r['PointId']) for r in rows])
    if not np.array_equal(ids, np.arange(len(rows))):
        raise RuntimeError(f"{path} is not in point order")
    return np.array([float(r['Kmh']) for r in rows]), (rows[0].get('Source') or None)


def write_pace_reference(path, kmh, source):
    with open(path, 'w') as f:
        f.write('PointId,Kmh,Source\n')
        for i, v in enumerate(kmh):
            f.write(f'{i},{v:.3f},{source}\n')


def course_pace_reference(data_dir, course, world, *_old_args, **_old_kwargs):
    """The course's reference speed per point, km/h (what a pace of 1.0 asks for), from
    <data_dir>/<course>_pace_reference.csv; made with pace_model and written there when the file is
    missing, or was not made by pace_model (an older one, measured by a reference policy, is kept
    beside it as .before_pace_model). The same file goes to the server folder for the plugin, which
    reads only PointId and Kmh."""
    import pace_model
    import track_files as tf
    path = tf.pace_reference_path(data_dir, course)
    n = len(world.sp_pos)
    if os.path.exists(path):
        kmh, source = read_pace_reference(path)
        if source == pace_model.MODEL:
            if len(kmh) != n:
                raise RuntimeError(f"{path} does not have one row per AI-line point ({n}); delete it to make it again")
            return kmh
        old, k = path + '.before_pace_model', 1
        while os.path.exists(old):
            k += 1
            old = f'{path}.before_pace_model{k}'
        os.replace(path, old)
    kmh = pace_model.course_speed(world)
    write_pace_reference(path, kmh, pace_model.MODEL)
    return read_pace_reference(path)[0]


def require_direct(policy_meta):
    """The earliest policies corrected the hand-written controller ("residual" mode),
    which is gone from the plugin and from this simulator."""
    if policy_meta and policy_meta.get('control_mode', 'residual') != 'direct':
        raise SystemExit("this policy corrects the hand-written controller (residual mode), which the "
                         "plugin and the simulator no longer have; only direct-control policies run")


def server_run_configs(policy_meta=None, ticks=3300, ceiling_kmh=None, sliding_grip=None, pace=None, skill=None):
    """(EnvConfig, RewardConfig) for simulating what the SERVER does with a car: one long run of
    `ticks` ticks (3300 is about 3 minutes) in which nothing ever resets the car - on the server a
    car that leaves the track has to drive back, and the time it spends doing that is what the
    first live run showed the 800-tick training episodes hide (they end when the car is lost,
    so a crash costs an episode ~1 s of bad numbers instead of ~15 s).
    Use with evaluate(): every car's single "episode" is then a whole server run.
    ceiling_kmh fixes the car's speed limit (a telemetry file's CeilingKmh); None keeps the
    training range. sliding_grip fixes RearSlidingGrip; None keeps the range the policy was
    trained over; pace fixes the pace (a fraction; a policy with paces), None keeps its range.
    A pace needs the world's pace_ref (course_pace_reference()). skill fixes the skill input (the
    plugin's BotSkill; a policy trained with one); None drives at full skill."""
    cfg = env_config_for(policy_meta, episode_ticks=int(ticks))
    if skill is not None:
        cfg = replace(cfg, skill_low=float(skill), skill_high=float(skill))
    if pace is not None:
        cfg = replace(cfg, pace_low=float(pace), pace_high=float(pace))
    if ceiling_kmh is not None:
        cfg = replace(cfg, max_speed_kmh_low=float(ceiling_kmh), max_speed_kmh_high=float(ceiling_kmh))
    if sliding_grip is not None:
        cfg = replace(cfg, rear_sliding_grip_low=float(sliding_grip), rear_sliding_grip_high=float(sliding_grip))
    never_reset = RewardConfig(max_abs_cte_m=1e9, max_abs_slip_deg=1e9, offroad_ticks_limit=1e9,
                               stall_ticks_limit=1e9)
    return cfg, never_reset
