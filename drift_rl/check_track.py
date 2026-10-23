"""
check_track.py - checks a track's data before anyone trains on it or puts bots on it.

    python check_track.py --track NAME [--tracks FOLDER] [--policy POLICY.json]

Bad track data cannot be trained away: Spirit Peaks' tire wall was too low for the sensors to see,
so gen12 learned a line through it (server test 29), and a speed reference that asked 70 km/h of
Drift Playground's 10 m hairpin had the bots fly straight through it (test 28). This looks for
those two kinds of problem and says where they are:

  * speed targets (<track>_pace_reference.csv): one per AI-line point, each a real speed; none
    asking more than MAX_GRIP_SHARE x the car's grip in a corner for MIN_RUN points in a row
    (pace_model asks at most ~1.4 on the owner's five tracks; the old flat-out reference asked
    3.1 in Drift Playground's hairpin); none under MIN_KMH (a kink in the AI line);
  * walls (<track>_collision.csv.gz): none closer to the AI line than MIN_ROOM_M, and no stretch
    of MIN_RUN points or more where a wall beside the course is too low for the policy's contact
    pins (ProximityPinHeightOffsetMeters, from the policy file) to feel it.

setup_track.py runs it after making a track's data, and gradient-drift-setup before it writes
anything to the server; both stop when it finds a problem. Exit code 1 when it does.
"""
import argparse
import json
import os
import sys

import numpy as np

import fast_sim as fs
import line_tools as lt
import pace_model as pm
import track_files as tf

MAX_GRIP_SHARE = 1.8         # sideways acceleration a speed target asks, over the car's grip x g
MIN_KMH = 15.0
MIN_ROOM_M = 0.5             # the AI line's distance to the nearest wall
MIN_RUN = 5                  # points in a row (about 7 m) before a stretch counts
CURVE_WINDOW_M = 10.0        # the corner's radius, over this much line
CAR_HEIGHT_M = 0.40          # the car's position above the road (GROUND_RAY_HEIGHT_OFFSET_M)
LOW_RAY_M = 0.03             # a wall the game collides with this high above the road is a wall
REACH_BEYOND_COURSE_M = 1.5  # walls this far outside the course's edge count


def pin_offset(policy=None):
    """The contact pins' height over the car's position: the policy file's
    ProximityPinHeightOffsetMeters, else the simulator's default."""
    if policy and os.path.exists(policy):
        with open(policy, encoding='utf-8') as f:
            settings = json.load(f).get('plugin_settings') or {}
        if 'ProximityPinHeightOffsetMeters' in settings:
            return float(settings['ProximityPinHeightOffsetMeters'])
    return float(fs.make_params()[fs.P['PROXIMITY_PIN_HEIGHT_OFFSET_M']])


def stretches(flags, min_run=MIN_RUN, gap=3):
    """(first, last) point of each run of flagged points, flagged points no more than gap apart,
    with at least min_run of them; round the loop."""
    idx = np.nonzero(flags)[0]
    if len(idx) == 0:
        return []
    n = len(flags)
    runs, run = [], [idx[0]]
    for i in idx[1:]:
        if i - run[-1] <= gap:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    if len(runs) > 1 and runs[0][0] + n - runs[-1][-1] <= gap:
        runs[0] = runs.pop() + runs[0]
    return [(r[0], r[-1]) for r in runs if len(r) >= min_run]


def check_speeds(world, kmh):
    n = len(world.sp_pos)
    if len(kmh) != n:
        return [f"the speed targets have {len(kmh)} points and the AI line {n}: they were made for another line "
                "(delete the pace reference to make it again)"]
    kmh = np.asarray(kmh, dtype=np.float64)
    if not np.all(np.isfinite(kmh)) or np.any(kmh <= 0):
        return ["the speed targets have a value that is not a speed above 0"]
    problems = []
    v = kmh / 3.6
    share = v * v * pm.curvature(world, CURVE_WINDOW_M) / (pm.car_terms()['mu'] * pm.G)
    for a, b in stretches(share > MAX_GRIP_SHARE, gap=1):
        worst = (a + int(np.argmax(np.roll(share, -a)[:(b - a) % n + 1]))) % n
        problems.append(f"the speed targets ask too much at points {a} - {b}: {kmh[worst]:.0f} km/h at point "
                        f"{worst} is {share[worst]:.1f} times the grip the car has there (radius "
                        f"{1.0 / max(pm.curvature(world, CURVE_WINDOW_M)[worst], 1e-6):.0f} m). The bots cannot "
                        "drift it and learn to go straight; delete the pace reference to make it again from the "
                        "track's shape")
    slow = np.nonzero(kmh < MIN_KMH)[0]
    if len(slow):
        problems.append(f"the speed targets ask under {MIN_KMH:g} km/h at {len(slow)} points (first {slow[0]}, "
                        f"{kmh[slow[0]]:.0f} km/h): the AI line probably has a kink there; check it, or rebuild it "
                        "(setup_track.py --line rebuild)")
    return problems


def check_walls(world, course_edges, pin_offset_m):
    """Problems with the walls near the AI line. course_edges: the rows of <track>_course_edges.csv."""
    R = world.ray_arrays()
    pos = np.asarray(world.sp_pos, dtype=np.float64)
    xz = pos[:, [0, 2]]
    _, nn = lt.normals(xz)
    n = len(xz)
    edges = np.asarray(course_edges, dtype=np.float64)
    # which side of the line is the file's positive side
    sign = 1.0 if np.median(np.sum((edges[:, 2:4] - xz) * nn, 1)) >= 0 else -1.0
    room = {sign: edges[:, 0], -sign: edges[:, 1]}
    pin_m = CAR_HEIGHT_M + pin_offset_m
    nearest = np.full(n, np.inf)
    hidden = np.zeros(n, bool)
    for i in range(n):
        gd, _ = fs.ground_below(xz[i, 0], pos[i, 1] + 1.0, xz[i, 1], 4.0, *R)
        gy = pos[i, 1] + 1.0 - gd if gd >= 0 else pos[i, 1] - CAR_HEIGHT_M
        for sg in (1.0, -1.0):
            dx, dz = sg * nn[i, 0], sg * nn[i, 1]
            d, _ = fs.raycast(xz[i, 0], gy + LOW_RAY_M, xz[i, 1], dx, 0.0, dz, room[sg][i] + REACH_BEYOND_COURSE_M,
                              *R, 2)
            if d < 0:
                continue
            nearest[i] = min(nearest[i], d)
            # from where a car beside the wall would be, on the ground there: can its pins feel it?
            ox, oz = xz[i, 0] + dx * max(d - 1.0, 0.0), xz[i, 1] + dz * max(d - 1.0, 0.0)
            g2, _ = fs.ground_below(ox, gy + 1.0, oz, 4.0, *R)
            oy = gy + 1.0 - g2 if g2 >= 0 else gy
            low, _ = fs.raycast(ox, oy + LOW_RAY_M, oz, dx, 0.0, dz, 2.5, *R, 2)
            pin, _ = fs.raycast(ox, oy + pin_m, oz, dx, 0.0, dz, 3.0, *R, 2)
            if low >= 0 and (pin < 0 or pin > low + 0.5):
                hidden[i] = True
    problems = []
    close = np.nonzero(nearest < MIN_ROOM_M)[0]
    if len(close):
        problems.append(f"the AI line runs within {MIN_ROOM_M:g} m of a wall at {len(close)} points (first {close[0]}, "
                        f"{nearest[close[0]]:.2f} m): the bots would follow it into the wall; rebuild the line "
                        "(setup_track.py --line rebuild)")
    for a, b in stretches(hidden):
        problems.append(f"a wall beside the course at points {a} - {b} is lower than the bots' wall sensors "
                        f"({pin_m:.2f} m above the road): they cannot feel it and will learn to drive through it. "
                        "A policy with lower sensors (ProximityPinHeightOffsetMeters) sees it")
    return problems


def check(tracks_dir, track, policy=None, log=print, pace_file=None):
    """The problems found with a track's data (an empty list: none). Needs its AI line, collision
    file and course edges; checks its pace reference when there is one (pace_file: another file to
    check instead, such as the one a server already runs)."""
    import pandas as pd
    import drift_env as de
    world = fs.load_world(tracks_dir, track=track)
    log(f"checking {track}'s speed targets and walls")
    problems = []
    ref = pace_file or tf.pace_reference_path(tracks_dir, track)
    if os.path.exists(ref):
        problems += check_speeds(world, de.read_pace_reference(ref)[0])
    edges_path = tf.path(tracks_dir, 'course_edges', track)
    if os.path.exists(edges_path):
        edges = pd.read_csv(edges_path)[['PositiveSideM', 'NegativeSideM', 'PositiveEdgeX', 'PositiveEdgeZ']]
        problems += check_walls(world, edges.to_numpy(), pin_offset(policy))
    return problems


def report(track, problems):
    """The message for a track that failed: what is wrong, and how to go on anyway."""
    return (f"{track}'s data looks wrong for the bots:\n"
            + "\n".join(f"  - {p}" for p in problems))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', required=True)
    tf.tracks_arg(ap)
    ap.add_argument('--policy', default=None, help="the policy that will drive it (its sensors' height)")
    args = ap.parse_args(argv)
    problems = check(args.tracks, args.track, args.policy)
    if problems:
        print(report(args.track, problems))
        return 1
    print(f"{args.track}: speed targets and walls look right")
    return 0


if __name__ == '__main__':
    sys.exit(main())
