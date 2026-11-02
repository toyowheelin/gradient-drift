"""
surface_grip.py - where on a track the ground grips differently from the main road (SurfaceGrip).

    python surface_grip.py --track spiritpeaksraceway [--tracks FOLDER] [--offsets 0,3,-3]

Reads the track's <track>_collision_grip.csv.gz (tools/track_triangles.py: each colliding surface's
FRICTION in surfaces.ini over the main road's) and walks the AI line, looking straight down at
each point and at --offsets metres to its left (+) and right (-). Prints each stretch of the lap
where the ground under any of them grips under 0.98, as lap progress (0 - 1, the plugin
telemetry's LapProgress) and metres, with the lowest grip found there.
"""
import argparse
import os
import sys

import numpy as np

import fast_sim as fs
import track_files as tf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', default=tf.TRACK)
    ap.add_argument('--tracks', default=tf.TRACKS_DIR)
    ap.add_argument('--offsets', default='0,3,-3', help='metres beside the line to look at too (+ = left)')
    ap.add_argument('--below', type=float, default=0.98, help='grip that counts as different')
    args = ap.parse_args(argv)
    if not os.path.exists(tf.path(args.tracks, 'collision_grip', args.track)):
        sys.exit(f"{tf.path(args.tracks, 'collision_grip', args.track)} not found: setup_track.py makes it")
    w = fs.load_world(args.tracks, track=args.track)
    ground = w.tri_grip[w.n_sensor:][w.tri_surf[w.n_sensor:] != fs.SURF_WALL]
    print(f"{args.track}: ground grip {ground.min():.3f} - {ground.max():.3f}; "
          f"{np.mean(ground < args.below):.1%} of the ground triangles under {args.below:g}")
    offsets = [float(o) for o in args.offsets.split(',') if o.strip()]
    tri_v, tri_surf, gmin, cell, dims, cs, ci = w.ray_arrays()
    n = len(w.sp_pos)
    grip = np.ones(n)
    for i in range(n):
        f = w.sp_fwd[i]
        norm = float(np.hypot(f[0], f[2])) or 1.0
        left = (-f[2] / norm, f[0] / norm)      # heading atan2(z, x): left is +90 deg
        for o in offsets:
            x = w.sp_pos[i, 0] + left[0] * o
            z = w.sp_pos[i, 2] + left[1] * o
            g = fs.ground_grip(x, w.sp_pos[i, 1] + 3.0, z, 12.0, tri_v, tri_surf, w.tri_grip, gmin, cell, dims, cs, ci)
            if g >= 0.0:
                grip[i] = min(grip[i], g)
    low = grip < args.below
    lap = w.lap_length
    if not low.any():
        print(f"the ground along the AI line (offsets {args.offsets} m) grips {args.below:g} or more everywhere")
        return
    i = 0
    print(f"stretches of the AI line gripping under {args.below:g} (offsets {args.offsets} m):")
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        print(f"  lap progress {w.sp_cum[i] / lap:.3f} - {(w.sp_cum[j] + w.sp_length[j]) / lap:.3f} "
              f"({w.sp_cum[i]:.0f} - {w.sp_cum[j] + w.sp_length[j]:.0f} m): lowest grip {grip[i:j + 1].min():.3f}")
        i = j + 1


if __name__ == '__main__':
    main()
