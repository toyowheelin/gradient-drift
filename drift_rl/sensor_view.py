"""
sensor_view.py - makes what the bot's sensors see on a track: the walls the game collides with,
plus the course's tire stacks, cones and barrels (which the game does not collide with).

    python sensor_view.py --track <track> [--track ...] [--tracks <folder>]
    python sensor_view.py --all [--tracks <folder>]

At first the sensors saw the track's model as it is drawn. On Drift Playground the model
has a wedge-shaped object 5 m tall on the grass inside a hairpin where the game has only a thin
barrier, and gen5, steering by a wall the car could never hit, cut the hairpin in 44% of runs
(3% with the sensors seeing only solid walls). The drawn model is kept as <track>_visual.csv.gz,
which build_course.py reads to find the markers; <track>_triangles.csv.gz becomes the collision
file's walls plus an upright prism for each marker. Run again after build_course.py; a track
already done is made again from its visual file, so this is safe to repeat. setup_track.py runs
it as its last step; the generated courses are made this way already.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))


def prism(cx, y0, cz, r, h, sides=8):
    """Triangles of an upright prism (a tire stack) standing at height y0: its sides and its top."""
    import numpy as np
    a = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.c_[cx + r * np.cos(a), cz + r * np.sin(a)]
    out = []
    for (x0, z0), (x1, z1) in zip(ring, np.roll(ring, -1, 0)):
        out.append((x0, y0, z0, x1, y0, z1, x1, y0 + h, z1, 'Wall'))
        out.append((x0, y0, z0, x1, y0 + h, z1, x0, y0 + h, z0, 'Wall'))
        out.append((cx, y0 + h, cz, x0, y0 + h, z0, x1, y0 + h, z1, 'Wall'))
    return out


def make(tracks_dir, track):
    """Writes <track>_triangles.csv.gz from the collision file and the markers; the first time,
    the drawn model's triangles move to <track>_visual.csv.gz. Returns (walls, markers)."""
    sensors, visual = tf.path(tracks_dir, 'triangles', track), tf.path(tracks_dir, 'visual', track)
    if not os.path.exists(visual):
        if not os.path.exists(sensors):
            raise FileNotFoundError(sensors)
        shutil.move(sensors, visual)
    collision = pd.read_csv(tf.path(tracks_dir, 'collision', track))
    walls = collision[collision['SurfaceType'] == 'Wall']
    rows = []
    markers_path = tf.path(tracks_dir, 'course_markers', track)
    markers = pd.read_csv(markers_path) if os.path.exists(markers_path) else pd.DataFrame()
    for m in markers.itertuples():
        rows += prism(m.X, m.GroundY, m.Z, m.RadiusM, m.HeightM)
    out = pd.concat([walls, pd.DataFrame(rows, columns=collision.columns)], ignore_index=True)
    tmp = sensors + '.tmp'
    out.to_csv(tmp, index=False, float_format='%.4f', compression='gzip')
    os.replace(tmp, sensors)
    return len(walls), len(markers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--track', action='append', default=[])
    ap.add_argument('--all', action='store_true', help="every track folder that has a collision file")
    args = ap.parse_args(argv)
    tracks = list(args.track)
    if args.all:
        tracks += sorted(t for t in os.listdir(args.tracks)
                         if os.path.exists(tf.path(args.tracks, 'collision', t)) and t not in tracks)
    if not tracks:
        sys.exit("give --track <track> or --all")
    for t in tracks:
        walls, markers = make(args.tracks, t)
        print(f"{t}: sensors see {walls} wall triangles and {markers} markers")


if __name__ == '__main__':
    main()
