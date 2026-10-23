"""
build_road_mesh.py - writes <track>_road_mesh.csv, the road-height points the plugin and the trainer
use for the car's ride height where its ground ray finds nothing, from the Road triangles of the
track's collision surfaces (<track>_collision.csv.gz, from tools/track_triangles.py).

    python build_road_mesh.py --track <name> [--cell 0.5]

One point per cell of the road surface (X/Z grid, default 0.5 m): the mean of the Road triangles'
corners in it. The same file has to be in the server folder for the plugin.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.tracks_arg(ap)
    ap.add_argument('--track', default=None, help='the track folder name (default: track_files.TRACK)')
    ap.add_argument('--cell', type=float, default=0.5, help='grid cell, metres')
    args = ap.parse_args(argv)
    sys.path.insert(0, _HERE)
    track = args.track or tf.TRACK
    data = args.tracks
    t = pd.read_csv(tf.path(data, 'collision', track))
    road = t[t.SurfaceType == 'Road']
    if road.empty:
        sys.exit(f"no Road triangles in {tf.path(data, 'collision', track)}")
    v = np.concatenate([road[[f'X{k}', f'Y{k}', f'Z{k}']].to_numpy() for k in (1, 2, 3)])
    key = np.floor(v[:, [0, 2]] / args.cell).astype(np.int64)
    cells = pd.DataFrame({'kx': key[:, 0], 'kz': key[:, 1], 'X': v[:, 0], 'Y': v[:, 1], 'Z': v[:, 2]})
    points = cells.groupby(['kx', 'kz'])[['X', 'Y', 'Z']].mean()
    out = tf.writable_path(data, 'road_mesh', track)
    points.to_csv(out, index=False)
    print(f"{len(road)} Road triangles -> {len(points)} points (one per {args.cell:g} m cell), "
          f"height {points.Y.min():.2f} - {points.Y.max():.2f} m\nwritten: {out}")


if __name__ == '__main__':
    main()
