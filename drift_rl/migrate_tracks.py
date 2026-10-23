"""
migrate_tracks.py - copies track data from the old layout (closed_loop_sweep_bundle/data/, every
track's files side by side) into the new one (drift_rl/tracks/<track>/, one folder per track).

    python migrate_tracks.py                         # ../closed_loop_sweep_bundle/data -> tracks/
    python migrate_tracks.py --from <old data folder> [--tracks <new folder>]

Files are copied, never moved or deleted: the old folder stays as it was until you delete it
yourself. A file already in the new place is left alone (and reported if it differs). Spirit
Peaks' unprefixed fast_lane.ai becomes spiritpeaksraceway_fast_lane.ai. Anything that is not a
track file (old telemetry, pictures) is left where it is.
"""
import argparse
import filecmp
import os
import shutil
import sys

import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIXES = sorted(list(tf.KINDS.values()) + list(tf.OPTIONAL.values()), key=len, reverse=True)


def track_of(filename):
    """(track, kind file name) for a track file, else None."""
    if filename == 'fast_lane.ai':
        return tf.TRACK, tf.name('fast_lane', tf.TRACK)
    for suffix in SUFFIXES:
        if filename.endswith('_' + suffix) and len(filename) > len(suffix) + 1:
            return filename[:-len(suffix) - 1], filename
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='source', default=os.path.join(_HERE, '..', 'closed_loop_sweep_bundle', 'data'))
    tf.tracks_arg(ap)
    args = ap.parse_args()
    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        sys.exit(f"no folder {source}")
    copied, kept, differ, tracks = 0, 0, [], set()
    names = sorted(os.listdir(source))
    if 'fast_lane.ai' in names and tf.name('fast_lane', tf.TRACK) in names:
        names.remove('fast_lane.ai')            # the prefixed copy wins
    for filename in names:
        found = track_of(filename)
        if found is None or not os.path.isfile(os.path.join(source, filename)):
            continue
        track, target_name = found
        target = os.path.join(tf.folder(args.tracks, track), target_name)
        tracks.add(track)
        if os.path.exists(target):
            if filecmp.cmp(os.path.join(source, filename), target, shallow=False):
                kept += 1
            else:
                differ.append(target)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(source, filename), target)
        copied += 1
    print(f"{len(tracks)} tracks: {copied} files copied into {os.path.abspath(args.tracks)}, "
          f"{kept} already there")
    for t in differ:
        print(f"  left as it is (it differs from the old copy): {t}")
    print(f"The old folder {source} is unchanged; delete it yourself once training works from the new one.")


if __name__ == '__main__':
    main()
