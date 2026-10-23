"""Where each track's files live: drift_rl/tracks/<track>/, one folder per track.

TRACK is a track's folder name under content/tracks/ (the default one). The policy file records
the track it was trained on, and the plugin drives a policy only on the track it names.

The files keep the track's name in front (<track>_triangles.csv.gz and so on) even inside the
track's own folder: those are the names the plugin looks for in the server folder, so they are
copied there as they are, and a file that ends up somewhere on its own still says which track
it belongs to. No track data ships: each user makes their own (setup_track.py) from their own
copy of the game. migrate_tracks.py copies an older closed_loop_sweep_bundle/data/ folder in.
"""
import os
import re

# the track the tools fall back on when no --track is given (the first one trained on)
TRACK = 'spiritpeaksraceway'
TRACKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks')

KINDS = {
    'fast_lane': 'fast_lane.ai',            # the track's AI line, from content/tracks/<track>/ai/
    'triangles': 'triangles.csv.gz',        # what the sensors see, sensor_view.py
    'collision': 'collision.csv.gz',        # what the game collides with, tools/track_triangles.py
    'road_mesh': 'road_mesh.csv',           # the road-mesh points the plugin reads
    'course_edges': 'course_edges.csv',     # build_course.py
    'course_markers': 'course_markers.csv', # build_course.py
    'course_zones': 'course_zones.csv',     # build_zones.py
}

# Each course's reference speed per AI-line point (drift_env.course_pace_reference): what a pace of
# 1.0 asks for (pace_model). Made by the trainer or setup the first time a course needs it; the plugin
# reads it from the server folder for a policy that reads its pace error. Not one of KINDS: a
# course is complete without it.
PACE_REFERENCE = 'pace_reference.csv'
# What the track's model shows (tools/track_triangles.py's sensor triangles as they come from the
# model): where build_course.py finds the course's tire stacks and cones. The sensors themselves
# see <track>_triangles.csv.gz, which sensor_view.py makes from what the game collides with plus
# those markers. Stays in the track's folder; the server does not need it.
OPTIONAL = {'pace_reference': PACE_REFERENCE, 'visual': 'visual.csv.gz'}


def name(kind, track=TRACK):
    """The file's name, here and in the server folder."""
    return f'{track}_{KINDS.get(kind) or OPTIONAL[kind]}'


def folder(tracks_dir, track=TRACK):
    return os.path.join(tracks_dir, track)


def path(tracks_dir, kind, track=TRACK):
    return os.path.join(folder(tracks_dir, track), name(kind, track))


def writable_path(tracks_dir, kind, track=TRACK):
    """path(), with the track's folder made if it is not there yet."""
    os.makedirs(folder(tracks_dir, track), exist_ok=True)
    return path(tracks_dir, kind, track)


def pace_reference_path(tracks_dir, track=TRACK):
    return path(tracks_dir, 'pace_reference', track)


def game_path(base, *parts):
    """base joined with parts, each part matched ignoring capitals when the exact name is missing.
    Game content is made on Windows, where AI/ and ai/ or Track.KN5 and track.kn5 are the same file;
    on Linux (a Steam / Proton install) they are not. A part that is not there under any capitals
    leaves the plain join, so a caller's exists() check and messages work as before."""
    names = [n for part in parts for n in re.split(r'[\\/]', part) if n]
    path = base
    for i, n in enumerate(names):
        exact = os.path.join(path, n)
        if os.path.exists(exact):
            path = exact
            continue
        try:
            match = next((e for e in sorted(os.listdir(path)) if e.lower() == n.lower()), None)
        except OSError:
            match = None
        if match is None:
            return os.path.join(path, *names[i:])
        path = os.path.join(path, match)
    return path


def tracks_arg(parser):
    """The --tracks option every tool takes."""
    parser.add_argument('--tracks', default=TRACKS_DIR,
                        help='the folder with one folder of data per track (default: drift_rl/tracks)')
