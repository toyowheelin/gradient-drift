"""
setup_track.py - everything the trainer and the plugin need for one track, in one command.

    python setup_track.py <track folder> [--layout NAME] [--tracks FOLDER] [--line auto|keep|rebuild]
                          [--server FOLDER] [--game-surfaces FILE]

<track folder> is the track's folder in the game (content/tracks/<track>). Writes into the track's
own folder, tracks/<track>/ (track_files.py), all named after the track folder:

1. <track>_collision.csv.gz and <track>_visual.csv.gz, the track's geometry: what the game collides
   with and what its model shows (tools/track_triangles.py)
2. <track>_road_mesh.csv (build_road_mesh.py)
3. <track>_fast_lane.ai, the AI line: the track's own (its ai/fast_lane.ai, or the recorded
   fast_lane.ai.candidate when that is all there is), checked and, if it needs it, rebuilt down the
   middle of the road (line_tools.py). --line keep never rebuilds it, --line rebuild always does.
4. <track>_course_edges.csv and _course_markers.csv (build_course.py), <track>_course_zones.csv
   (build_zones.py)
5. <track>_triangles.csv.gz, what the sensors see: the walls the game collides with and the
   course's markers (sensor_view.py)

6. a check of what the data asks of the bots (check_track.py): the speed targets, if there are any yet,
   and the walls beside the course, low ones the policy's sensors cannot feel included (--policy:
   the policy that will drive it; the simulator's sensors without). It stops on a problem unless
   --skip-check.

With --server (an AssettoServer folder) it also copies the files the plugin reads into it (and the
pace reference, once training has made one) and,
if the line was rebuilt, the line to <server>/content/tracks/<track>/[<layout>/]ai/fast_lane.ai,
keeping the one there as fast_lane.ai.before_setup. Nothing is ever deleted.

Then train with:  python train_ppo_gpu.py --track <track> ...
"""
import argparse
import os
import shutil
import sys
import time
import track_files as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def tools_module(name):
    """A module from tools/ beside drift_rl/ (or already importable, as in the packaged setup)."""
    import importlib
    for folder in ('tools', 'kn5_tools'):
        path = os.path.abspath(os.path.join(_HERE, '..', folder))
        if os.path.exists(os.path.join(path, name + '.py')) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.exit(f"{name}.py not found next to drift_rl/ (in tools/ or kn5_tools/)")


def find_line(track, layout):
    """(path, recorded) of the track's AI line: the layout's, then the track's; a .candidate only
    when there is no fast_lane.ai."""
    folders = ([(layout, 'ai')] if layout else []) + [('ai',)]
    for suffix, recorded in (('', False), ('.candidate', True)):
        for folder in folders:
            path = tf.game_path(track, *folder, 'fast_lane.ai' + suffix)
            if os.path.exists(path):
                return path, recorded
    return None, False


def run(module, args):
    """Runs a step's own command line in this process (so it also works inside the packaged setup)."""
    import importlib
    print(f'  $ {module} ' + ' '.join(args), flush=True)
    mod = module if not isinstance(module, str) else importlib.import_module(module)
    mod.main(list(args))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track_folder')
    ap.add_argument('--layout', default=None)
    ap.add_argument('--name', default=None, help='file name prefix (default: the track folder name)')
    tf.tracks_arg(ap)
    ap.add_argument('--line', choices=['auto', 'keep', 'rebuild'], default='auto')
    ap.add_argument('--server', default=None, help='an AssettoServer folder to copy the plugin\'s files into')
    ap.add_argument('--game-surfaces', default=None, help="the game's system/data/surfaces.ini")
    ap.add_argument('--policy', default=None, help="the policy that will drive it, for the check of its walls (its sensors' height)")
    ap.add_argument('--skip-check', action='store_true', help="don't stop when the track's speed targets or walls look wrong")
    ap.add_argument('--for-setup', action='store_true', help=argparse.SUPPRESS)   # setup copies the line itself and checks
    args = ap.parse_args(argv)
    track = os.path.normpath(args.track_folder)
    name = args.name or os.path.basename(track)
    data = os.path.abspath(args.tracks)
    os.makedirs(tf.folder(data, name), exist_ok=True)
    t0 = time.time()

    print(f'1. geometry of {name}')
    cmd = [track, tf.folder(data, name), '--name', name]
    if args.layout:
        cmd += ['--layout', args.layout]
    if args.game_surfaces:
        cmd += ['--game-surfaces', args.game_surfaces]
    run(tools_module('track_triangles'), cmd)
    shutil.copyfile(tf.path(data, 'triangles', name), tf.path(data, 'visual', name))   # the drawn model (sensor_view.py)

    print('2. road mesh')
    run('build_road_mesh', ['--tracks', data, '--track', name])

    print('3. AI line')
    source, recorded = find_line(track, args.layout)
    if source is None:
        sys.exit(f"no ai/fast_lane.ai (or fast_lane.ai.candidate) in {track}"
                 + (f" or its layout {args.layout}" if args.layout else ""))
    line_path = tf.path(data, 'fast_lane', name)
    shutil.copyfile(source, line_path)
    print(f'  from {source}' + (' (a recorded lap: the track has no fast_lane.ai)' if recorded else ''))
    import numpy as np
    import fast_sim as fs
    import line_tools as lt
    pos, sgb = lt.read_line(line_path)
    rep = lt.check(pos)
    world = fs.load_world(data, track=name)
    cen = lt.centring(world, pos)
    print(f"  {rep['points']} points, {rep['length_m']:.0f} m; spacing median {rep['spacing_median']:.2f} m, "
          f"largest {rep['spacing_max']:.2f} m; sharpest turn {rep['turn_max_deg']:.1f} deg a point")
    print(f"  off the middle of the paving: median {cen['off_centre_median_m']:.2f} m "
          f"(where it is under {lt.PAD_WIDTH_M:g} m wide: {cen['narrow_share']:.0%} of the lap); "
          f"narrowest room to a wall or edge {cen['room_min_m']:.1f} m")
    reasons = list(rep['problems'])
    if cen['narrow_share'] > 0.5 and cen['off_centre_median_m'] > 1.0:
        reasons.append(f"runs {cen['off_centre_median_m']:.1f} m off the middle of the road (median)")
    if recorded and not reasons:
        reasons.append('a recorded lap, not the track\'s own AI line')
    rebuilt = args.line == 'rebuild' or (args.line == 'auto' and reasons)
    for r in reasons:
        print(f'  - {r}')
    if rebuilt:
        print('  rebuilding it' + (' (--line rebuild)' if args.line == 'rebuild' else ''))
        pos, sgb, left, right = lt.rebuild(world, pos, sgb, log=print)
        lt.write_line(line_path, pos, sgb, left, right)
        rep = lt.check(pos)
        world = fs.load_world(data, track=name)
        cen = lt.centring(world, pos)
        print(f"  now {rep['points']} points, {rep['length_m']:.0f} m; spacing {rep['spacing_median']:.2f} - "
              f"{rep['spacing_max']:.2f} m; sharpest turn {rep['turn_max_deg']:.1f} deg; off the middle "
              f"{cen['off_centre_median_m']:.2f} m; narrowest room {cen['room_min_m']:.1f} m")
    elif reasons:
        print('  kept as it is (--line keep)')
    else:
        print('  good as it is')

    print('4. course and zones')
    run('build_course', ['--tracks', data, '--track', name])
    run('build_zones', ['--tracks', data, '--track', name])

    print('5. what the sensors see')
    run('sensor_view', ['--tracks', data, '--track', name])
    world = fs.load_world(data, track=name)
    fs.require_road_mesh(world, data, name)
    fs.require_course(world, data, name)

    if not args.for_setup:
        print('6. checking it')
        import check_track
        problems = check_track.check(data, name, args.policy, log=lambda line: print(f'  {line}'))
        if problems and not args.skip_check:
            sys.exit(check_track.report(name, problems) + "\n(--skip-check goes on anyway)")
        for p in problems:
            print(f'  - {p} (--skip-check: going on)')

    if args.server:
        print(f'7. into the server at {args.server}')
        kinds = ['triangles', 'collision', 'road_mesh', 'course_edges']
        if os.path.exists(tf.path(data, 'pace_reference', name)):
            kinds.append('pace_reference')
        for kind in kinds:
            shutil.copyfile(tf.path(data, kind, name), os.path.join(args.server, tf.name(kind, name)))
            print(f'  {tf.name(kind, name)}')
        if rebuilt:
            ai = os.path.join(args.server, 'content', 'tracks', os.path.basename(track), *([args.layout] if args.layout else []), 'ai')
            os.makedirs(ai, exist_ok=True)
            target = os.path.join(ai, 'fast_lane.ai')
            if os.path.exists(target) and not os.path.exists(target + '.before_setup'):
                shutil.copyfile(target, target + '.before_setup')
            shutil.copyfile(line_path, target)
            print(f'  {target}')

    print(f'done in {time.time() - t0:.0f} s. Train with: python train_ppo_gpu.py --track {name} ...')
    if rebuilt and not args.server and not args.for_setup:
        print(f'The line was rebuilt: the server needs it too, as content/tracks/{os.path.basename(track)}/'
              f'{args.layout + "/" if args.layout else ""}ai/fast_lane.ai (copy {line_path}).')
    return {'name': name, 'line': line_path, 'line_rebuilt': bool(rebuilt)}


if __name__ == '__main__':
    main()
