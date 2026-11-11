"""gradient-drift-setup on the command line.

    gradient-drift-setup                  the walk-through: in a window where one can open (Windows),
                                          else the same five steps as questions in the terminal
    gradient-drift-setup walk             the walk-through in the terminal (Windows and Linux alike)
    gradient-drift-setup check <server folder> [<server folder> ...] [--game <AC install>]
        [--bots N] [--car MODEL[,MODEL...]] [--skin NAME[,NAME...]] [--name NAME] [--pace PERCENT]
        [--plugin FOLDER] [--track-data FOLDER] [--shipped-policy] [--details]
    gradient-drift-setup apply <server folder> [the same options] [--yes]
    gradient-drift-setup undo <server folder> [--yes]
    gradient-drift-setup bundle <server folder> [--yes]
    gradient-drift-setup export <server folder> --out <empty folder> [the same options]
    gradient-drift-setup zones <server folder or track> [list | add | move | delete | kind | side | auto |
        picture | score ...] [--track-data FOLDER]

check reads each server and prints what it needs for drift bots, changing nothing (it exits 1 when
a server has something setup cannot fix). apply shows the same list, asks, and makes the changes
(originals kept in <server>/gradient-drift-setup-backups/). undo puts back what the last apply
changed. bundle only packs the plugin's loose data files (policy, car files, track data) into
plugins/GradientDriftPlugin/GradientDriftPlugin.bundle.zip, which apply also does. export is for a server setup cannot reach (a hosting plan): give it the server's cfg/
folder downloaded from the host, and it puts the files to upload in <out>/upload/ and what to do
with them in <out>/HOW_TO_INSTALL.txt, changing nothing it was given. zones shows a track's drift zones
(where the trainer pays the bots to put the rear of the car) in the zone editor, or changes them with
the words after it (`zones <server> help` lists them). Without --game setup uses the
Assetto Corsa install it finds. The walk-through prints its run as one of these lines at the end.

Example:
    gradient-drift-setup apply "/srv/assetto/server" --bots 3 --pace 90
"""
import argparse
import os

from .apply import apply_plan, undo
from .plan import plan_bot
from .server import read_server


def describe(plan, details=False):
    s = plan.server
    out = [f"\n{s.folder}" + (f"  -  {s.name}" if s.name else '')]
    if not s.problems:
        layout = f" ({s.layout})" if s.layout else ''
        bots = len(s.bots)
        out.append(f"  track {s.track}{layout}; {len(s.entries)} entries, {bots} AI; AI {'on' if s.enable_ai else 'off'}; "
                   f"plugins: {', '.join(s.plugins) or 'none'}")
        if plan.program is not None and plan.program.kind == 'assettoserver':
            out.append(f"  AssettoServer {plan.program.version or '(version not found)'}")
        if plan.slots:
            out.append("  bots' cars: " + '; '.join(
                f"{f'CAR_{i}' if i is not None else 'new'} {model}" + (f" ({look})" if look else '')
                for i, model, look in plan.slots))
    if plan.ready:
        out.append("  READY: nothing to change")
    if plan.changes:
        out.append("  would change:")
        for c in plan.changes:
            out.append(f"    - {c.file}: {c.what}")
            if details:
                for label, text in (('before', c.before), ('after', c.after)):
                    for line in text.splitlines():
                        out.append(f"        {label:6s} | {line}")
    if plan.blockers:
        out.append("  not possible yet:")
        out += [f"    ! {b}" for b in plan.blockers]
    out += [f"  note: {n}" for n in plan.notes]
    return '\n'.join(out)


def bundle_only(folders, yes):
    """The bundle command: each server's plan with nothing but packing its loose data files."""
    from .plan import Plan, plan_bundle
    status = 0
    for folder in folders:
        server = read_server(folder)
        print(f"\n{folder}")
        if server.problems:
            print('\n'.join(f"    ! {p}" for p in server.problems))
            status = 1
            continue
        plan = Plan(server=server)
        plan_bundle(plan, None)
        if not plan.bundle:
            print("  not packed: the server's plugin reads loose files only (or is not installed); update the "
                  "plugin first")
            status = 1
            continue
        if not plan.changes:
            print("  nothing to pack: no loose data files")
            continue
        print(f"  {plan.changes[0].what}")
        if not yes and input("  pack them? [y/N] ").strip().lower() not in ('y', 'yes'):
            print("  nothing changed")
            continue
        apply_plan(plan)
        print("  done. Restart the server for the plugin to read the bundle.")
    return status


ZONES_HELP = """zones <server folder or track> [what] [--track-data FOLDER]

  (nothing)                  the zone editor, in a window where one can open; else the list
  list                       the zones, in order round the lap
  add left|right FROM TO [touch]
                             a new zone on the left or right edge (looking along the direction of
                             travel) from FROM to TO: course point ids, or metres from the start
                             with an m (120m); touch makes it a touch-and-go area
  move N FROM TO             zone N's ends
  delete N                   take zone N out
  kind N outside|touch       outside zone, or touch and go
  side N left|right          move zone N to the other edge, over the same stretch
  auto                       the zones setup places from the course's shape (replaces them all)
  picture FILE.svg [RECORDING [DRIVER]]
                             a map of the zones (opens in a web browser), with a recording's
                             rear-axle path if given
  score RECORDING [DRIVER]   how well a recording fills each zone (AI Car Monitor's, the plugin's
                             telemetry or the game's telemetry recorder; DRIVER picks one of several)

Each change is saved straight away (the file it replaces is kept in the track's zone_backups
folder) and marked as edited, so setup keeps it when it makes the track's data again (auto
takes the mark off). Run setup's apply again to copy them to the server, where the plugin
paints them on the road for players (ShowZones). For the bots, zones change what the trainer pays
for: they drive them only after training on that track."""


def zones_command(target, words, track_data):
    """The zones command: target is a server folder or a track's name."""
    import sys
    from . import zones as Z
    from .find import default_track_data, is_server, track_data_home
    if os.path.isdir(target) and is_server(target):
        server = read_server(target)
        if server.problems:
            print('\n'.join(f"  ! {p}" for p in server.problems))
            return 1
        track = server.track
    else:
        track = os.path.basename(os.path.normpath(target))
    tracks = track_data or default_track_data() or track_data_home()
    words = list(words)
    if words[:1] == ['help']:
        print(ZONES_HELP)
        return 0
    if not Z.has_course(tracks, track):
        print(f"no course data for {track} in {tracks}: setup makes it when it makes the track's data "
              "(apply), or give --track-data")
        return 1
    if not words:
        from .find import gui_possible
        if gui_possible():
            from .zones_gui import run
            return run(tracks, track)
        words = ['list']
    course, zones, edited = Z.load(tracks, track)
    zones = Z.ordered(course, zones)

    def point(text):
        text = text.strip().lower()
        return course.point_at(float(text[:-1])) if text.endswith('m') else int(text) % course.n

    def zone_no(text):
        k = int(text)
        if not 1 <= k <= len(zones):
            raise ValueError(f"there is no zone {k} (1 - {len(zones)})")
        return k - 1

    def side_of(text):
        if text.lower() not in ('left', 'right'):
            raise ValueError('the side is left or right')
        return 1 if text.lower() == 'left' else -1

    def show():
        print(f"{track}: {len(zones)} zone(s) on a {course.lap_m:.0f} m lap, {Z.DEPTH_M:g} m deep; "
              + ('edited by hand' if edited else 'placed automatically by setup' if zones else 'none') + f"\n  {tracks}")
        for i, z in enumerate(zones, 1):
            print(f"  {i:2d}. {Z.describe(course, z)}")
        for p in Z.problems(course, zones):
            print(f"  ! {p}")

    def run_from(rest):
        if not rest:
            return None
        runs = Z.load_runs(rest[0])
        if not runs:
            raise ValueError(f"{rest[0]} has no driving above {Z.MOVING_KMH:g} km/h")
        if len(rest) > 1:
            named = [r for r in runs if r.name.lower() == rest[1].lower()]
            if not named:
                raise ValueError(f"no driver {rest[1]} in it; there are: {', '.join(r.name for r in runs)}")
            return named[0]
        if len(runs) > 1:
            print(f"  {len(runs)} drivers in it; showing {runs[0].name} (the most driving). Name one after the "
                  f"file for another: {', '.join(r.name for r in runs[1:])}")
        return runs[0]

    what, rest = words[0].lower(), words[1:]
    try:
        if what == 'list':
            show()
            return 0
        if what == 'picture':
            if not rest:
                raise ValueError('picture needs a file name (FILE.svg)')
            name = rest[0] if rest[0].lower().endswith('.svg') else os.path.splitext(rest[0])[0] + '.svg'
            out = Z.picture(course, zones, name, run=run_from(rest[1:]))
            print(f"written: {out}")
            return 0
        if what == 'score':
            run = run_from(rest)
            if run is None:
                raise ValueError('score needs a recording')
            rows = Z.score(course, zones, run)
            print(f"{run.name}: share of the time on each zone's stretch with the rear axle in the zone while "
                  f"sliding {Z.SLIDING_DEG:g} deg or more")
            for i, (fill, depth, past) in enumerate(rows, 1):
                print(f"  zone {i:2d}  " + ('not reached' if fill != fill else
                                           f"filled {fill:4.0%}   rear {depth:5.1f} m inside the edge   past it {past:4.0%}"))
            score = Z.line_score(rows)
            print("  line score, Formula Drift style (mean fill x 40): " + (f"{score:.1f} / 40" if score == score else 'none reached'))
            return 0
        if what == 'add':
            if len(rest) < 3:
                raise ValueError('add needs left|right FROM TO')
            kind = Z.KINDS[1] if rest[3:4] and rest[3].lower().startswith('touch') else Z.KINDS[0]
            zones.append(Z.new_zone(course, side_of(rest[0]), point(rest[1]), point(rest[2]), kind))
        elif what == 'move':
            k = zone_no(rest[0])
            zones[k] = Z.move(course, zones[k], first=point(rest[1]), last=point(rest[2]))
        elif what == 'delete':
            zones.pop(zone_no(rest[0]))
        elif what == 'kind':
            k = zone_no(rest[0])
            word = rest[1].lower() if len(rest) > 1 else ''
            if not word.startswith(('outside', 'touch')):
                raise ValueError('the kind is outside or touch')
            zones[k] = Z.move(course, zones[k], kind=Z.KINDS[0] if word.startswith('outside') else Z.KINDS[1])
        elif what == 'side':
            k = zone_no(rest[0])
            zones[k] = Z.move(course, zones[k], side=side_of(rest[1]))
        elif what == 'auto':
            zones = Z.auto(course)
        else:
            print(f"not a zones word: {what}\n\n{ZONES_HELP}", file=sys.stderr)
            return 2
    except (ValueError, IndexError) as e:
        print(f"  not done: {e if str(e) else 'more words needed'}  (zones <track> help lists them)")
        return 1
    path, backup, edited = Z.save(tracks, track, course, zones)
    zones = Z.ordered(course, zones)
    show()
    print(f"saved: {path}" + (f"\n  the file it replaced: {backup}" if backup else ''))
    return 0


def find_game(given):
    """--game, else the install setup finds (said, so a wrong pick is seen), else None."""
    if given:
        return given
    from .find import find_games
    games = find_games()
    if games:
        print(f"using the Assetto Corsa install at {games[0]}" + (" (the first of several found)" if len(games) > 1 else '')
              + "; give --game to use another")
        return games[0]
    print("no Assetto Corsa install found: give --game <the folder with content/ in it>")
    return None


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['selftest']:        # CI's check of the packaged program; not in the help
        from .terminal import HAVE_TOOLKIT
        if not HAVE_TOOLKIT:
            print("prompt_toolkit is missing: the walk-through would fall back to numbered lists with no Tab completion")
            return 1
        print('terminal prompts OK')
        from .trackdata import selftest
        return selftest()
    if argv[:1] == ['walk']:
        from .walk import run
        return run()
    if argv[:1] == ['zones']:           # its own words after the track (zones <track> help)
        zp = argparse.ArgumentParser(prog='gradient-drift-setup zones', description=ZONES_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        zp.add_argument('target', metavar='server_or_track')
        zp.add_argument('words', nargs='*')
        zp.add_argument('--track-data', default=None, help='the track files folder (a folder per track)')
        za = zp.parse_args(argv[1:])
        return zones_command(za.target, za.words, za.track_data)
    if not argv:
        from . import console
        from .find import gui_possible
        clicked = console.double_clicked()
        if clicked:
            console.hide()          # straight away, so the console barely shows
        if gui_possible():
            if clicked:
                console.detach()
            try:
                from .gui import run
                return run()
            except Exception as e:
                if clicked:
                    console.error_box(f'Setup could not open its window:\n\n{type(e).__name__}: {e}')
                    return 1
                raise
        if clicked:
            console.show()
        if sys.stdin.isatty():
            from .walk import run
            status = run()
            if clicked:
                input("Press Enter to close.")
            return status
        argv = ['--help']
    ap = argparse.ArgumentParser(prog='gradient-drift-setup', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True, metavar='{walk,check,apply,undo,bundle,export,zones}')
    sub.add_parser('walk', help='the walk-through, as questions in the terminal')
    for name, text in (('check', 'what each server needs for drift bots (changes nothing)'),
                       ('apply', 'make those changes, after asking (originals kept)'),
                       ('export', 'for a hosted server: the files to upload and how, from its downloaded cfg/ folder')):
        c = sub.add_parser(name, help=text)
        if name == 'export':
            c.add_argument('servers', nargs=1, metavar='server',
                           help="a folder with the server's cfg/ in it (downloaded from the host)")
            c.add_argument('--out', required=True, help='an empty or new folder for the files to upload')
        else:
            c.add_argument('servers', nargs='+')
        c.add_argument('--game', default=None,
                       help='the Assetto Corsa install (the folder with content/); default: the one setup finds')
        c.add_argument('--car', default=None,
                       help="the bots' car, or one per bot slot as MODEL,MODEL,... (existing AI entries first; "
                            "an empty one keeps that bot's car; bots past the list take its last car)")
        c.add_argument('--skin', default=None, help='the skin, or one per bot slot like --car')
        c.add_argument('--bots', type=int, default=1, help='how many bots in all')
        c.add_argument('--name', default=None,
                       help="the bots' name in game, given to each new bot, numbered when there are several (default: the server's, else DriftBot)")
        c.add_argument('--pace', type=float, default=None, metavar='PERCENT',
                       help="the bots' pace, %% of the flat-out speed at each point of the lap (a policy trained with paces)")
        c.add_argument('--plugin', default=None, help="the folder with the plugin's build (GradientDriftPlugin.dll ...)")
        c.add_argument('--track-data', default=None, help="a folder of track data you already have (a folder per track); leave out and setup makes it from the game")
        c.add_argument('--shipped-policy', action='store_true',
                       help="put the policy that comes with the plugin in place of a server's own one (the same policy on every server)")
        c.add_argument('--details', action='store_true', help='show each change as before / after')
        if name == 'apply':
            c.add_argument('--yes', action='store_true', help="don't ask")
    u = sub.add_parser('undo', help='put back what the last apply changed, after asking')
    u.add_argument('servers', nargs='+')
    u.add_argument('--yes', action='store_true', help="don't ask")
    b = sub.add_parser('bundle', help="pack the plugin's loose data files into its one bundle file (originals kept)")
    b.add_argument('servers', nargs='+')
    b.add_argument('--yes', action='store_true', help="don't ask")
    sub.add_parser('zones', help="a track's drift zones: the zone editor, or changes in words (zones <track> help)")
    args = ap.parse_args(argv)
    if args.command == 'bundle':
        return bundle_only(args.servers, args.yes)
    if args.command == 'walk':
        from .walk import run
        return run()
    if args.command == 'undo':
        from .apply import last_run
        for folder in args.servers:
            print(folder)
            run = last_run(folder)
            if run is None:
                print('  nothing to undo')
                continue
            if not args.yes and input(f"  put back the files setup changed in its run of {os.path.basename(run)}? [y/N] "
                                      ).strip().lower() not in ('y', 'yes'):
                print("  nothing changed")
                continue
            undo(folder)
        return 0
    from .find import default_plugin_source, default_track_data
    from .plan import bot_name_now
    from .server import server_running
    from .terminal import run_logged
    cars = (args.car or '').split(',')
    skins = (args.skin or '').split(',')
    if len(cars) > 1 or len(skins) > 1:     # slot by slot
        n = max(len(cars), len(skins))
        cars_args = dict(slots=[(cars[min(i, len(cars) - 1)], skins[i] if i < len(skins) else '') for i in range(n)])
    else:
        cars_args = dict(car=args.car, skin=args.skin)
    plugin = args.plugin or default_plugin_source()
    track_data = args.track_data or default_track_data()
    game = find_game(args.game)
    if args.command == 'export':
        from .export import INSTRUCTIONS, UPLOAD, export_server
        name = args.name or bot_name_now(read_server(args.servers[0]))
        try:
            plan, files = run_logged(lambda log: export_server(
                args.servers[0], args.out, game, log=log, **cars_args, bots=args.bots, bot_name=name,
                plugin_source=plugin, track_data=track_data, pace_percent=args.pace,
                shipped_policy=args.shipped_policy))
        except ValueError as e:
            print(f"  not exported: {e}")
            return 1
        print(describe(plan, args.details))
        if files is None:
            print("  not exported: " + ("fix the problems above first" if plan.blockers or plan.server.problems
                                         else "the server needs no changes"))
            return 1 if plan.blockers or plan.server.problems else 0
        print(f"  exported {len(files)} file(s) to {os.path.join(args.out, UPLOAD)}")
        print(f"  next: follow {os.path.join(args.out, INSTRUCTIONS)}")
        return 0
    status = 0
    for folder in args.servers:
        server = read_server(folder)
        plan = plan_bot(server, game, **cars_args, bots=args.bots, bot_name=args.name or bot_name_now(server),
                        plugin_source=plugin, track_data=track_data,
                        pace_percent=args.pace, shipped_policy=args.shipped_policy)
        print(describe(plan, args.details))
        if plan.blockers or plan.server.problems:
            status = 1
        if args.command != 'apply' or not plan.changes:
            continue
        if plan.blockers:
            print("  not applied: fix the problems above first (setup cannot make a server work around them)")
            continue
        running = server_running(folder)
        if running and any((c.op or ())[:1] == ('copy',) and c.op[3:4] == ('plugin',) for c in plan.changes):
            print("  not applied: AssettoServer is running from this folder (the plugin is in use): stop it, then run "
                  "setup again. Nothing was changed.")
            status = 1
            continue
        slow = any((c.op or ())[:1] == ('track_data',) for c in plan.changes)
        print("  " + ("AssettoServer looks to be running from this folder: restart it afterwards." if running else
                      "Stop the server first if it is running, and start it again afterwards.")
              + " Setup keeps the originals, and undo puts them back."
              + (" Making the track's data takes a few minutes." if slow else '')
              + ("" if args.details else " --details shows each change before and after."))
        if not args.yes and input(f"  make these {len(plan.changes)} change(s)? [y/N] ").strip().lower() not in ('y', 'yes'):
            print("  nothing changed")
            continue
        try:
            run_logged(lambda log: apply_plan(plan, log=log))
        except RuntimeError as e:
            print(f"  not applied: {e}")
            status = 1
            continue
        print('  done. Start the server again: its log should say "GradientDrift: policy loaded from"; '
              'the bots join when a player is on.')
    return status
