"""gradient-drift-setup on the command line.

    python -m gradientdrift_setup                  the walk-through, in a window (Windows, or Linux with a desktop)
    python -m gradientdrift_setup check <server folder> [<server folder> ...] --game <AC install>
        [--car MODEL[,MODEL...]] [--skin NAME[,NAME...]] [--bots N] [--train-size N] [--pace PERCENT] [--plugin FOLDER] [--track-data FOLDER] [--details]
    python -m gradientdrift_setup apply <server folder> --game <AC install> [the same options] [--yes]
    python -m gradientdrift_setup undo <server folder>
    python -m gradientdrift_setup export <server folder> --out <empty folder> --game <AC install> [the same options]

check reads each server and prints what it needs for drift bots, changing nothing. apply shows the
same list, asks, and makes the changes (originals kept in <server>/gradient-drift-setup-backups/). undo
puts back what the last apply changed. export is for a server setup cannot reach (a hosting plan):
give it the server's cfg/ folder downloaded from the host, and it puts the files to upload in
<out>/upload/ and what to do with them in <out>/HOW_TO_INSTALL.txt, changing nothing it was given.
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


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
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
        print("No window can be opened here (no display, or Qt is not installed); using the command line.\n")
        if clicked:
            print(__doc__)
            input("Press Enter to close.")
            return 0
        argv = ['--help']
    ap = argparse.ArgumentParser(prog='gradient-drift-setup', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
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
        c.add_argument('--game', default=None, help='the Assetto Corsa install (the folder with content/)')
        c.add_argument('--car', default=None,
                       help="the bots' car, or one per bot slot as MODEL,MODEL,... (existing AI entries first; "
                            "an empty one keeps that bot's car; bots past the list take its last car)")
        c.add_argument('--skin', default=None, help='the skin, or one per bot slot like --car')
        c.add_argument('--bots', type=int, default=1, help='how many bots in all')
        c.add_argument('--train-size', type=int, default=None,
                       help='bots to a tandem train (1: every bot on its own; default: leave the server as it is)')
        c.add_argument('--pace', type=float, default=None, metavar='PERCENT',
                       help="the bots' pace, %% of the flat-out speed at each point of the lap (a policy trained with paces)")
        c.add_argument('--plugin', default=None, help="the folder with the plugin's build (GradientDriftPlugin.dll ...)")
        c.add_argument('--track-data', default=None, help="a folder of track data you already have (a folder per track); leave out and setup makes it from the game")
        c.add_argument('--details', action='store_true', help='show each change as before / after')
        if name == 'apply':
            c.add_argument('--yes', action='store_true', help="don't ask")
    u = sub.add_parser('undo', help='put back what the last apply changed')
    u.add_argument('servers', nargs='+')
    sub.add_parser('selftest', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.command == 'selftest':
        from .trackdata import selftest
        return selftest()
    if args.command == 'undo':
        for folder in args.servers:
            print(folder)
            undo(folder)
        return 0
    from .find import default_plugin_source, default_track_data
    cars = (args.car or '').split(',')
    skins = (args.skin or '').split(',')
    if len(cars) > 1 or len(skins) > 1:     # slot by slot
        n = max(len(cars), len(skins))
        cars_args = dict(slots=[(cars[min(i, len(cars) - 1)], skins[i] if i < len(skins) else '') for i in range(n)])
    else:
        cars_args = dict(car=args.car, skin=args.skin)
    plugin = args.plugin or default_plugin_source()
    track_data = args.track_data or default_track_data()
    if args.command == 'export':
        from .export import INSTRUCTIONS, UPLOAD, export_server
        try:
            plan, files = export_server(args.servers[0], args.out, args.game, **cars_args,
                                        bots=args.bots, plugin_source=plugin, track_data=track_data,
                                        train_size=args.train_size, pace_percent=args.pace)
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
        plan = plan_bot(read_server(folder), args.game, **cars_args, bots=args.bots,
                        plugin_source=plugin, track_data=track_data, train_size=args.train_size,
                        pace_percent=args.pace)
        print(describe(plan, args.details or args.command == 'apply'))
        if args.command != 'apply' or not plan.changes:
            continue
        if plan.blockers:
            print("  not applied: fix the problems above first (setup cannot make a server work around them)")
            status = 1
            continue
        if not args.yes and input(f"  make these {len(plan.changes)} change(s)? [y/N] ").strip().lower() not in ('y', 'yes'):
            print("  nothing changed")
            continue
        try:
            apply_plan(plan)
        except RuntimeError as e:
            print(f"  not applied: {e}")
            status = 1
            continue
        print("  done. Restart the server for the changes to take effect.")
    return status
