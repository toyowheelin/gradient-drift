"""The walk-through in a terminal: the window's five pages (gui.py) as questions, the same on
Windows and Linux. Everything it shows comes from the core (find, server, game, plan), as in the
window; this file only asks and prints. It ends with the same run as one command line."""
import os
import shlex
import subprocess
import sys

from . import terminal as t
from .apply import apply_plan, last_run, undo
from .find import (default_plugin_source, default_track_data, find_games, find_servers, in_game_content, is_game,
                   is_server, track_data_root)
from .game import car_facts, track_facts
from .identify import identify
from .plan import bot_name_now, bot_slots, plan_bot
from .server import read_server

STEPS = 5
OTHER = object()          # "type a folder" in a pick list


def _sep():
    return '\\' if sys.platform == 'win32' else '/'


def server_line(path):
    s = read_server(path)
    if s.problems:
        return f'{path}   -   {s.problems[0]}'
    kunos = identify(path).kind == 'kunos'
    return (f'{s.name or os.path.basename(path)} - {s.track}' + (f' ({s.layout})' if s.layout else '')
            + (" - the game's own server, not AssettoServer" if kunos else '') + f'   {path}')


def pick_folder(question, found, check, refusal, label=str, current=None):
    """One of the folders found, or one typed (with Tab completion)."""
    options = [(f, label(f)) for f in found] + [(OTHER, 'Another folder: type it')]
    pick = t.choose(question, options, default=current if current in found else None) if found else OTHER
    if pick is OTHER:
        return t.ask_path(question if not found else 'Folder:', check, refusal, default=current or '')
    return pick


# --- 1. the game ---------------------------------------------------------------------------------

def game_step(st):
    t.heading(1, STEPS, 'Where is Assetto Corsa installed?')
    t.hint('Setup reads the track and the cars from the game. It changes nothing in it.')
    t.hint(f'This is the game you play, not the server. With Steam it is usually steamapps{_sep()}common'
           f'{_sep()}assettocorsa\nin your Steam library. The server\'s track and cars must be installed here too.')
    games = find_games()
    if not games:
        t.say(f'     {t.WARN} Setup did not find it in the usual places.')
    st['game'] = pick_folder('The Assetto Corsa folder (the one with content):', games, is_game,
                             lambda p: f'{p} is not an Assetto Corsa install (no content{_sep()}tracks and content{_sep()}cars in it).',
                             current=st.get('game'))


# --- 2. the server -------------------------------------------------------------------------------

def server_facts(path):
    s = read_server(path)
    if s.problems:
        return s, [f'{t.BAD} ' + '; '.join(s.problems)]
    prog = identify(path)
    if prog.kind == 'kunos':
        program = f'{t.BAD} {prog.note}'
    elif prog.kind == 'assettoserver':
        program = f'{t.OK} AssettoServer {prog.version or "(version not found)"}'
    else:
        program = f'{t.WARN} {prog.note}'
    layout = f' ({s.layout})' if s.layout else ''
    return s, [f'{s.name or os.path.basename(path)}', program, f'Track: {s.track}{layout}',
               f'{len(s.entries)} car slots, {len(s.bots)} AI; AI is {"on" if s.enable_ai else "off"}; '
               f'plugins: {", ".join(s.plugins) or "none"}']


def server_step(st):
    t.heading(2, STEPS, 'Which server gets the drift bot?')
    t.hint(f'Pick your AssettoServer folder (the one with cfg{_sep()}server_cfg.ini). For a server on a hosting plan,\n'
           'download its cfg folder into a folder here and pick that; the last step then exports the\n'
           'files to upload.')
    t.hint('Start the server once before setup if you have not yet, so AssettoServer\n'
           f'writes cfg{_sep()}extra_cfg.yml. Nothing is changed until the last step.')
    found = find_servers([st['game'], os.getcwd()] + st.get('server_hints', []))
    while True:
        path = pick_folder('The server:', found, is_server,
                           lambda p: f'{p} has no cfg{_sep()}server_cfg.ini: not a server folder.',
                           label=server_line, current=st['server'].folder if st.get('server') else None)
        s, lines = server_facts(path)
        for line in lines:
            t.say(f'     {line}')
        if s.problems or identify(path).kind == 'kunos':
            t.say('     Pick another server.')
            continue
        st['server'] = s
        return


# --- 3. the track --------------------------------------------------------------------------------

def track_step(st):
    t.heading(3, STEPS, 'Checking the track')
    t.hint("Is the server's track in the game install, with what the bots need? Nothing to choose here.")
    s = st['server']
    facts = track_facts(st['game'], s.track, s.layout)
    layout = f' ({s.layout})' if s.layout else ''
    t.say(f'     {s.track}{layout}')
    if not facts.exists:
        t.say(f'     {t.BAD} The game install has no content{_sep()}tracks{_sep()}{s.track}.')
    else:
        t.say(f'     {t.OK} Model files: {", ".join(facts.model_files) or "none listed"}')
        if facts.ai_points >= 2:
            t.say(f'     {t.OK} AI line: {facts.ai_points} points')
        else:
            t.say(f'     {t.BAD} AI line: {facts.ai_note}. AssettoServer needs one for any AI car.')
    t.hint("The plugin also needs data about this track (its walls, road and course). If the server does\n"
           "not have it yet, setup makes it from the game install when you make the changes; that takes\n"
           "a few minutes.")


# --- 4. the bots ---------------------------------------------------------------------------------

def car_skins(st, model):
    cache = st.setdefault('skins', {})
    if model not in cache:
        c = car_facts(st['game'], model) if model else None
        used = [e.skin for e in st['server'].entries if e.model == model and e.skin]
        cache[model] = (list(dict.fromkeys(used + (c.skins if c else []))), c is not None and not c.exists)
    return cache[model]


def pick_skin(st, model, current=''):
    skins, _ = car_skins(st, model)
    if not skins:
        return current
    options = [(k, k) for k in ([current] if current and current not in skins else []) + skins]
    return t.choose(f'Skin for {model}:', options, default=current or skins[0])


def show_slots(rows):
    t.say('     Slot     Car                                       Skin')
    for n, (index, model, look) in enumerate(rows, 1):
        label = f'CAR_{index}' if index is not None else 'new'
        t.say(f'     {label:8s} {model:41s} {look}')


def bots_step(st):
    t.heading(4, STEPS, "Pick the bots' cars")
    t.hint("Any car the server already offers, slot by slot. A bot already in the entry list is put in the\n"
           "car you pick for its slot. Each bot takes a car slot on the server; players keep the others.")
    s = st['server']
    config = s.plugin_config or {}
    low = max(1, len(s.bots))
    t.hint('Bots in all, existing ones included (setup does not take bots away). The plugin spreads them\n'
           'round the lap.')
    count = t.ask_number('How many bots', max(low, st.get('bots') or 1), low, 12)
    old = st.get('slots') or []
    rows = bot_slots(s, st['game'], count, slots=[(m, k) for m, k in old] or None)
    show_slots(rows)
    if len(s.cars) > 1 or len(rows) > 1:
        every = t.choose('Every bot in:', [(None, '(as listed above)')] + [(c, c) for c in s.cars])
        if every:
            look = pick_skin(st, every)
            rows = [(i, every, look) for i, _, _ in rows]
            show_slots(rows)
    while len(rows) > 1 or len(s.cars) > 1 or car_skins(st, rows[0][1])[0][1:]:
        options = [(None, 'Go on')] + [(n, f'Change {f"CAR_{i}" if i is not None else "new"}: {m} ({k or "no skin"})')
                                        for n, (i, m, k) in enumerate(rows)]
        n = t.choose('Change a slot?', options)
        if n is None:
            break
        index, model, look = rows[n]
        model = t.choose('Car:', [(c, c) for c in s.cars + ([model] if model not in s.cars else [])], default=model)
        look = pick_skin(st, model, look if model == rows[n][1] else '')
        rows[n] = (index, model, look)
        show_slots(rows)
    models = list(dict.fromkeys(m for _, m, _ in rows))
    missing = [m for m in models if car_skins(st, m)[1]]
    if missing:
        t.say(f'     {t.BAD} The game install has no ' + ', '.join(f'content{_sep()}cars{_sep()}{m}' for m in missing) + '.')
    else:
        t.say(f'     {t.OK} ' + ('All found' if len(models) > 1 else 'Found') + ' in the game install.')
    name = t.ask_text("Bots' name in game (each shows as this and its slot number):",
                      st.get('bot_name') or bot_name_now(s))
    t.hint('A tandem train: the first car of each drives, the others follow it door to door. 1 = every bot\n'
           'drives on its own.')
    train = t.ask_number('Bots to a train', st.get('train_size') or int(config.get('TrainSize', 0) or 0), 0, 6,
                         zero='as it is')
    t.hint('How fast the bots drive, as a share of the flat-out speed round the lap (a policy trained with\n'
           'paces). Lower is calmer.')
    pace = t.ask_number('Pace', st.get('pace_percent') or int(float(config.get('BotPacePercent', 0) or 0)), 0, 105,
                        zero='as it is', suffix=' %')
    st.update(slots=[(m, k) for _, m, k in rows], bots=len(rows), bot_name=name,
              train_size=train or None, pace_percent=pace or None)


# --- 5. the changes ------------------------------------------------------------------------------

def plan_now(st):
    return plan_bot(st['server'], st['game'], slots=st.get('slots'), bots=st['bots'], bot_name=st['bot_name'],
                    plugin_source=st.get('plugin'), track_data=st.get('track_data'),
                    train_size=st.get('train_size'), pace_percent=st.get('pace_percent'))


def show_plan(plan, details=False, brief=False):
    """The change list as the last page shows it; brief: only its summary line (the list was just shown)."""
    if plan.ready:
        t.say(f'     {t.OK} This server already has everything the bot needs.')
    elif plan.blockers:
        t.say(f'     {t.WARN} {len(plan.changes)} change(s) setup can make, and {len(plan.blockers)} thing(s) it cannot fix yet.')
    else:
        t.say(f'     {t.OK} {len(plan.changes)} change(s), all of which setup can make.')
    if brief:
        return
    for c in plan.changes:
        t.say(f'       - {c.file}: {c.what}' if not c.file.startswith('(') else f'       - {c.what}')
        if details:
            for label, text in (('before', c.before), ('after', c.after)):
                for line in text.splitlines():
                    t.say(f'           {label:6s} | {line}')
    for b in plan.blockers:
        t.say(f'     {t.BAD} {b}')
    for n in plan.notes:
        t.say(f'     {t.WARN} {n}')


def has_plugin(p):
    return os.path.exists(os.path.join(p, 'GradientDriftPlugin.dll'))


def make_changes(st, plan):
    from .server import server_running
    folder = plan.server.folder
    if server_running(folder):
        if any((c.op or ())[:1] == ('copy',) and c.op[3:4] == ('plugin',) for c in plan.changes):
            t.say(f'     {t.BAD} AssettoServer is running from this folder: stop it, then make the changes. Nothing was changed.')
            return
        t.say(f'     {t.WARN} AssettoServer looks to be running from this folder: restart it afterwards.')
    slow = any((c.op or ())[:1] == ('track_data',) for c in plan.changes)
    t.hint('Stop the server first if it is running, and start it again afterwards. Setup keeps the\n'
           'originals, and Undo puts them back.' + (" Making the track's data takes a few minutes." if slow else ''))
    if not t.confirm(f'Make {len(plan.changes)} change(s) to {folder} now?', yes='Make the changes', no='Cancel'):
        t.say('     Nothing changed.')
        return
    before = last_run(folder)
    try:
        t.run_logged(lambda log: apply_plan(plan, log=log))
    except Exception as e:
        t.say(f'     {t.BAD} Stopped: {type(e).__name__}: {e}')
        t.say('     ' + ('Nothing in the server was changed.' if last_run(folder) == before else
                          'Some files were changed before it stopped: Undo the last changes puts them back.'))
        return
    t.say(f'     {t.OK} Done. Start the server again. Its log should say "GradientDrift: policy loaded from";')
    t.say('     the bots join when a player is on.')
    st['server'] = read_server(folder)
    st['applied'] = True


def export_changes(st):
    from .export import INSTRUCTIONS, UPLOAD, export_server
    t.hint('The changes go into a separate folder instead: the files to upload to your host, and\n'
           f'{INSTRUCTIONS}. The server folder is not changed.')
    out = t.ask_path('An empty or new folder for the export:',
                     lambda p: not (os.path.isdir(p) and os.listdir(p)) and not os.path.isfile(p),
                     lambda p: f'{p} is not empty: give an empty or new folder.')
    try:
        plan, files = t.run_logged(lambda log: export_server(
            st['server'].folder, out, st['game'], log=log, slots=st.get('slots'), bots=st['bots'],
            bot_name=st['bot_name'], plugin_source=st.get('plugin'), track_data=st.get('track_data'),
            train_size=st.get('train_size'), pace_percent=st.get('pace_percent')))
    except Exception as e:
        t.say(f'     {t.BAD} Nothing was exported: {type(e).__name__}: {e}')
        return
    if files is None:
        t.say(f'     {t.BAD} Nothing was exported: ' + ('; '.join(plan.blockers) or 'the server needs no changes.'))
        return
    t.say(f'     {t.OK} {len(files)} file(s) to upload are in {os.path.join(out, UPLOAD)}')
    t.say(f'     {INSTRUCTIONS} beside them says where each one goes and which config lines change.')
    st['exported'] = out


def changes_step(st):
    """The last page: the change list and what to do with it. Returns 'bots' or 'server' to go back
    to that step, or None when done."""
    t.heading(5, STEPS, "Here's what setup would change")
    t.hint('Setup keeps every file it changes in gradient-drift-setup-backups in the server folder, and\n'
           'Undo puts them back.')
    st.setdefault('plugin', default_plugin_source())
    st.setdefault('track_data', default_track_data())
    details, shown = False, None
    while True:
        plan = plan_now(st)
        st['plan'] = plan
        t.say()
        listed = [(c.file, c.what) for c in plan.changes] + plan.blockers + plan.notes
        show_plan(plan, details, brief=listed == shown and not details)
        shown, details = listed, False
        can = bool(plan.changes) and not plan.blockers
        options = ([('apply', 'Make these changes'), ('details', 'Show each change, before and after')] if can else []) \
            + ([('export', 'Export for a hosted server (the files to upload, the server folder unchanged)')] if can else []) \
            + ([('undo', 'Undo the last changes')] if last_run(plan.server.folder) else []) \
            + [('plugin', f"Plugin build: {st.get('plugin') or 'not found, pick it'}"),
               ('tracks', f"Track files: {st.get('track_data') or '(empty: setup makes them from the game)'}"),
               ('bots', "Change the bots"), ('server', 'Pick another server'), ('quit', 'Quit')]
        no_plugin = not st.get('plugin') and any('plugin is not installed' in b for b in plan.blockers)
        pick = t.choose('What next?', options, default='apply' if can else 'plugin' if no_plugin else 'quit')
        if pick == 'apply':
            make_changes(st, plan)
        elif pick == 'details':
            details = True
        elif pick == 'export':
            export_changes(st)
        elif pick == 'undo':
            if t.confirm('Put back the files the last run of setup changed?', yes='Put them back', no='Cancel'):
                done = []
                undo(plan.server.folder, log=done.append)
                for line in done or ['  Nothing to undo.']:
                    t.say(f'   {line}')
                st['server'] = read_server(plan.server.folder)
        elif pick == 'plugin':
            t.hint('The folder with GradientDriftPlugin.dll in it: the release download has it beside setup.')
            st['plugin'] = t.ask_path("The plugin's build folder:", has_plugin,
                                      lambda p: f'{p} has no GradientDriftPlugin.dll.', default=st.get('plugin') or '')
        elif pick == 'tracks':
            t.hint("Usually left empty: setup makes the track's data from the game. Pick a folder only to use track\n"
                   f"data you already have (a folder per track, e.g. drift_rl{_sep()}tracks). An empty answer"
                   + (" (or -)" if not t.rich() else '') + " leaves it empty.")
            path = t.ask_path('Track files folder:', lambda p: os.path.isdir(p) and not in_game_content(p),
                              lambda p: ("That is a track's folder in the game or server content. Leave Track files "
                                         "empty and setup makes the data from it." if in_game_content(p)
                                         else f'{p} is not a folder.'),
                              default=st.get('track_data') or '', allow_empty=True)
            st['track_data'] = (track_data_root(path) or path) if path else None
        elif pick in ('bots', 'server'):
            return pick
        else:
            return None


# --- the whole walk --------------------------------------------------------------------------------

def program_words():
    """How the operator started setup, for the command line it prints at the end."""
    if getattr(sys, 'frozen', False):
        return [sys.argv[0]]
    return ['py' if sys.platform == 'win32' else 'python3', '-m', 'gradientdrift_setup']


def command_line(st):
    """The run as one command: what to type next time to skip the questions."""
    s, plan = st['server'], st.get('plan')
    words = program_words() + ['apply' if plan is None or plan.changes else 'check', s.folder, '--game', st['game'],
                               '--bots', str(st['bots'])]
    defaults = [(m, k) for _, m, k in bot_slots(s, st['game'], st['bots'])]
    slots = st.get('slots') or defaults
    if slots != defaults:
        if len(set(slots)) == 1:
            words += ['--car', slots[0][0]] + (['--skin', slots[0][1]] if slots[0][1] else [])
        else:
            words += ['--car', ','.join(m for m, _ in slots), '--skin', ','.join(k for _, k in slots)]
    if st['bot_name'] != bot_name_now(s):
        words += ['--name', st['bot_name']]
    if st.get('train_size'):
        words += ['--train-size', str(st['train_size'])]
    if st.get('pace_percent'):
        words += ['--pace', str(st['pace_percent'])]
    if st.get('plugin') and st['plugin'] != default_plugin_source():
        words += ['--plugin', st['plugin']]
    if st.get('track_data') and st['track_data'] != default_track_data():
        words += ['--track-data', st['track_data']]
    if sys.platform == 'win32':
        return subprocess.list2cmdline(words)
    return ' '.join(shlex.quote(w) for w in words)


def run(state=None):
    st = state or {}
    t.say('Gradient Drift Setup')
    t.hint('Five steps, as in the window. Nothing is changed until the last one. Ctrl+C quits at any time.')
    if not t.rich():
        t.hint('(Type the number of your pick and press Enter; Enter alone takes the one in brackets.)')
    try:
        game_step(st)
        step = 'server'
        while step:
            if step == 'server':
                server_step(st)
                track_step(st)
                st.pop('slots', None)
                st.pop('bot_name', None)
            bots_step(st)
            step = changes_step(st)
    except t.Stop:
        t.say()
        t.say('Stopped.' + ('' if st.get('applied') else ' Nothing was changed.'))
        return 1
    t.say()
    if 'server' in st and 'bots' in st:
        t.say('Next time, the same in one line:')
        t.say(f'  {command_line(st)}', fold=False)
    return 0
