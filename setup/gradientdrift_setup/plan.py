"""The changes a server needs for drift bots, worked out without touching anything. Each change
carries the edit that makes it (apply.py)."""
import filecmp
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field

from .find import track_data_home, track_data_root
from .game import _DRIFT_RL, car_facts, track_facts
from .identify import TESTED_VERSION, identify, version_tuple
from .server import OLD_PLUGIN, OLD_PLUGIN_CONFIG_TAG, PLUGIN, PLUGIN_CONFIG_TAG, server_running

PLUGIN_FILES = ['GradientDriftPlugin.dll', 'GradientDriftPlugin.deps.json', 'GradientDriftPlugin.policy.json']
# Every policy a plugin release shipped (SHA-256 of the file, line endings as LF), with the release.
# A server's policy that is one of these is a stock one, replaced by the policy setup carries even
# beside the same plugin; any other was made for the server (polished for its track) and is kept.
# Add each release's policy here (build-plugin.yml refuses a plugin-v* tag whose policy is missing).
SHIPPED_POLICIES = {
    'd4388a8477bf5284d0d358e12908eb7293a7018b0140a46fd55f13bac99d2212': 'plugin-v0.1.0',
}
# Plugin settings a release renamed (old name: new name). A setting the plugin going in does not know
# at all is taken out of extra_cfg.yml (AssettoServer stops at start on a setting the plugin's
# configuration lacks); a renamed one keeps its value under the new name.
RENAMED_SETTINGS = {}
# the files the plugin reads from the server's folder for its track (drift_rl/setup_track.py makes them)
TRACK_FILES = ['{track}_triangles.csv.gz', '{track}_collision.csv.gz', '{track}_road_mesh.csv',
               '{track}_course_edges.csv', '{track}_pace_reference.csv']
# the road mesh is optional; the course edges are needed by a policy that reads the room to them,
# the pace reference by one that reads its pace error
OPTIONAL_TRACK_FILES = ('_road_mesh.csv', '_course_edges.csv', '_pace_reference.csv')

# AiParams the bots need (the six-bot test): AssettoServer spawns AI only for a player
# within MaxPlayerDistanceToAiSplineMeters of the AI line who has not been idle for
# PlayerAfkTimeoutSeconds - a drift player is often well off the line, or parked watching - and
# takes away an AI car further than PlayerRadiusMeters from every player, which spaced bots are
# unless it covers half a lap.
MIN_DISTANCE_TO_LINE_M = 100
MIN_AFK_TIMEOUT_S = 600
# AssettoServer spawns an AI car only at a point of the line at least
# SpawnSafetyDistanceToPlayerMeters (150) from every player and a random 20-70 m
# (Min/MaxAiSafetyDistanceMeters) from the AI cars either side: on a small track no point is
# that far from a player, so bots spawn only by luck. Setup sizes these to the track: the player
# distance to this share of the track's span, the AI gap to this share of the even spacing.
SPAWN_PLAYER_DISTANCE_SHARE = 0.25
SPAWN_AI_GAP_SHARE = 0.5


@dataclass
class Change:
    file: str          # relative to the server folder
    what: str          # one line, plain words
    before: str = ''
    after: str = ''
    op: tuple = ()     # how apply.py makes it


@dataclass
class Plan:
    server: object
    program: object = None
    car: str = ''      # the first bot's car (every bot's, when they share one)
    skin: str = ''
    slots: list = field(default_factory=list)      # [(entry index or None for a new bot, car, skin)] per bot
    changes: list = field(default_factory=list)
    blockers: list = field(default_factory=list)   # setup cannot fix these by editing files
    notes: list = field(default_factory=list)

    @property
    def ready(self):
        return not self.blockers and not self.changes


def choose_car(server):
    """The car the bots get when the operator does not say: the car of an existing bot entry, else
    the GravyGarage Miata the policy was trained as, else the server's first car."""
    if server.bots:
        return server.bots[0].model
    if 'gravygarage_street_miata' in server.cars:
        return 'gravygarage_street_miata'
    return server.cars[0] if server.cars else ''


def bot_slots(server, game=None, bots=1, car=None, skin=None, slots=None):
    """[(entry index, car, skin)] for each bot the server has after setup, existing AI entries first
    (in the entry list's order; index None for a bot setup adds). slots: [(car, skin)] slot by slot
    (an empty car or skin = the default below); bots past the end of it take its last car. car,
    skin: every bot's, the existing ones put in it too. Without either, an existing bot keeps its car
    and a new one takes choose_car's. A skin left out is the slot's own when its car is unchanged,
    else the one the server's entries use most for that car, else the car's first in the game."""
    existing = server.bots
    count = max(bots, len(existing), len(slots or []))
    skins = {}

    def default_skin(model):
        if model not in skins:
            used = Counter(e.skin for e in ([e for e in existing if e.model == model]
                                            or [e for e in server.entries if e.model == model]) if e.skin)
            c = car_facts(game, model) if game and not used else None
            skins[model] = used.most_common(1)[0][0] if used else (c.skins[0] if c and c.skins else '')
        return skins[model]

    out = []
    for i in range(count):
        entry = existing[i] if i < len(existing) else None
        want_car, want_skin = (slots[min(i, len(slots) - 1)] if slots else (car, skin))
        if slots and i >= len(slots):
            want_skin = ''          # a repeated car, not a repeated skin choice
        model = want_car or car or (entry.model if entry else choose_car(server))
        if want_skin:
            look = want_skin
        elif entry and entry.model == model:
            look = entry.skin
        else:
            look = default_skin(model) if model else ''
        out.append((entry.index if entry else None, model, look))
    return out


def policy_facts(path):
    """(the tracks it was trained on (empty for a policy that drives any track: one trained on
    generated courses too), how many inputs it reads, the paces it was trained for as fractions or
    None), or (None, 0, None) if unreadable."""
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        paces = doc.get('pace_fraction_range')
        tracks = [] if isinstance(doc.get('generated'), dict) else [doc.get('track', '')] + list(doc.get('tracks') or [])
        return [t for t in tracks if t], int(doc.get('obs_size', 46)), (tuple(paces) if paces else None)
    except (OSError, ValueError):
        return None, 0, None


def walls_only(triangles):
    """Whether a <track>_triangles.csv.gz is what the sensors see (sensor_view.py: solid walls and
    marker prisms only) rather than the track's drawn model."""
    import csv
    import gzip
    with gzip.open(triangles, 'rt', newline='') as f:
        return all(row.get('SurfaceType') == 'Wall' for row in csv.DictReader(f))


PACE_MODEL = 'pace_model 3'     # drift_rl/pace_model.py MODEL: what a current pace reference says it was made by


def current_pace_reference(path):
    """Whether a <track>_pace_reference.csv was made by pace_model (its Source column), rather than
    measured by a policy driving the track (gen3, before 2026-10-18), which can ask the impossible
    where that policy cut a corner."""
    import csv
    try:
        with open(path, newline='', encoding='utf-8') as f:
            row = next(csv.DictReader(f), None)
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return bool(row) and row.get('Source') == PACE_MODEL
    from . import game
    if ref and os.path.exists(ref if os.path.isabs(ref) else os.path.join(game._DRIFT_RL, ref)):
        return ref
    return None


def policy_sha256(path):
    """The SHA-256 of a policy file with its line endings as LF (a copy on Windows may have CRLF)."""
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()


def plugin_settings(dll):
    """The setting names the plugin build knows: each property of its configuration has a set_<Name>
    method, whose name is in the assembly's metadata strings. None when the file cannot be read."""
    import re
    try:
        with open(dll, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    return {m.decode() for m in re.findall(rb'set_([A-Za-z][A-Za-z0-9_]*)\x00', data)}


def same_data(a, b):
    """Whether two track data files hold the same data: their text the same apart from line endings
    (a copy made on Windows writes CRLF), inside the .gz for a gzipped one."""
    import gzip
    if filecmp.cmp(a, b, shallow=False):
        return True

    def text(path):
        with (gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')) as f:
            return f.read().replace(b'\r\n', b'\n')
    try:
        return text(a) == text(b)
    except (OSError, EOFError):
        return False


def plan_bot(server, game, car=None, skin=None, bots=1, bot_name='DriftBot', ai_mode='fixed',
             plugin_source=None, track_data=None, train_size=None, pace_percent=None, slots=None):
    """car, skin, slots: the bots' cars (bot_slots); an existing AI entry in another car or skin is
    put in the one asked for. plugin_source: a folder with the plugin's build (GradientDriftPlugin.dll etc.); track_data: a folder
    with a folder of files per track (drift_rl/tracks/), copied in when the server lacks them or has
    a different copy. train_size: bots to a tandem train (1 = every bot drives on its own; None =
    leave the server's setting). pace_percent: the bots' pace, % of each point's flat-out speed, for
    a policy trained with paces (None = leave it)."""
    p = Plan(server=server)
    if track_data:
        # one track's own folder (drift_rl/tracks/<track>) picked for the folder of them all
        track_data = track_data_root(track_data) or track_data
    if server.problems:
        p.blockers += server.problems
        return p

    # --- the server program ---
    p.program = prog = identify(server.folder)
    if prog.kind == 'kunos':
        p.blockers.append(prog.note)
        return p
    if prog.kind == 'unknown':
        p.notes.append(prog.note + '; setup assumes AssettoServer')
    elif not prog.version:
        p.notes.append("AssettoServer's version could not be read; the plugin is built for " + TESTED_VERSION)
    elif version_tuple(prog.version) != version_tuple(TESTED_VERSION):
        p.notes.append(f"AssettoServer {prog.version}: the plugin is built and tested for {TESTED_VERSION}, "
                       "and may not load in another version")

    # --- the track ---
    t = track_facts(game, server.track, server.layout) if game else None
    where = server.track + (f' ({server.layout})' if server.layout else '')
    lap_m = 0.0
    span_m = 0.0
    ai_copy = None          # the game's AI line copied as it is, when the server lacks one
    rel_ai = ''
    if t is None:
        p.notes.append("no game install given: the track and car checks were skipped")
    elif not t.exists:
        p.blockers.append(f"the game install has no content/tracks/{server.track}")
    else:
        lap_m = t.lap_m
        span_m = t.span_m
        server_ai = server.server_ai_line()
        rel_ai = os.path.relpath(server_ai, server.folder).replace(os.sep, '/')
        if t.ai_points < 2 and not t.recorded:
            p.blockers.append(f"{where} has no usable AI line ({t.ai_note}) and no recorded lap to make one from; "
                              "AssettoServer needs one to place any AI car")
        elif t.ai_points >= 2 and not os.path.exists(server_ai):
            ai_copy = Change(rel_ai, f"copy the track's AI line ({t.ai_points} points) from the game install",
                             op=('copy', t.ai_line, rel_ai))

    # an installed plugin that differs from the build setup carries is replaced by it, its policy
    # with it: an older plugin rejects the settings a newer policy brings (the bots fall back to the
    # stock AI), and a newer plugin may read inputs an older policy lacks. A policy beside the same
    # plugin is left alone (it may have been polished for the track); the originals go to the backup.
    plugin_rel = f'plugins/{PLUGIN}'
    policy_name = f'{PLUGIN}.policy.json'

    def differs(f):
        there, source = os.path.join(server.folder, plugin_rel, f), os.path.join(plugin_source or '', f)
        return (f in server.plugin_files and plugin_source is not None and os.path.exists(source)
                and not filecmp.cmp(source, there, shallow=False))
    stale = [f for f in PLUGIN_FILES if differs(f)]
    plugin_update = any(f != policy_name for f in stale)
    installed = os.path.join(server.folder, plugin_rel, policy_name)
    carried = os.path.join(plugin_source, policy_name) if plugin_source else ''
    # beside the same plugin, a server's policy that a release shipped is replaced too (an older
    # stock policy); one that no release shipped is the server's own and stays
    stock_policy = (not plugin_update and policy_name in stale
                    and SHIPPED_POLICIES.get(policy_sha256(installed)))
    # the policy the bots will run, which the track's checks are for
    if os.path.exists(installed) and not plugin_update and not stock_policy:
        policy = installed
    elif not os.path.exists(installed) and server.old_policy:
        policy = server.old_policy
    else:
        policy = carried if carried and os.path.exists(carried) else installed
    trained, inputs, paces = policy_facts(policy) if os.path.exists(policy) else (None, 0, None)
    missing = []
    for pattern in TRACK_FILES:
        name = pattern.format(track=server.track)
        source = os.path.join(track_data, server.track, name) if track_data else ''
        there = os.path.join(server.folder, name)
        if name.endswith('_pace_reference.csv'):
            # only pace_model's: an older one (measured by a policy) is made again below
            source = source if source and current_pace_reference(source) else ''
            if os.path.exists(there) and not current_pace_reference(there):
                if source:
                    p.changes.append(Change(name, "replace the server's pace reference (measured by a policy driving "
                                            "the track) with the one from the track's shape", op=('copy', source, name)))
                else:
                    missing.append(name)
                continue
        if os.path.exists(there):
            if source and os.path.exists(source) and not same_data(source, there):
                p.changes.append(Change(name, "update the track's data for the plugin (the server's copy differs "
                                        "from the one in the track files)", op=('copy', source, name)))
            continue
        if source and os.path.exists(source):
            p.changes.append(Change(name, "copy the track's data for the plugin", op=('copy', source, name)))
        else:
            missing.append(name)

    sensors = os.path.join(track_data, server.track, f'{server.track}_triangles.csv.gz') if track_data else ''
    if sensors and os.path.exists(sensors) and not walls_only(sensors):
        p.notes.append(f"{server.track}'s track data is from before the sensors saw only solid walls: "
                       f"py {os.path.abspath(os.path.join(_DRIFT_RL, 'sensor_view.py'))} "
                       f"--tracks \"{track_data}\" --track {server.track} remakes what they see")

    # the track's data, made from the game install when it is not there yet (drift_rl/setup_track.py)
    home = track_data or track_data_home()
    to_make = [m for m in missing if not m.endswith('_pace_reference.csv')]    # training makes that one
    needs_line = (t is not None and t.exists and t.ai_points < 2 and bool(t.recorded)
                  and not os.path.exists(os.path.join(server.folder, rel_ai)))
    if t is not None and t.exists and (t.ai_points >= 2 or t.recorded) and (to_make or needs_line):
        p.changes.append(Change('(track data)', f"make {server.track}'s data from the game install into "
                                f"{os.path.join(home, server.track)}: its geometry, road mesh, AI line (checked, and "
                                "rebuilt down the middle of the road if it needs it), course and zones; a few minutes",
                                op=('track_data', t.folder, server.layout or '', home, server.track, rel_ai)))
        for name in to_make:
            p.changes.append(Change(name, "copy the track's data for the plugin (made above)",
                                    op=('copy', os.path.join(home, server.track, name), name)))
        missing = [m for m in missing if m not in to_make]
        ai_copy = None      # the build puts its line in the server: the game's, or rebuilt if it needed it
        if needs_line:
            p.notes.append(f"{where} has no AI line of its own: setup makes one from the recorded lap "
                           f"({os.path.basename(t.recorded)}) and puts it in the server")
    # the track's data is made per point of one AI line (setup_track.py saves it beside the data as
    # <track>_fast_lane.ai, rebuilt down the middle of the road if the game's needed it): the server
    # must run that line, or the course edges and pace reference do not fit it
    data_line = os.path.join(track_data, server.track, f'{server.track}_fast_lane.ai') if track_data and rel_ai else ''
    if data_line and os.path.exists(data_line) and not any(c.op[:1] == ('track_data',) for c in p.changes):
        there = os.path.join(server.folder, rel_ai)
        if not os.path.exists(there):
            ai_copy = Change(rel_ai, "copy the AI line the track's data was made for", op=('copy', data_line, rel_ai))
        elif not filecmp.cmp(data_line, there, shallow=False):
            ai_copy = Change(rel_ai, "replace the server's AI line with the one the track's data was made for "
                             "(the plugin's course edges and pace reference follow its points)",
                             op=('copy', data_line, rel_ai))
        else:
            ai_copy = None
    if ai_copy:
        p.changes.append(ai_copy)
    needed_missing = [m for m in missing if not m.endswith(OPTIONAL_TRACK_FILES)
                      or (m.endswith('_course_edges.csv') and inputs > 46)
                      or (m.endswith('_pace_reference.csv') and inputs > 54)]
    pace_missing = [m for m in needed_missing if m.endswith('_pace_reference.csv')]
    needed_missing = [m for m in needed_missing if m not in pace_missing]
    if needed_missing:
        p.blockers.append(f"the plugin needs {' and '.join(needed_missing)} in the server folder. Setup makes them "
                          f"from the game's copy of {server.track} once what stops it above is fixed (the track "
                          "missing from the game install, or no AI line)")
    if pace_missing:
        # made here from the track's shape (drift_rl/pace_model.py), as training makes it, when the
        # track's data is (or is about to be) there
        made_here = any(c.op[:1] == ('track_data',) for c in p.changes)
        if made_here or os.path.exists(os.path.join(home, server.track, f'{server.track}_fast_lane.ai')):
            p.changes.append(Change(pace_missing[0], "work out the track's reference speed round the lap (what a pace "
                                    "of 100% asks for) from its shape, into "
                                    f"{os.path.join(home, server.track)}, and copy it for the plugin",
                                    op=('pace_reference', policy, home, server.track)))
            pace_missing = []
    def final_pace_reference():
        """The pace reference the server runs after this (the check's speed targets), or None for the
        track files' own (copied, or made there by this run)."""
        name = f'{server.track}_pace_reference.csv'
        if any(c.file == name for c in p.changes):
            return None
        there = os.path.join(server.folder, name)
        return there if os.path.exists(there) else None

    # what the track's data asks of the bots, checked before anything is written (drift_rl/check_track.py)
    track_names = {pattern.format(track=server.track) for pattern in TRACK_FILES}
    if any(c.op[:1] in (('track_data',), ('pace_reference',)) or (c.op[:1] == ('copy',) and c.file in track_names)
           for c in p.changes):
        p.changes.append(Change('(track check)', f"check {server.track}'s speed targets and walls before anything is "
                                "written, and stop if they look wrong",
                                op=('track_check', policy, home, server.track, final_pace_reference())))
    if pace_missing:
        p.blockers.append(f"this policy needs {pace_missing[0]} in the server folder. Setup works it out from the "
                          "track's data, which it makes from the game install once what stops it above is fixed")

    # --- the cars, slot by slot ---
    p.slots = bot_slots(server, game, bots, car, skin, slots)
    if not p.slots or not all(model for _, model, _ in p.slots):
        p.blockers.append("no car to give the bot: CARS= in server_cfg.ini is empty")
        return p
    p.car, p.skin = p.slots[0][1], p.slots[0][2]
    models = list(dict.fromkeys(model for _, model, _ in p.slots))
    for model in models:
        c = car_facts(game, model) if game else None
        if c is not None and not c.exists:
            p.blockers.append(f"the game install has no content/cars/{model}")

    # --- entry_list.ini ---
    entries = {e.index: e for e in server.entries}
    for index, model, look in p.slots:
        e = entries.get(index)
        if e is None or (e.model, e.skin) == (model, look):
            continue
        keys = (('MODEL', model),) + ((('SKIN', look),) if look != e.skin else ())
        p.changes.append(Change('cfg/entry_list.ini', f"put the bot in CAR_{index} in {model}, skin {look or '(default)'} "
                                f"(was {e.model}" + (f", skin {e.skin})" if e.skin else ")"),
                                f'[CAR_{index}]\nMODEL={e.model}\nSKIN={e.skin}', f'[CAR_{index}]\nMODEL={model}\nSKIN={look}',
                                op=('ini_keys', 'cfg/entry_list.ini', f'CAR_{index}', keys)))
    new = [(model, look) for index, model, look in p.slots if index is None]
    to_add = len(new)
    bots = len(p.slots)                     # the bots the server has after this, existing ones included
    next_index = max((e.index for e in server.entries), default=-1) + 1
    for k, (model, look) in enumerate(new):
        n = next_index + k
        body = '\n'.join([f'[CAR_{n}]', f'MODEL={model}', f'SKIN={look}', 'SPECTATOR_MODE=0',
                          f'DRIVERNAME={bot_name}' + (f' {k + 1}' if to_add > 1 else ''), 'TEAM=', 'GUID=',
                          'BALLAST=0', 'RESTRICTOR=0', f'AI={ai_mode}'])
        p.changes.append(Change('cfg/entry_list.ini', f"add a bot as CAR_{n}: {model}, skin {look or '(default)'}, "
                                f"AI={ai_mode}", after=body, op=('append', 'cfg/entry_list.ini', body)))
    entries_after = len(server.entries) + to_add
    players = len([e for e in server.entries if e.ai not in ('auto', 'fixed')])

    # --- server_cfg.ini ---
    missing_cars = [m for m in models if m not in server.cars]
    if missing_cars:
        cars = ';'.join(server.cars + missing_cars)
        p.changes.append(Change('cfg/server_cfg.ini', f"add {', '.join(missing_cars)} to the server's cars",
                                'CARS=' + ';'.join(server.cars), 'CARS=' + cars,
                                op=('ini', 'cfg/server_cfg.ini', 'SERVER', 'CARS', cars)))
        p.notes.append(f"players need {', '.join(missing_cars)} installed to join once "
                       f"{'it is' if len(missing_cars) == 1 else 'they are'} on the server's car list")
    if server.max_clients < entries_after:
        p.changes.append(Change('cfg/server_cfg.ini', f"allow {entries_after} cars instead of {server.max_clients}",
                                f'MAX_CLIENTS={server.max_clients}', f'MAX_CLIENTS={entries_after}',
                                op=('ini', 'cfg/server_cfg.ini', 'SERVER', 'MAX_CLIENTS', str(entries_after))))

    # --- extra_cfg.yml ---
    x = 'cfg/extra_cfg.yml'
    if not server.enable_ai:
        p.changes.append(Change(x, "turn on AI", 'EnableAi: false', 'EnableAi: true', op=('yaml_top', 'EnableAi', 'true')))
    if server.old_plugin:
        # set up under the plugin's old name: renamed in place, its settings kept
        if OLD_PLUGIN in server.plugins:
            p.changes.append(Change(x, f"the plugin's new name: {OLD_PLUGIN} -> {PLUGIN}",
                                    f'EnablePlugins: [{", ".join(server.plugins)}]',
                                    f'EnablePlugins: [{", ".join(PLUGIN if q == OLD_PLUGIN else q for q in server.plugins)}]',
                                    op=('yaml_rename_plugin', OLD_PLUGIN, PLUGIN)))
        if server.old_plugin_doc:
            p.changes.append(Change(x, "the plugin's settings under its new name", f'--- {OLD_PLUGIN_CONFIG_TAG}',
                                    f'--- {PLUGIN_CONFIG_TAG}', op=('yaml_rename_doc', OLD_PLUGIN_CONFIG_TAG, PLUGIN_CONFIG_TAG)))
        p.notes.append(f"plugins/{OLD_PLUGIN}/ is left where it is (nothing loads it once {PLUGIN} is enabled); "
                       "delete it yourself when the new one works")
    if PLUGIN not in server.plugins and OLD_PLUGIN not in server.plugins:
        p.changes.append(Change(x, "enable the GradientDrift plugin", f'EnablePlugins: [{", ".join(server.plugins)}]',
                                f'EnablePlugins: [{", ".join(server.plugins + [PLUGIN])}]', op=('yaml_plugin', PLUGIN)))

    def ai_param(key, now, want, why):
        p.changes.append(Change(x, why, f'AiParams.{key}: {now}', f'AiParams.{key}: {want}',
                                op=('yaml_child', 'AiParams', key, str(want))))

    a = server.ai_params
    if server.ai_max_player_count != players:
        ai_param('MaxPlayerCount', server.ai_max_player_count, players, f"keep {players} slots for players, the rest for AI")
    if a.get('MaxPlayerDistanceToAiSplineMeters', 7) < MIN_DISTANCE_TO_LINE_M:
        ai_param('MaxPlayerDistanceToAiSplineMeters', a.get('MaxPlayerDistanceToAiSplineMeters', 7), MIN_DISTANCE_TO_LINE_M,
                 "spawn bots for a player anywhere on the paving, not only within a few metres of the AI line")
    if a.get('PlayerAfkTimeoutSeconds', 10) < MIN_AFK_TIMEOUT_S:
        ai_param('PlayerAfkTimeoutSeconds', a.get('PlayerAfkTimeoutSeconds', 10), MIN_AFK_TIMEOUT_S,
                 "keep the bots for a player who parks to watch them")
    if lap_m and bots > 1:
        want = int(math.ceil(lap_m * 0.625 / 50.0) * 50)
        if a.get('PlayerRadiusMeters', 200) < want:
            ai_param('PlayerRadiusMeters', a.get('PlayerRadiusMeters', 200), want,
                     f"keep bots spaced round the {lap_m:.0f} m lap instead of taking away the ones far from a player")
    if span_m:
        want = max(20, int(span_m * SPAWN_PLAYER_DISTANCE_SHARE))
        if a.get('SpawnSafetyDistanceToPlayerMeters', 150) > want:
            ai_param('SpawnSafetyDistanceToPlayerMeters', a.get('SpawnSafetyDistanceToPlayerMeters', 150), want,
                     f"let bots spawn on a track {span_m:.0f} m across (no point of it is 150 m from a player)")
    if lap_m and bots > 1:
        gap = int(lap_m / (bots + 1) * SPAWN_AI_GAP_SHARE)
        if a.get('MaxAiSafetyDistanceMeters', 70) > gap:
            ai_param('MaxAiSafetyDistanceMeters', a.get('MaxAiSafetyDistanceMeters', 70), gap,
                     f"room to spawn {bots} bots round the {lap_m:.0f} m lap")
            if a.get('MinAiSafetyDistanceMeters', 20) > gap:
                ai_param('MinAiSafetyDistanceMeters', a.get('MinAiSafetyDistanceMeters', 20), gap,
                         f"room to spawn {bots} bots round the {lap_m:.0f} m lap")
    for key in ('MinCollisionStopTimeSeconds', 'MaxCollisionStopTimeSeconds'):
        if a.get(key, 1 if key.startswith('Min') else 3) != 0:
            ai_param(key, a.get(key, 1 if key.startswith('Min') else 3), 0,
                     "don't freeze a bot for seconds when a player touches it (traffic stops; a drift bot drives on)")

    config = server.plugin_config or {}
    if not config.get('EnableLearnedPolicy'):
        p.changes.append(Change(x, "turn on the GradientDrift plugin's driving", after=f'--- {PLUGIN_CONFIG_TAG}\nEnableLearnedPolicy: true',
                                op=('yaml_doc', PLUGIN_CONFIG_TAG, 'EnableLearnedPolicy', 'true')))
    if bots > 1 and config.get('HoldBotSpacing') is False:
        p.changes.append(Change(x, "spread the bots round the lap", 'HoldBotSpacing: false', 'HoldBotSpacing: true',
                                op=('yaml_doc', PLUGIN_CONFIG_TAG, 'HoldBotSpacing', 'true')))
    now_train = int(config.get('TrainSize', 1) or 1)
    if train_size is not None and train_size != now_train:
        what = ("let every bot drive on its own" if train_size <= 1
                else f"run the bots as tandem trains of {train_size}: the first car of each train drives, "
                     "the others follow it door to door")
        p.changes.append(Change(x, what, f'TrainSize: {now_train}', f'TrainSize: {train_size}',
                                op=('yaml_doc', PLUGIN_CONFIG_TAG, 'TrainSize', str(train_size))))
    size = train_size if train_size is not None else now_train
    if size > 1 and bots % size:
        p.notes.append(f"{bots} bots in trains of {size}: the last train has {bots % size}")
    if pace_percent is not None:
        if not paces:
            p.notes.append("the policy has no pace (it was trained without one): the bots' pace is not set")
        else:
            low, high = 100.0 * paces[0], 100.0 * paces[1]
            if not low <= pace_percent <= high:
                p.notes.append(f"pace {pace_percent:g}% is outside the {low:g} - {high:g}% the policy was trained "
                               "for; the plugin will hold it inside")
            now_pace = config.get('BotPacePercent', 0)
            if float(now_pace or 0) != float(pace_percent):
                p.changes.append(Change(x, f"have the bots drive at {pace_percent:g}% of the flat-out speed round the lap",
                                        f'BotPacePercent: {now_pace}', f'BotPacePercent: {pace_percent:g}',
                                        op=('yaml_doc', PLUGIN_CONFIG_TAG, 'BotPacePercent', f'{pace_percent:g}')))

    # --- the plugin ---
    missing = [f for f in PLUGIN_FILES if f not in server.plugin_files]
    if server.old_policy and policy_name in missing:
        # the policy the server ran under the old name goes with it, not the build's own
        p.changes.append(Change(f'{plugin_rel}/{policy_name}', f"the policy plugins/{OLD_PLUGIN}/ ran",
                                op=('copy', server.old_policy, f'{plugin_rel}/{policy_name}')))
        missing.remove(policy_name)
    if missing:
        sources = [os.path.join(plugin_source, f) for f in missing] if plugin_source else []
        if sources and all(os.path.exists(s) for s in sources):
            for f, s in zip(missing, sources):
                p.changes.append(Change(f'{plugin_rel}/{f}', 'install the plugin', op=('copy', s, f'{plugin_rel}/{f}')))
        else:
            p.blockers.append(f"the plugin is not installed ({', '.join(missing)} missing from {plugin_rel}/)"
                              + ("" if plugin_source else ": give setup the folder with the plugin's build"))
    if plugin_update:
        for f in stale:
            what = ("update the plugin to the build setup carries" if f != policy_name else
                    "the policy that comes with that plugin (the server's goes to the backup; undo puts it back)")
            p.changes.append(Change(f'{plugin_rel}/{f}', what, op=('copy', os.path.join(plugin_source, f),
                                                                   f'{plugin_rel}/{f}', 'plugin')))
        if server_running(server.folder):
            p.blockers.append("AssettoServer is running from this folder (the plugin is in use): stop it, then run "
                              "setup again to update the plugin")
    elif stock_policy:
        p.changes.append(Change(f'{plugin_rel}/{policy_name}',
                                f"update the policy (the server's is the one {stock_policy} shipped) to the one setup "
                                "carries; it drives from the server's next start",
                                op=('copy', carried, f'{plugin_rel}/{policy_name}')))
    elif stale:
        p.notes.append(f"the server's policy ({plugin_rel}/{policy_name}) differs from the one setup carries for the "
                       "same plugin and no release shipped it, so it is kept (it may have been polished for this "
                       f"track). To use setup's, copy {os.path.join(plugin_source, policy_name)} over it")
    # settings the plugin going in (or the one installed) does not know: renamed ones move to their new
    # name, the rest come out, so AssettoServer does not stop at start on them
    going_in = plugin_update or PLUGIN_FILES[0] in missing
    dll = (os.path.join(plugin_source, PLUGIN_FILES[0]) if plugin_source else '') if going_in \
        else os.path.join(server.folder, plugin_rel, PLUGIN_FILES[0])
    known = plugin_settings(dll) if dll and os.path.exists(dll) else None
    if known and server.plugin_config:
        for key, value in server.plugin_config.items():
            if key in known:
                continue
            new_key = RENAMED_SETTINGS.get(key)
            if new_key and new_key in known:
                if new_key in server.plugin_config:
                    p.changes.append(Change(x, f"take out {key} (renamed {new_key}, which is set already)",
                                            f'{key}: {value}', '', op=('yaml_doc_remove', PLUGIN_CONFIG_TAG, key)))
                else:
                    p.changes.append(Change(x, f"the plugin setting {key} is called {new_key} now", f'{key}: {value}',
                                            f'{new_key}: {value}', op=('yaml_doc_rename', PLUGIN_CONFIG_TAG, key, new_key)))
            else:
                p.changes.append(Change(x, f"take out the plugin setting {key}: this plugin has no such setting",
                                        f'{key}: {value}', '', op=('yaml_doc_remove', PLUGIN_CONFIG_TAG, key)))
    # an any-car policy (it reads the car inputs: more than 55) drives each bot with its own car's numbers
    if inputs > 55 and game:
        from .carfile import car_file_text, pace_source, same_car_file
        pace_made = any(c.op[:1] == ('pace_reference',) for c in p.changes)
        source, why_not = pace_source(policy, home, server.folder, server.track, made_here=pace_made) \
            if paces else (None, None)
        if why_not:
            p.notes.append(f"the bots' cars keep PACE_SCALE 1 (each asked for the reference car's speed): {why_not}")
        for model in sorted(models):
            rel = f'{plugin_rel}/cars/{model}.json'
            text, why = car_file_text(game, model)
            if text is None:
                p.blockers.append(f"no car file for {model}: {why}")
                continue
            there = os.path.join(server.folder, rel)
            old = ''
            if os.path.exists(there):
                with open(there, encoding='utf-8') as f:
                    old = f.read()
            if source:
                if not same_car_file(old, text, source[-1]):
                    p.changes.append(Change(rel, f"the numbers the bots in {model} drive with, from the game's car files, "
                                            f"and its pace scale {source[-1]}" + (" (updated)" if old else ""),
                                            after=text, op=('car_file', rel, text, source)))
            elif old != text:
                p.changes.append(Change(rel, f"the numbers the bots in {model} drive with, from the game's car files"
                                        + (" (updated)" if old else ""), after=text, op=('write', rel, text)))
    if trained and not any(t.lower() in server.track.lower() for t in trained):
        p.blockers.append(f"the plugin's policy was trained on {', '.join(trained)} only; on {server.track} the plugin "
                          "leaves the cars to the stock AI. A policy for this track: drift_rl/train_ppo_gpu.py --track "
                          f"{server.track}")
    return p
