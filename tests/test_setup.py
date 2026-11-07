"""Setup on a made-up server: a fake game install (a generated course and a made-up rear-drive car),
a fake AssettoServer folder and a fake plugin build. Setup has to plan and make the changes a
server needs for drift bots, leave nothing to do on a second run, take out a plugin setting the
build does not know, and put every file back on undo."""
import json
import os
import re
import shutil

import pytest
from ruamel.yaml import YAML

from conftest import ROOT, SHIPPED_POLICY

CAR = 'test_drift_car'
SKIN = 'red'

# A rear-drive car of made-up numbers in the shape of the game's data files (cars/car_numbers.py
# reads these).
CAR_FILES = {
    'car.ini': """[INFO]
SCREEN_NAME=Test drift car
[BASIC]
TOTALMASS=1150
INERTIA=1.70,1.20,4.20
[CONTROLS]
STEER_LOCK=450
STEER_RATIO=15
""",
    'suspensions.ini': """[BASIC]
WHEELBASE=2.45
CG_LOCATION=0.52
[FRONT]
TRACK=1.46
BASEY=-0.10
SPRING_RATE=70000
[REAR]
TRACK=1.45
BASEY=-0.10
SPRING_RATE=60000
[ARB]
FRONT=25000
REAR=12000
""",
    'tyres.ini': """[HEADER]
VERSION=10
[COMPOUND_DEFAULT]
INDEX=0
[FRONT]
NAME=Street
WIDTH=0.205
RADIUS=0.30
DY_REF=1.22
DY1=-0.05
LS_EXPY=0.80
FZ0=3000
FRICTION_LIMIT_ANGLE=8.5
FLEX=0.0009
[REAR]
NAME=Street
WIDTH=0.225
RADIUS=0.31
DY_REF=1.20
DY1=-0.05
LS_EXPY=0.80
FZ0=3000
FRICTION_LIMIT_ANGLE=8.5
FLEX=0.0009
""",
    'drivetrain.ini': """[TRACTION]
TYPE=RWD
[GEARS]
COUNT=5
GEAR_1=3.40
GEAR_2=2.10
GEAR_3=1.45
GEAR_4=1.10
GEAR_5=0.90
FINAL=4.10
""",
    'engine.ini': """[HEADER]
POWER_CURVE=power.lut
[ENGINE_DATA]
LIMITER=7500
""",
    'power.lut': "1000|160\n3000|220\n5000|250\n6500|230\n7500|200\n",
}


def plugin_build(folder):
    """A plugin build setup can read: the DLL only has to carry its settings' set_<Name> strings
    (all setup looks for in it), the policy is the shipped one."""
    os.makedirs(folder)
    with open(os.path.join(ROOT, 'GradientDriftPlugin', 'GradientDriftConfiguration.cs'), encoding='utf-8-sig') as f:
        names = re.findall(r'public [\w<>?,\[\] ]+? (\w+) \{ get; set; \}', f.read())
    assert names
    with open(os.path.join(folder, 'GradientDriftPlugin.dll'), 'wb') as f:
        f.write(b'MZ fake plugin build\0' + b''.join(b'set_' + n.encode() + b'\0' for n in names))
    with open(os.path.join(folder, 'GradientDriftPlugin.deps.json'), 'w') as f:
        f.write('{}\n')
    shutil.copyfile(SHIPPED_POLICY, os.path.join(folder, 'GradientDriftPlugin.policy.json'))


@pytest.fixture
def world(tmp_path, tracks_dir, course_names):
    """(server folder, game folder, plugin build folder, track data folder, track)."""
    import drift_env as de
    import fast_sim as fs
    track = course_names[0]
    # the track's data, as setup_track.py leaves it (a pace reference included)
    data = tmp_path / 'track_data'
    shutil.copytree(os.path.join(tracks_dir, track), data / track)
    de.course_pace_reference(str(data), track, fs.load_world(str(data), track=track))
    # and its surface grip (one row per collision row; a generated course grips 1 everywhere), which
    # a policy trained with SurfaceGrip needs
    import gzip
    with gzip.open(data / track / f'{track}_collision.csv.gz', 'rt') as f:
        rows = sum(1 for _ in f) - 1
    with gzip.open(data / track / f'{track}_collision_grip.csv.gz', 'wt') as f:
        f.write('Key,Grip\n' + '-,1.0000\n' * rows)

    game = tmp_path / 'game'
    ai = game / 'content' / 'tracks' / track / 'ai'
    ai.mkdir(parents=True)
    shutil.copyfile(data / track / f'{track}_fast_lane.ai', ai / 'fast_lane.ai')
    car = game / 'content' / 'cars' / CAR
    (car / 'data').mkdir(parents=True)
    (car / 'skins' / SKIN).mkdir(parents=True)
    for name, text in CAR_FILES.items():
        (car / 'data' / name).write_text(text)

    server = tmp_path / 'server'
    (server / 'cfg').mkdir(parents=True)
    (server / 'AssettoServer').write_bytes(b'\0fake "AssettoServer/0.0.54-gabcdef AssettoServer\0')
    (server / 'cfg' / 'server_cfg.ini').write_text(
        f'[SERVER]\nNAME=Test server\nCARS={CAR}\nTRACK={track}\nCONFIG_TRACK=\nMAX_CLIENTS=1\n')
    (server / 'cfg' / 'entry_list.ini').write_text(
        f'[CAR_0]\nMODEL={CAR}\nSKIN={SKIN}\nSPECTATOR_MODE=0\nDRIVERNAME=\nTEAM=\nGUID=\nBALLAST=0\nRESTRICTOR=0\n')
    (server / 'cfg' / 'extra_cfg.yml').write_text(
        'EnableAi: false\nEnablePlugins: []\nAiParams:\n  MaxPlayerCount: 0\n'
        '--- !GradientDriftConfiguration\nNoSuchSetting: 3\n')

    plugin = tmp_path / 'plugin'
    plugin_build(str(plugin))
    return str(server), str(game), str(plugin), str(data), track


def plan(world, bots=2):
    from gradientdrift_setup.plan import plan_bot
    from gradientdrift_setup.server import read_server
    server, game, plugin, data, _ = world
    return plan_bot(read_server(server), game, bots=bots, bot_name='TestBot', plugin_source=plugin, track_data=data,
                    pace_percent=90)


def originals(server):
    out = {}
    for root, _, files in os.walk(server):
        for name in files:
            path = os.path.join(root, name)
            with open(path, 'rb') as f:
                out[os.path.relpath(path, server)] = f.read()
    return out


def test_setup_makes_a_server_ready_and_undo_puts_it_back(world):
    from gradientdrift_setup.apply import apply_plan, undo
    server, game, plugin, data, track = world
    before = originals(server)

    p = plan(world)
    assert not p.blockers, p.blockers
    assert not p.server.problems, p.server.problems
    assert p.program.kind == 'assettoserver' and p.program.version == '0.0.54'
    whats = [c.what for c in p.changes]
    assert any('NoSuchSetting' in w for w in whats), 'the setting the build does not know is not taken out'

    apply_plan(p, log=lambda *_: None)

    plugin_dir = os.path.join(server, 'plugins', 'GradientDriftPlugin')
    for name in ('GradientDriftPlugin.dll', 'GradientDriftPlugin.deps.json', 'GradientDriftPlugin.policy.json'):
        with open(os.path.join(plugin_dir, name), 'rb') as a, open(os.path.join(plugin, name), 'rb') as b:
            assert a.read() == b.read(), f'{name} is not the build setup carries'
    for kind in ('triangles.csv.gz', 'collision.csv.gz', 'road_mesh.csv', 'course_edges.csv', 'pace_reference.csv'):
        assert os.path.exists(os.path.join(server, f'{track}_{kind}')), f'{track}_{kind} was not put in the server'
    assert os.path.exists(os.path.join(server, 'content', 'tracks', track, 'ai', 'fast_lane.ai')), 'no AI line in the server'

    car_file = os.path.join(plugin_dir, 'cars', f'{CAR}.json')
    with open(car_file) as f:
        numbers = json.load(f)
    assert numbers['_car'] == CAR and 0.3 < numbers['PACE_SCALE'] < 3.0, numbers.get('PACE_SCALE')

    entries = open(os.path.join(server, 'cfg', 'entry_list.ini')).read()
    assert entries.count('AI=fixed') == 2 and '[CAR_2]' in entries
    server_cfg = open(os.path.join(server, 'cfg', 'server_cfg.ini')).read()
    assert 'MAX_CLIENTS=3' in server_cfg
    with open(os.path.join(server, 'cfg', 'extra_cfg.yml')) as f:
        docs = [d for d in YAML(typ='rt').load_all(f.read()) if d is not None]
    main, config = docs[0], docs[1]
    assert main['EnableAi'] is True and 'GradientDriftPlugin' in main['EnablePlugins']
    assert main['AiParams']['MaxPlayerCount'] == 1 and main['AiParams']['NamePrefix'] == 'TestBot'
    assert config['EnableLearnedPolicy'] is True and float(config['BotPacePercent']) == 90.0
    assert 'NoSuchSetting' not in config

    again = plan(world)
    assert again.ready, 'a second run still finds changes: ' + '; '.join(c.what for c in again.changes)

    undo(server, log=lambda *_: None)
    after = {k: v for k, v in originals(server).items() if not k.startswith('gradient-drift-setup-backups')}
    assert sorted(after) == sorted(before), 'undo left files behind or lost some'
    changed = [k for k in before if after[k] != before[k]]
    assert not changed, f'undo did not put back {changed}'


def test_setup_refuses_a_kunos_server(world):
    server = world[0]
    os.remove(os.path.join(server, 'AssettoServer'))
    with open(os.path.join(server, 'acServer'), 'wb') as f:
        f.write(b'\0the game\'s own server\0')
    p = plan(world)
    assert p.blockers and 'AssettoServer' in p.blockers[0]


def test_setup_keeps_a_policy_made_for_the_server(world):
    """A policy beside the same plugin that no release shipped was made for the server: it stays."""
    from gradientdrift_setup.apply import apply_plan
    server, _, plugin, _, _ = world
    apply_plan(plan(world), log=lambda *_: None)
    own = os.path.join(server, 'plugins', 'GradientDriftPlugin', 'GradientDriftPlugin.policy.json')
    with open(own) as f:
        doc = json.load(f)
    doc['trained_ticks'] = 1
    with open(own, 'w') as f:
        json.dump(doc, f)
    p = plan(world)
    assert not any(c.file.endswith('policy.json') for c in p.changes)
    assert any('kept' in n for n in p.notes)


def test_setup_writes_the_players_cars_too(world):
    """CarContactPush weighs a player's car by its model's car file: setup writes one for every model
    in the entry list, not only the bots'."""
    from gradientdrift_setup.apply import apply_plan
    server, game, plugin, data, track = world
    other = 'test_player_car'
    car = os.path.join(game, 'content', 'cars', other)
    os.makedirs(os.path.join(car, 'data'))
    for name, text in CAR_FILES.items():
        if name == 'car.ini':
            text = text.replace('TOTALMASS=1150', 'TOTALMASS=1450')
        with open(os.path.join(car, 'data', name), 'w') as f:
            f.write(text)
    cfg = os.path.join(server, 'cfg')
    with open(os.path.join(cfg, 'server_cfg.ini')) as f:
        text = f.read()
    with open(os.path.join(cfg, 'server_cfg.ini'), 'w') as f:
        f.write(text.replace(f'CARS={CAR}', f'CARS={CAR};{other}').replace('MAX_CLIENTS=1', 'MAX_CLIENTS=2'))
    with open(os.path.join(cfg, 'entry_list.ini'), 'a') as f:
        f.write(f'[CAR_1]\nMODEL={other}\nSKIN=\nSPECTATOR_MODE=0\nDRIVERNAME=\nTEAM=\nGUID=\nBALLAST=0\nRESTRICTOR=0\n')

    p = plan(world)
    assert not p.blockers, p.blockers
    assert other not in {m for _, m, _ in p.slots}, 'the test meant the bots to drive the other car'
    apply_plan(p, log=lambda *_: None)
    cars = os.path.join(server, 'plugins', 'GradientDriftPlugin', 'cars')
    with open(os.path.join(cars, f'{other}.json')) as f:
        theirs = json.load(f)
    with open(os.path.join(cars, f'{CAR}.json')) as f:
        ours = json.load(f)
    mass = lambda n: (n['FRONT_STATIC_LOAD_N'] + n['REAR_STATIC_LOAD_N']) / 9.80665
    assert mass(theirs) > mass(ours) + 200, (mass(theirs), mass(ours))
    assert plan(world).ready, 'a second run still finds changes'


def test_a_pace_reference_for_another_line_is_made_again(world):
    """A rebuilt AI line can gain or lose a point: the pace reference made for the old line is found
    (setup then makes it again) and remade for the new one, the old kept beside it."""
    import drift_env as de
    import fast_sim as fs
    import track_files as tf
    from gradientdrift_setup.apply import pace_for_another_line
    _, _, _, data, track = world
    ref = tf.pace_reference_path(data, track)
    assert pace_for_another_line(data, track) is None
    with open(ref) as f:
        lines = f.readlines()
    with open(ref, 'w') as f:
        f.writelines(lines[:-1])                     # made for a line one point shorter
    assert pace_for_another_line(data, track) == ref
    world_ = fs.load_world(data, track=track)
    kmh = de.course_pace_reference(data, track, world_)
    assert len(kmh) == len(world_.sp_pos)
    assert os.path.exists(ref + '.before_new_line')
    assert pace_for_another_line(data, track) is None
