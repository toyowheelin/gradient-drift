"""Makes a plan's changes, with a backup of everything first, and undoes the last run.

Before a file is changed or replaced its original goes to
<server>/gradient-drift-setup-backups/<date_time>/<its path>, and that run's manifest.json lists what
was changed and what was new. undo() puts the originals back and moves the files the run added
into the same backup folder (under added/), so nothing is ever deleted. Edited files are checked
after editing (the .ini and YAML parsed back, the values read) before anything is written; one
that fails stops the run with nothing changed.
"""
import json
import os
import shutil
import time

from ruamel.yaml import YAML

from . import edits
from .acini import AcIni
from .server import server_running

BACKUPS = 'gradient-drift-setup-backups'
OLD_BACKUPS = 'driftai-setup-backups'     # setup's folder before the rename
EXTRA = 'cfg/extra_cfg.yml'


def _abs(server, rel):
    return os.path.join(server, *rel.split('/'))


def _edit(tf, op):
    kind = op[0]
    if kind == 'ini':
        edits.ini_set(tf, op[2], op[3], op[4])
    elif kind == 'ini_keys':
        for key, value in op[3]:
            edits.ini_set(tf, op[2], key, value)
    elif kind == 'append':
        edits.append_block(tf, op[2])
    elif kind == 'yaml_top':
        edits.yaml_set_top(tf, op[1], op[2])
    elif kind == 'yaml_plugin':
        edits.yaml_add_plugin(tf, op[1])
    elif kind == 'yaml_child':
        edits.yaml_set_child(tf, op[1], op[2], op[3])
    elif kind == 'yaml_doc':
        edits.yaml_plugin_doc(tf, op[1], op[2], op[3])
    elif kind == 'yaml_rename_plugin':
        edits.yaml_rename_plugin(tf, op[1], op[2])
    elif kind == 'yaml_rename_doc':
        edits.yaml_rename_doc(tf, op[1], op[2])
    elif kind == 'yaml_doc_rename':
        edits.yaml_plugin_doc_rename(tf, op[1], op[2], op[3])
    elif kind == 'yaml_doc_remove':
        edits.yaml_plugin_doc_remove(tf, op[1], op[2])
    else:
        raise ValueError(f'unknown edit {kind}')


def _file_of(op):
    return EXTRA if op[0].startswith('yaml') else op[1]


def _check_yaml(text, ops):
    docs = [d for d in YAML(typ='rt').load_all(text) if d is not None]
    main = docs[0]
    for op in ops:
        kind = op[0]
        if kind == 'yaml_top':
            assert str(main.get(op[1])).lower() == op[2].lower(), f'{op[1]} reads back as {main.get(op[1])}'
        elif kind == 'yaml_plugin':
            assert op[1] in (main.get('EnablePlugins') or []), 'EnablePlugins does not read back with the plugin'
        elif kind == 'yaml_child':
            got = (main.get(op[1]) or {}).get(op[2])
            assert str(got).lower() == op[3].lower(), f'{op[1]}.{op[2]} reads back as {got}'
        elif kind == 'yaml_doc':
            doc = next((d for d in docs[1:] if getattr(getattr(d, 'tag', None), 'value', None) == op[1]), None)
            assert doc is not None and str(doc.get(op[2])).lower() == op[3].lower(), f'{op[1]} {op[2]} does not read back'
        elif kind == 'yaml_rename_plugin':
            plugins = main.get('EnablePlugins') or []
            assert op[2] in plugins and op[1] not in plugins, 'EnablePlugins does not read back renamed'
        elif kind == 'yaml_rename_doc':
            tags = [getattr(getattr(d, 'tag', None), 'value', None) for d in docs[1:]]
            assert op[2] in tags and op[1] not in tags, f'{op[2]} does not read back'
        elif kind in ('yaml_doc_rename', 'yaml_doc_remove'):
            doc = next((d for d in docs[1:] if getattr(getattr(d, 'tag', None), 'value', None) == op[1]), None)
            assert doc is not None and op[2] not in doc, f'{op[1]} {op[2]} is still there'
            if kind == 'yaml_doc_rename':
                assert op[3] in doc, f'{op[1]} {op[3]} does not read back'


def _check_ini(path_text, ops):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False, encoding='utf-8') as f:
        f.write(path_text)
        tmp = f.name
    try:
        ini = AcIni(tmp)
        for op in ops:
            if op[0] == 'ini':
                got = ini.section(op[2]).get(op[3].upper())
                assert got == op[4], f'{op[3]} reads back as {got}'
            elif op[0] == 'ini_keys':
                for key, value in op[3]:
                    got = ini.section(op[2]).get(key.upper())
                    assert got == value, f'[{op[2]}] {key} reads back as {got}'
            elif op[0] == 'append':
                first = op[2].splitlines()[0].strip('[]').upper()
                assert first in [s for s, _ in ini.sections()], f'{first} not found after adding it'
    finally:
        os.unlink(tmp)


def make_track_data(op, server, log=print):
    """Runs drift_rl/setup_track.py for a track_data change, into the track data folder (never the
    server's). Returns the AI line copy the server then needs (rebuilt, or one it lacks), or None."""
    _, game_track, layout, home, name, rel_ai = op
    from . import game  # noqa: F401  (puts drift_rl on the path)
    import setup_track
    log(f"  making {name}'s data (a few minutes)")
    argv = [game_track, '--tracks', home, '--name', name, '--for-setup'] + (['--layout', layout] if layout else [])
    try:
        made = setup_track.main(argv)
    except SystemExit as e:
        raise RuntimeError(f"making {name}'s data stopped: {e}") from None
    if rel_ai and (made['line_rebuilt'] or not os.path.exists(_abs(server, rel_ai))):
        return made['line']
    return None


def make_pace_reference(op, log=print):
    """Makes a track's pace reference for a pace_reference change, into the track data folder
    (drift_env.course_pace_reference: drift_rl/pace_model.py, from the track's shape, as training
    makes it; an older file there is kept as .before_pace_model). Returns the file it wrote."""
    _, _policy, home, name = op
    import fast_sim as fs
    import drift_env as de
    import track_files as tf
    log(f"  working out {name}'s reference speed round the lap")
    try:
        world = fs.load_world(home, track=name)
        de.course_pace_reference(home, name, world)
    except (FileNotFoundError, RuntimeError, SystemExit) as e:
        raise RuntimeError(f"making {name}'s pace reference stopped: {e}") from None
    return tf.pace_reference_path(home, name)


def check_track_data(op, log=print):
    """drift_rl/check_track.py on the track's data in the track data folder: a RuntimeError with what
    is wrong, before anything in the server is written."""
    _, policy, home, name, pace_file = (op + (None,))[:5]
    import check_track
    try:
        problems = check_track.check(home, name, policy, log=lambda line: log(f"  {line}"), pace_file=pace_file)
    except (FileNotFoundError, RuntimeError, SystemExit) as e:
        raise RuntimeError(f"checking {name}'s data stopped: {e}") from None
    if problems:
        raise RuntimeError(check_track.report(name, problems)
                           + f"\nNothing in the server was changed. The track's data is in {os.path.join(home, name)}.")


def apply_plan(plan, log=print):
    """Makes the plan's changes; returns the backup folder of this run."""
    server = plan.server.folder
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = os.path.join(server, BACKUPS, stamp)
    edits_by_file, copies, builds, paces, writes, checks = {}, [], [], [], [], []
    for c in plan.changes:
        if not c.op:
            continue
        if c.op[0] == 'copy':
            copies.append(c)
        elif c.op[0] in ('write', 'car_file'):
            writes.append(c)
        elif c.op[0] == 'track_data':
            builds.append(c)
        elif c.op[0] == 'pace_reference':
            paces.append(c)
        elif c.op[0] == 'track_check':
            checks.append(c)
        else:
            edits_by_file.setdefault(_file_of(c.op), []).append(c.op)

    # a plugin update replaces files AssettoServer has open while it runs: checked again here, in
    # case the server was started after the plan was made
    if any(c.op[3:4] == ('plugin',) for c in copies) and server_running(server):
        raise RuntimeError("AssettoServer is running from this folder (the plugin is in use): stop it, then run "
                           "setup again. Nothing was changed.")

    # the track's data first: it writes only into the track data folder, and if it fails nothing
    # in the server has been touched
    for c in builds:
        line = make_track_data(c.op, server, log)
        if line:
            copies.append(type(c)(c.op[5], "the track's AI line", op=('copy', line, c.op[5])))
    # then its pace reference, from that data (also only into the track data folder)
    for c in paces:
        made = make_pace_reference(c.op, log)
        copies.append(type(c)(c.file, "the track's pace reference", op=('copy', made, c.file)))
    # and what it asks of the bots
    for c in checks:
        check_track_data(c.op, log)

    # each car's pace scale on this track, measured before anything in the server is written
    for i, c in enumerate(writes):
        if c.op[0] == 'car_file':
            from .carfile import measure_pace_scale
            rel, text, source = c.op[1:]
            log(f"  measuring {os.path.basename(rel)[:-5]}'s pace scale on {source[2]}")
            try:
                text = measure_pace_scale(text, source)
            except (RuntimeError, ValueError, OSError, SystemExit) as e:
                log(f"  could not measure it ({e}); PACE_SCALE stays 1")
            writes[i] = type(c)(rel, c.what, after=text, op=('write', rel, text))

    # everything edited in memory and checked before anything is written
    new_text = {}
    for rel, ops in edits_by_file.items():
        tf = edits.TextFile(_abs(server, rel))
        for op in ops:
            _edit(tf, op)
        if rel.endswith('.yml'):
            _check_yaml(tf.text(), ops)
        else:
            _check_ini(tf.text(), ops)
        new_text[rel] = tf

    manifest = {'when': stamp, 'changed': [], 'added': []}

    def keep(rel):
        path = _abs(server, rel)
        if os.path.exists(path):
            dst = os.path.join(backup, *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)
            manifest['changed'].append(rel)
        else:
            manifest['added'].append(rel)

    os.makedirs(backup, exist_ok=True)
    try:
        for rel, tf in new_text.items():
            keep(rel)
            with open(_abs(server, rel), 'wb') as f:
                f.write(tf.data())
            log(f'  changed {rel}')
        for c in copies:
            src, rel = c.op[1], c.op[2]
            keep(rel)
            os.makedirs(os.path.dirname(_abs(server, rel)), exist_ok=True)
            shutil.copy2(src, _abs(server, rel))
            log(f'  copied  {rel}')
        for c in writes:
            rel, text = c.op[1], c.op[2]
            keep(rel)
            os.makedirs(os.path.dirname(_abs(server, rel)), exist_ok=True)
            with open(_abs(server, rel), 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            log(f'  wrote   {rel}')
    finally:
        with open(os.path.join(backup, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=1)
    log(f'  originals kept in {os.path.relpath(backup, server)}')
    return backup


def last_run(server):
    runs = []
    for name in (BACKUPS, OLD_BACKUPS):      # runs from before the rename are undone too
        root = os.path.join(server, name)
        if os.path.isdir(root):
            runs += [(d, os.path.join(root, d)) for d in os.listdir(root)
                     if os.path.exists(os.path.join(root, d, 'manifest.json'))
                     and not os.path.exists(os.path.join(root, d, 'undone'))]
    return max(runs)[1] if runs else None


def undo(server, log=print):
    """Puts back what the last (not yet undone) run changed; returns its backup folder or None."""
    run = last_run(server)
    if run is None:
        log('  nothing to undo')
        return None
    with open(os.path.join(run, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    for rel in manifest['changed']:
        shutil.copy2(os.path.join(run, *rel.split('/')), _abs(server, rel))
        log(f'  restored {rel}')
    for rel in manifest['added']:
        path = _abs(server, rel)
        if os.path.exists(path):
            dst = os.path.join(run, 'added', *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(path, dst)
            log(f'  moved away {rel} (it was new; kept in {os.path.relpath(dst, server)})')
    with open(os.path.join(run, 'undone'), 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S\n'))
    return run
