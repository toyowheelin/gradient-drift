"""For a server setup cannot reach (a hosting plan, a machine without the game): the plan's changes
made into a copy of the server's files, and what changed put in an export folder to upload.

The operator downloads the server's cfg/ folder (server_cfg.ini, entry_list.ini, extra_cfg.yml) from
their host into a folder on the PC with the game, and points setup at that folder. export() copies
it to a scratch folder, plans and applies there exactly as apply would (the downloaded files are
never changed), and writes:

    <out>/upload/...            every file the run changed or added, at its path in the server folder
    <out>/HOW_TO_INSTALL.txt    what to upload where, the same config edits as lines to make by hand,
                                and setup's notes
"""
import json
import os
import shutil
import tempfile
import time

from .apply import BACKUPS, OLD_BACKUPS, apply_plan
from .identify import TESTED_VERSION
from .plan import TRACK_FILES, plan_bot
from .server import PLUGIN, PLUGIN_CONFIG_TAG, read_server

INSTRUCTIONS = 'HOW_TO_INSTALL.txt'
UPLOAD = 'upload'


def _copy_server(src, dst, server):
    """The parts of a server folder setup reads: cfg/, the plugin's folder, the track's data files
    and its AI line (a whole server folder can carry gigabytes of content; a downloaded one has cfg/ only)."""
    shutil.copytree(os.path.join(src, 'cfg'), os.path.join(dst, 'cfg'))
    plugin = os.path.join(src, 'plugins', PLUGIN)
    if os.path.isdir(plugin):
        shutil.copytree(plugin, os.path.join(dst, 'plugins', PLUGIN))
    for pattern in TRACK_FILES:
        name = pattern.format(track=server.track)
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    line = server.server_ai_line()
    if os.path.exists(line):
        rel = os.path.relpath(line, src)
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copy2(line, os.path.join(dst, rel))


def _where(op):
    """Where in its file a config edit goes, for someone making it by hand."""
    kind = op[0] if op else ''
    if kind == 'ini':
        return f"in the [{op[2]}] section, change the line"
    if kind == 'ini_keys':
        return f"in the [{op[2]}] section, change the lines"
    if kind == 'append':
        return "add these lines at the end of the file"
    if kind == 'yaml_child':
        return f"under {op[1]}: (indented like the lines beside it; add the line if it is not there)"
    if kind == 'yaml_doc':
        return f"in the --- {op[1]} section at the end of the file (add the line if it is not there)"
    if kind in ('yaml_top', 'yaml_plugin', 'yaml_rename_plugin', 'yaml_rename_doc'):
        return "change the line"
    return ''


def _label(rel):
    """What an uploaded file is, in a few words."""
    if rel.startswith('cfg/'):
        return "your config file with setup's edits (step 3)"
    if rel.startswith(f'plugins/{PLUGIN}/cars/'):
        return f"the numbers the bots in {rel.rsplit('/', 1)[1][:-5]} drive with"
    if rel.startswith(f'plugins/{PLUGIN}/'):
        return "the plugin"
    if rel.startswith('content/'):
        return ("the track's AI line, which the plugin's track data follows point by point: it replaces the "
                "server's own line if it has one")
    return "the track's data for the plugin"


def _hand_edits(changes):
    """The lines for one config file's edits, made by hand."""
    out = []
    # the plugin's section is new: one block with every setting setup gives it
    new_doc = any(c.op[0] == 'yaml_doc' and c.after.startswith(f'--- {PLUGIN_CONFIG_TAG}') for c in changes)
    if new_doc:
        lines = [f'--- {PLUGIN_CONFIG_TAG}'] + [f'{c.op[2]}: {c.op[3]}' for c in changes if c.op[0] == 'yaml_doc']
        changes = [c for c in changes if c.op[0] != 'yaml_doc']
    for c in changes:
        before, after = c.before, c.after
        if c.op[0] == 'yaml_child':          # "AiParams.Key: value" is the line "Key: value" under AiParams:
            before, after = (t.replace(f'{c.op[1]}.', '', 1) for t in (before, after))
        out.append(f"     - {c.what}: {_where(c.op)}")
        if before and after:
            out += [f"         {line}" for line in before.splitlines()] + ["       becomes"]
            out += [f"         {line}" for line in after.splitlines()]
        else:
            out += [f"         {line}" for line in after.splitlines()]
        out.append("")
    if new_doc:
        out.append("     - the plugin's own settings: add these lines at the end of the file")
        out += [f"         {line}" for line in lines] + [""]
    return out


def instructions(plan, files, source):
    """HOW_TO_INSTALL.txt: files = the paths (relative to the server folder) in upload/."""
    s = plan.server
    layout = f" ({s.layout})" if s.layout else ''
    cars = ', '.join(dict.fromkeys(model for _, model, _ in plan.slots)) or plan.car
    out = [f"Gradient Drift for {s.name or 'your server'}: {s.track}{layout}, {cars}",
           f"Made by gradient-drift-setup export on {time.strftime('%Y-%m-%d %H:%M')} from {source}.",
           f"The plugin is built for AssettoServer {TESTED_VERSION}; it may not load in another version.",
           "",
           "1. Stop the server.",
           "",
           "2. Upload everything in the upload folder into the server's folder (the one with cfg/ in it),",
           "   keeping the folders: upload/plugins/... goes to plugins/... on the server, and so on. Create",
           "   any folder that is not there yet. Keep the file names exactly as they are, capitals included",
           "   (most hosts run Linux, where GradientDriftPlugin.dll and gradientdriftplugin.dll are different",
           "   files). These are the files:",
           ""]
    for rel in files:
        out.append(f"     {rel}")
        out.append(f"         {_label(rel)}")
    cfg = [c for c in plan.changes if c.file.startswith('cfg/') and c.op and c.op[0] != 'copy']
    if cfg:
        out += ["",
                "3. The config files in upload/cfg/ are the ones you downloaded with only setup's edits made.",
                "   Upload them as they are, unless the server's copies have changed since you downloaded",
                "   them (or your host only lets you edit them in its panel). Then make the same edits by",
                "   hand instead and leave upload/cfg/ out:",
                ""]
        for rel in sorted({c.file for c in cfg}):
            out.append(f"   {rel}")
            out += _hand_edits([c for c in cfg if c.file == rel])
    out += ["",
            f"{4 if cfg else 3}. Start the server. Its log has lines starting \"GradientDrift:\": one per bot car with its",
            "   weight and steering lock. A warning there with \"the stock AI drives\" says what is missing.",
            "",
            "Bots appear only while a player is on the server, near the track's AI line and not idle.",
            "Players need the bots' cars installed to join. To change the bots' skill, pace or trains later,",
            f"edit the --- {PLUGIN_CONFIG_TAG} section of cfg/extra_cfg.yml (SETTINGS.md lists them)",
            "and restart the server."]
    if plan.notes:
        out += ["", "Setup's notes:"] + [f"  - {n}" for n in plan.notes]
    return '\n'.join(out) + '\n'


def export_server(folder, out, game, log=print, **plan_args):
    """Plans the bots for the server files in `folder` (a server folder, or the cfg/ downloaded from
    a host), makes the changes into a scratch copy, and puts the changed and added files in
    <out>/upload/ with <out>/HOW_TO_INSTALL.txt. `folder` is not changed. Returns (plan, files):
    files is None when the plan has blockers (nothing written), else the paths in upload/."""
    out = os.path.abspath(out)
    if os.path.exists(out) and os.listdir(out):
        raise ValueError(f"{out} is not empty: give an empty or new folder for the export")
    if os.path.commonpath([out, os.path.abspath(folder)]) == os.path.abspath(folder):
        raise ValueError("the export folder cannot be inside the server folder")
    server = read_server(folder)
    if server.problems:
        return plan_bot(server, game, **plan_args), None
    scratch = tempfile.mkdtemp(prefix='gradient-drift-export-')
    try:
        _copy_server(folder, scratch, server)
        plan = plan_bot(read_server(scratch), game, **plan_args)
        if plan.blockers or not plan.changes:
            return plan, None
        backup = apply_plan(plan, log=lambda line: log(line.replace(scratch + os.sep, '')))
        with open(os.path.join(backup, 'manifest.json'), encoding='utf-8') as f:
            manifest = json.load(f)
        files = sorted(set(manifest['changed'] + manifest['added']))
        files = [f for f in files if not f.startswith((BACKUPS + '/', OLD_BACKUPS + '/'))]
        for rel in files:
            dst = os.path.join(out, UPLOAD, *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(scratch, *rel.split('/')), dst)
        plan.server = server        # the operator's folder, for the instructions (name, track)
        with open(os.path.join(out, INSTRUCTIONS), 'w', encoding='utf-8', newline='\n') as f:
            f.write(instructions(plan, files, os.path.abspath(folder)))
        return plan, files
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
