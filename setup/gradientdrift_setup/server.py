"""What an AssettoServer folder holds: its track, its cars and entries, whether AI and the drift
plugin are on, and the files the bot needs."""
import atexit
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field

from ruamel.yaml import YAML

from . import bundle
from .acini import AcIni

PLUGIN = 'GradientDriftPlugin'
PLUGIN_CONFIG_TAG = '!GradientDriftConfiguration'
# the plugin's name before the rename: a server set up with it is moved to the new name
OLD_PLUGIN = 'DriftAiPlugin'
OLD_PLUGIN_CONFIG_TAG = '!DriftAiPluginConfiguration'


@dataclass
class Entry:
    index: int
    model: str
    skin: str
    ai: str          # '' (a player slot), 'auto' or 'fixed'
    name: str


@dataclass
class Server:
    folder: str
    name: str = ''
    track: str = ''
    layout: str = ''
    cars: list = field(default_factory=list)          # server_cfg.ini CARS=
    max_clients: int = 0
    entries: list = field(default_factory=list)
    enable_ai: bool = False
    plugins: list = field(default_factory=list)       # extra_cfg.yml EnablePlugins
    ai_max_player_count: int | None = None
    ai_params: dict = field(default_factory=dict)     # extra_cfg.yml AiParams' plain values
    plugin_config: dict | None = None                 # the --- !GradientDriftConfiguration document
    plugin_files: list = field(default_factory=list)  # plugins/GradientDriftPlugin/*
    old_plugin: bool = False                          # set up under the old name (OLD_PLUGIN)
    old_plugin_doc: bool = False                      # its settings under OLD_PLUGIN_CONFIG_TAG
    old_policy: str = ''                              # plugins/DriftAiPlugin/DriftAiPlugin.policy.json, if there
    problems: list = field(default_factory=list)      # things that stop setup reading the folder
    bundled: dict = field(default_factory=dict)       # server path -> its copy unpacked from the plugin's bundle
    bundle_dir: str = ''                              # where the bundle was unpacked (laid out like the server)

    def data_path(self, rel):
        """Where to read one of the plugin's data files (a path relative to the server folder): the
        loose file if there is one, else its copy from the bundle (as the plugin reads them)."""
        loose = os.path.join(self.folder, *rel.split('/'))
        return loose if os.path.exists(loose) else self.bundled.get(rel, loose)

    @property
    def data_folder(self):
        """A folder holding the track's data files as the plugin sees them (the loose file where there
        is one, else the bundle's copy): the server folder itself when nothing is in a bundle."""
        if not self.bundled:
            return self.folder
        view = self.bundle_dir
        for name in os.listdir(self.folder):
            if name.endswith(bundle.TRACK_SUFFIXES) and os.path.isfile(os.path.join(self.folder, name)):
                shutil.copy2(os.path.join(self.folder, name), os.path.join(view, name))
        return view

    @property
    def cfg(self):
        return os.path.join(self.folder, 'cfg')

    @property
    def bots(self):
        return [e for e in self.entries if e.ai in ('auto', 'fixed')]

    def server_track_folder(self):
        return os.path.join(self.folder, 'content', 'tracks', self.track)

    def server_ai_line(self):
        base = self.server_track_folder()
        return os.path.join(base, self.layout, 'ai', 'fast_lane.ai') if self.layout else os.path.join(base, 'ai', 'fast_lane.ai')


def read_server(folder):
    s = Server(folder=folder)
    for needed in ('server_cfg.ini', 'entry_list.ini'):
        if not os.path.exists(os.path.join(s.cfg, needed)):
            s.problems.append(f"no cfg/{needed} - is this an AssettoServer folder?")
    if s.problems:
        return s

    server = AcIni(os.path.join(s.cfg, 'server_cfg.ini')).section('SERVER')
    s.name = server.get('NAME', '')
    s.track = server.get('TRACK', '')
    s.layout = server.get('CONFIG_TRACK', '')
    s.cars = [c for c in server.get('CARS', '').split(';') if c]
    s.max_clients = int(server.get('MAX_CLIENTS', '0') or 0)

    for sec, values in AcIni(os.path.join(s.cfg, 'entry_list.ini')).sections():
        m = re.fullmatch(r'CAR_(\d+)', sec)
        if m:
            s.entries.append(Entry(int(m.group(1)), values.get('MODEL', ''), values.get('SKIN', ''),
                                   values.get('AI', '').lower(), values.get('DRIVERNAME', '')))

    extra = os.path.join(s.cfg, 'extra_cfg.yml')
    if os.path.exists(extra):
        with open(extra, encoding='utf-8-sig') as f:
            docs = [d for d in YAML(typ='rt').load_all(f.read()) if d is not None]
        main = docs[0] if docs else {}
        s.enable_ai = bool(main.get('EnableAi', False))
        s.plugins = list(main.get('EnablePlugins') or [])
        ai_params = main.get('AiParams') or {}
        s.ai_params = {k: v for k, v in ai_params.items() if isinstance(v, (int, float, str, bool))}
        if 'MaxPlayerCount' in ai_params:
            s.ai_max_player_count = int(ai_params['MaxPlayerCount'])
        for d in docs[1:]:
            tag = getattr(getattr(d, 'tag', None), 'value', None)
            if tag == PLUGIN_CONFIG_TAG:
                s.plugin_config = dict(d)
            elif tag == OLD_PLUGIN_CONFIG_TAG and s.plugin_config is None:
                s.plugin_config = dict(d)          # renamed by setup with the rest
                s.old_plugin = s.old_plugin_doc = True
        if OLD_PLUGIN in s.plugins:
            s.old_plugin = True
    else:
        s.problems.append("no cfg/extra_cfg.yml - start the server once so AssettoServer writes it")

    plugin_dir = os.path.join(folder, 'plugins', PLUGIN)
    if os.path.isdir(plugin_dir):
        s.plugin_files = sorted(os.listdir(plugin_dir))
    bundle_path = os.path.join(plugin_dir, bundle.BUNDLE)
    if os.path.isfile(bundle_path):
        try:
            s.bundle_dir, s.bundled = _unpack(bundle_path)
        except (OSError, zipfile.BadZipFile) as e:
            s.problems.append(f"plugins/{PLUGIN}/{bundle.BUNDLE} cannot be read ({e}): move it away, or put back "
                              "the one from setup's last backup")
        if f'plugins/{PLUGIN}/{bundle.POLICY}' in s.bundled and bundle.POLICY not in s.plugin_files:
            s.plugin_files = sorted(s.plugin_files + [bundle.POLICY])
    old_policy = os.path.join(folder, 'plugins', OLD_PLUGIN, OLD_PLUGIN + '.policy.json')
    if os.path.exists(old_policy):
        s.old_policy = old_policy
    return s


def _unpack(path):
    """The bundle's files setup knows, unpacked into a folder of their own laid out like the server's
    that goes when setup ends: (that folder, {server path: unpacked copy})."""
    out = tempfile.mkdtemp(prefix='gradient-drift-bundle-')
    atexit.register(shutil.rmtree, out, True)
    files = {}
    for name, data in bundle.read(path).items():
        rel = bundle.rel_of(name)
        if rel is None:
            continue
        dst = os.path.join(out, *rel.split('/'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(data)
        files[rel] = dst
    return out, files


def server_running(folder):
    """Whether AssettoServer looks to be running from this folder, so its plugin cannot be replaced:
    on Windows the loaded plugin cannot be opened for writing; on Linux a process's program, or the
    AssettoServer.dll dotnet runs, is in the folder. False when it cannot tell."""
    folder = os.path.realpath(folder)
    if sys.platform == 'win32':
        dll = os.path.join(folder, 'plugins', PLUGIN, PLUGIN + '.dll')
        try:
            with open(dll, 'r+b'):
                return False
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
    if not os.path.isdir('/proc'):
        return False

    def inside(path):
        return os.path.commonpath([folder, os.path.realpath(path)]) == folder

    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            exe = os.readlink(f'/proc/{pid}/exe')
            if os.path.basename(exe).startswith('AssettoServer') and inside(exe):
                return True
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                args = [a.decode(errors='replace') for a in f.read().split(b'\0') if a]
            cwd = os.readlink(f'/proc/{pid}/cwd')
            if any(os.path.basename(a) == 'AssettoServer.dll' and inside(os.path.join(cwd, a)) for a in args):
                return True
        except (OSError, ValueError):
            continue
    return False
