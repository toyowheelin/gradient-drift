"""Finding the Assetto Corsa install and AssettoServer folders without being told."""
import os
import re
import sys

STEAM_DEFAULTS = [r'C:\Program Files (x86)\Steam', r'C:\Program Files\Steam',
                  os.path.expanduser('~/.steam/steam'), os.path.expanduser('~/.local/share/Steam')]
OTHER_PLACES = [r'C:\assettocorsa', r'C:\Games\assettocorsa', r'D:\assettocorsa']


def _bases():
    """Setup's own folder (the packaged program's, else the repository's)."""
    if getattr(sys, 'frozen', False):
        return [os.path.dirname(sys.executable)]
    return [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))]


def default_plugin_source():
    """The plugin's build: plugin/ beside the packaged program, else the repository's build folder."""
    for base in _bases():
        for sub in ('plugin', *(os.path.join(*parent, 'GradientDriftPlugin', 'bin', build, 'net8.0')
                                for parent in ((), ('AssettoServer',)) for build in ('Release', 'Debug'))):
            folder = os.path.join(base, sub)
            if os.path.exists(os.path.join(folder, 'GradientDriftPlugin.dll')):
                return folder
    return None


def default_track_data():
    """Track files (setup_track.py's output, one folder per track): tracks/ beside the packaged
    program, else the repository's drift_rl/tracks/."""
    for base in _bases():
        for sub in ('tracks', os.path.join('drift_rl', 'tracks')):
            folder = os.path.join(base, sub)
            if os.path.isdir(folder):
                return folder
    return None


def track_data_root(folder):
    """The track files folder (setup_track.py's output, a folder per track) for a folder the operator
    picked: the folder itself, or its parent when it is one track's own folder (drift_rl/tracks/<track>).
    None when neither holds a track's data (a game track's folder, say)."""
    def holds(d):
        try:
            return any(n.endswith(('_triangles.csv.gz', '_fast_lane.ai')) for n in os.listdir(d))
        except OSError:
            return False
    if not os.path.isdir(folder):
        return None
    if holds(folder):
        return os.path.dirname(os.path.normpath(folder))
    if any(holds(os.path.join(folder, sub)) for sub in os.listdir(folder)):
        return os.path.normpath(folder)
    return None


def in_game_content(folder):
    """Whether a folder is inside a game or server's content/tracks (a track's own files, not
    setup_track.py's output)."""
    parts = [x.lower() for x in os.path.normpath(folder).replace('\\', '/').split('/')]
    return any(a == 'content' and b == 'tracks' for a, b in zip(parts, parts[1:]))


def track_data_home():
    """Where setup keeps the track data it makes: tracks/ beside the packaged program, else the
    repository's drift_rl/tracks/ (made if missing)."""
    return default_track_data() or os.path.join(_bases()[0], 'tracks' if getattr(sys, 'frozen', False)
                                                 else os.path.join('drift_rl', 'tracks'))


def is_game(folder):
    return os.path.isdir(os.path.join(folder, 'content', 'tracks')) and os.path.isdir(os.path.join(folder, 'content', 'cars'))


def is_server(folder):
    return os.path.exists(os.path.join(folder, 'cfg', 'server_cfg.ini'))


def steam_libraries():
    libs = []
    for steam in STEAM_DEFAULTS:
        vdf = os.path.join(steam, 'steamapps', 'libraryfolders.vdf')
        if os.path.exists(vdf):
            libs.append(steam)
            with open(vdf, encoding='utf-8', errors='replace') as f:
                libs += [p.replace('\\\\', '\\') for p in re.findall(r'"path"\s+"([^"]+)"', f.read())]
    return list(dict.fromkeys(libs))


def find_games():
    found = [os.path.join(lib, 'steamapps', 'common', 'assettocorsa') for lib in steam_libraries()]
    found += OTHER_PLACES
    return [f for f in dict.fromkeys(found) if is_game(f)]


def find_servers(near=()):
    """Server folders in or beside the given folders (the usual places to unpack AssettoServer); the
    game's own dedicated server (its server/ folder) is among them and is told apart later."""
    out = []
    for base in near:
        for root in (base, os.path.dirname(os.path.normpath(base))):
            if not os.path.isdir(root):
                continue
            if is_server(root):
                out.append(root)
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and is_server(sub):
                    out.append(sub)
    return list(dict.fromkeys(os.path.normpath(p) for p in out))


def gui_possible():
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
                                                  or os.environ.get('QT_QPA_PLATFORM')):
        return False
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True
