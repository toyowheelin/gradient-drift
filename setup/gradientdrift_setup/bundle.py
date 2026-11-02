"""The plugin's data in one file: plugins/GradientDriftPlugin/GradientDriftPlugin.bundle.zip.

The plugin (PluginData.cs) reads its policy, the bots' car files and the track's data from the
bundle, except where a loose copy is there: that one is read instead (a policy copied in by hand
takes over). The bundle holds them under the same names:
the plugin folder's files by their path in it (GradientDriftPlugin.policy.json, cars/<model>.json),
the track's by their file name (<track>_triangles.csv.gz, ...), so it can carry several tracks.

Setup reads a server's files the same way (Server.data_path), and apply writes the files it changes
into the bundle instead of loose and packs any loose ones into it; those go into that run's backup,
so undo puts them back.
"""
import os
import zipfile

PLUGIN = 'GradientDriftPlugin'
BUNDLE = f'{PLUGIN}.bundle.zip'
BUNDLE_REL = f'plugins/{PLUGIN}/{BUNDLE}'
POLICY = f'{PLUGIN}.policy.json'
# plan.TRACK_FILES' endings: the track's data in the server's folder
TRACK_SUFFIXES = ('_triangles.csv.gz', '_collision.csv.gz', '_road_mesh.csv', '_course_edges.csv', '_pace_reference.csv',
                  '_collision_grip.csv.gz')
# the text the plugin's DLL carries (UTF-16, as .NET keeps its strings) once it reads a bundle
_MARK = BUNDLE.encode('utf-16-le')


def entry_name(rel):
    """The bundle's name for a server file (its path relative to the server folder, with /), or None
    for a file that does not go in the bundle."""
    plugin = f'plugins/{PLUGIN}/'
    if rel.startswith(plugin):
        name = rel[len(plugin):]
        if name == POLICY or (name.startswith('cars/') and name.count('/') == 1 and name.endswith('.json')):
            return name
        return None
    if '/' not in rel and rel.endswith(TRACK_SUFFIXES):
        return rel
    return None


def rel_of(name):
    """Where a bundle entry would sit loose in the server folder, or None for a name setup does not know."""
    if name == POLICY or (name.startswith('cars/') and name.count('/') == 1 and name.endswith('.json')):
        rel = f'plugins/{PLUGIN}/{name}'
    elif '/' not in name and name.endswith(TRACK_SUFFIXES):
        rel = name
    else:
        return None
    return rel if entry_name(rel) == name else None


def loose_files(folder):
    """The server's loose files that belong in the bundle, as paths relative to the folder."""
    out = []
    plugin = os.path.join(folder, 'plugins', PLUGIN)
    if os.path.isfile(os.path.join(plugin, POLICY)):
        out.append(f'plugins/{PLUGIN}/{POLICY}')
    cars = os.path.join(plugin, 'cars')
    if os.path.isdir(cars):
        out += [f'plugins/{PLUGIN}/cars/{n}' for n in sorted(os.listdir(cars))
                if n.endswith('.json') and os.path.isfile(os.path.join(cars, n))]
    if os.path.isdir(folder):
        out += sorted(n for n in os.listdir(folder) if n.endswith(TRACK_SUFFIXES) and os.path.isfile(os.path.join(folder, n)))
    return out


def read(path):
    """{entry name: bytes} of a bundle. zipfile.BadZipFile or OSError if it cannot be read."""
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}


def names(path):
    with zipfile.ZipFile(path) as z:
        return [i.filename for i in z.infolist() if not i.is_dir()]


def write(path, entries):
    """Writes {entry name: bytes} as the bundle at path (through a temporary file beside it, so a
    failure leaves the old bundle as it was). Gzip files are stored as they are."""
    tmp = path + '.writing'
    with zipfile.ZipFile(tmp, 'w') as z:
        for name in sorted(entries):
            kind = zipfile.ZIP_STORED if name.endswith('.gz') else zipfile.ZIP_DEFLATED
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), entries[name], compress_type=kind)
    os.replace(tmp, path)


def plugin_reads_bundle(dll):
    """Whether a GradientDriftPlugin.dll reads the bundle (a build older than it reads loose files only):
    its features file says so, or the DLL carries the bundle's name."""
    from .pairing import load_features
    features = load_features(os.path.dirname(dll))
    if features is not None:
        return bool(features.get('bundle'))
    try:
        with open(dll, 'rb') as f:
            return _MARK in f.read()
    except OSError:
        return False
