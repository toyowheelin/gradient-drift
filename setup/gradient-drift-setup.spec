# PyInstaller recipe for the setup app, one folder (no single-file build: slower start, more
# antivirus false alarms, and Qt's LGPL is simplest to meet with its libraries left as files).
#
#   pyinstaller --noconfirm setup/gradient-drift-setup.spec              (run from the repo root)
#
# One program, gradient-drift-setup: double-clicked it opens the window (on Windows it hides the console
# Windows opens for it, see gradientdrift_setup/console.py); from a terminal it is the command line.
# GRADIENTDRIFT_GUI=0 leaves Qt out (the Linux build does this): command line only.
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
GUI = os.environ.get('GRADIENTDRIFT_GUI', '1') != '0'

# The track tools (drift_rl/setup_track.py and the steps it runs) come along so setup can make a
# track's data itself: they are imported inside functions, so the scan is told about them. The
# modules numba compiles go in as source files too (module_collection_mode 'pyz+py'): its cache
# needs the source to stamp against (launcher.py points the cache at a folder the user can write).
TRACK_TOOLS = ['ai_spline', 'track_files', 'setup_track', 'track_triangles', 'kn5_reader', 'build_road_mesh',
               'build_course', 'build_zones', 'line_tools', 'generate_track', 'fast_sim', 'drift_env',
               'sim_target', 'policy', 'sensor_view', 'pace_model', 'check_track', 'cars.acd_to_ini', 'cars.car_numbers']
COMPILED = ['fast_sim', 'drift_env', 'sim_target']

a = Analysis(
    [os.path.join(SPECPATH, 'launcher.py')],
    pathex=[SPECPATH, os.path.join(ROOT, 'drift_rl'), os.path.join(ROOT, 'tools')],
    hiddenimports=TRACK_TOOLS + ['gradientdrift_setup.trackdata', 'gradientdrift_setup.carfile', 'gradientdrift_setup.export'] + (['gradientdrift_setup.gui'] if GUI else []),
    excludes=([] if GUI else ['PySide6', 'shiboken6', 'gradientdrift_setup.gui'])
    # left out: what the track tools never use but the libraries pull in when it is installed
    + ['tkinter', 'torch', 'matplotlib', 'IPython', 'PIL', 'cryptography', 'lxml', 'sqlalchemy',
       'pyarrow', 'openpyxl', 'jinja2', 'pytest',
       # and the terminal line editors, which would bring GNU readline (GPL) along on Linux
       'readline', 'curses'],
    module_collection_mode={m: 'pyz+py' for m in COMPILED},
    noarchive=False,
)
# the app is English only: Qt's own translations are 7 MB of nothing
a.datas = [d for d in a.datas if 'translations' not in d[0].replace('\\', '/').split('/')]
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='gradient-drift-setup', console=True)
COLLECT(exe, a.binaries, a.datas, name='gradient-drift-setup')

# Licences: the program is this project's (AGPL-3.0-or-later) and carries other people's work, each
# under its own licence. Their texts go beside the program, in licenses/, with LICENSE, CREDITS.md
# and THIRD_PARTY_NOTICES.txt (what is inside, its version and licence, and where its text is).
import importlib.metadata as _md
import re as _re
import shutil as _shutil
import sys as _sys

_out = os.path.join(DISTPATH, 'gradient-drift-setup')
_lic = os.path.join(_out, 'licenses')
_shutil.rmtree(_lic, ignore_errors=True)
_rows = []


def _put(src, *dest):
    os.makedirs(os.path.join(_lic, *dest[:-1]), exist_ok=True)
    _shutil.copyfile(src, os.path.join(_lic, *dest))
    return 'licenses/' + '/'.join(dest)


DISTRIBUTIONS = ['numpy', 'scipy', 'pandas', 'numba', 'llvmlite', 'python-dateutil', 'pytz', 'tzdata', 'six',
                 'ruamel.yaml', 'ruamel.yaml.clib', 'pyinstaller'] + (['PySide6_Essentials', 'shiboken6'] if GUI else [])
for _name in DISTRIBUTIONS:
    try:
        _d = _md.distribution(_name)
    except _md.PackageNotFoundError:
        continue
    _texts = []
    for _f in _d.files or []:
        _p = str(_f).replace('\\', '/')
        if '.dist-info/' in _p and _re.search(r'(LICEN[CS]E|COPYING|NOTICE)', _p.rsplit('/', 1)[-1], _re.I):
            _texts.append(_put(str(_f.locate()), _name, *_p.split('.dist-info/', 1)[1].split('/')))
    if _name in ('PySide6_Essentials', 'shiboken6'):
        _texts += [_put(os.path.join(SPECPATH, 'licenses', 'LGPL-3.0.txt'), _name, 'LGPL-3.0.txt'),
                   _put(os.path.join(SPECPATH, 'licenses', 'GPL-3.0.txt'), _name, 'GPL-3.0.txt')]
    _licence = _d.metadata.get('License-Expression') or ((_d.metadata.get('License') or '').strip().splitlines() or [''])[0]
    if not _licence or len(_licence) > 60:          # some packages put the whole text there
        _licence = '; '.join(c.split('::')[-1].strip() for c in _d.metadata.get_all('Classifier') or []
                             if c.startswith('License ::')) or 'see the text'
    if _name == 'pyinstaller':
        _licence = 'GPL-2.0-or-later with the bootloader exception (only its bootloader is in the program)'
    _rows.append((_d.metadata['Name'], _d.version, _licence[:120], _texts))
    if not _texts:
        raise SystemExit(f'no licence text found for {_name}: the program must not ship without it')

_py = [p for p in (os.path.join(_sys.base_prefix, 'LICENSE.txt'),
                   os.path.join(_sys.base_prefix, 'lib', 'python%d.%d' % _sys.version_info[:2], 'LICENSE.txt'),
                   '/usr/lib/python%d.%d/LICENSE.txt' % _sys.version_info[:2]) if os.path.exists(p)]
if not _py:
    raise SystemExit("Python's LICENSE.txt not found: the program carries Python and must ship its licence")
_rows.append(('Python', '%d.%d.%d' % _sys.version_info[:3], 'PSF-2.0', [_put(_py[0], 'python', 'LICENSE.txt')]))
_rows.append(('KN5 Blender Tools (tools/kn5_reader.py)', '', 'GPL-3.0-or-later',
              [_put(os.path.join(ROOT, 'tools', 'kn5_reader_LICENSE.txt'), 'kn5_reader', 'LICENSE.txt'),
               _put(os.path.join(ROOT, 'tools', 'kn5_reader_MIT.txt'), 'kn5_reader', 'MIT.txt')]))
_rows.append(('AssettoServer (drift_rl/ai_spline.py is a port of its AI spline code)', 'v0.0.54', 'AGPL-3.0',
              ['LICENSE (the same licence as this program)']))
if _sys.platform.startswith('linux'):
    _sys_libs = sorted(f for f in os.listdir(os.path.join(_out, '_internal')) if '.so' in f and not f.startswith('libpython'))
    _rows.append(('System libraries of the build machine (Debian 11)', '', 'each its own: OpenSSL Apache-2.0, libffi '
                  'expat and ncurses MIT, zlib and bzip2 BSD-style, xz and SQLite public domain, libstdc++ / libgcc GPL with '
                  'the GCC Runtime Library Exception; sources at sources.debian.org',
                  ['(the files in _internal/: ' + ', '.join(f for f in _sys_libs if f.startswith('lib')) + ')']))
for _f in ('LICENSE', 'CREDITS.md'):
    _shutil.copyfile(os.path.join(ROOT, _f), os.path.join(_out, _f))
with open(os.path.join(_out, 'THIRD_PARTY_NOTICES.txt'), 'w', encoding='utf-8') as _fh:
    _fh.write('gradient-drift-setup is free software under the GNU Affero General Public License, version 3 or\n'
              'later (LICENSE); its source is the gradient-drift repository. It carries the following,\n'
              "each under its own licence, whose full text is in the file(s) named. Qt for Python's\n"
              'libraries are separate files in _internal/ and can be replaced with other builds of the\n'
              'same version (LGPL-3.0).\n\n')
    for _n, _v, _l, _t in _rows:
        _fh.write(f"{_n} {_v}\n    licence: {_l}\n" + ''.join(f"    text: {t}\n" for t in _t) + '\n')
