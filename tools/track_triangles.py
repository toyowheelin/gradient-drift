"""
track_triangles.py - writes a track layout's geometry as two triangle files, read the way the game
reads the layout. The plugin and the trainer load the same two files.

    py track_triangles.py <track folder> <output folder> [--layout NAME] [--name NAME]
                          [--game-surfaces FILE]

<name>_triangles.csv.gz - what the sensors (ray fan, proximity pins) see: every mesh of the
    layout's model files except the colliding ground below (the colliding walls stay in). SurfaceType
    from the mesh name: Wall (walls, tire stacks), Road, Grass or Unknown.
<name>_collision.csv.gz - what the game collides with: the car stands on it and is pushed out of its
    walls. SurfaceType Wall for the game's walls, Road for a surface that counts as track
    (IS_VALID_TRACK=1), Grass for one that does not, Unknown for a key found in neither surfaces file.

<name> is the track folder's name unless --name gives another. Each row: X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,
SurfaceType in world coordinates (to 0.1 mm), after one header line; gzip-compressed.

* The model files: those models_<layout>.ini (or models.ini) lists, each moved by its POSITION;
  without either file, <track>.kn5. File names are matched ignoring case.
* The game collides only with meshes named <number><KEY> (1ROAD, 2GRASS, 10WALL). KEY is a surface
  from the layout's data/surfaces.ini or the game's system/data/surfaces.ini; WALL is the game's
  wall. Meshes named AC_ (the game's helper objects) are left out of both files.

The model files are read by kn5_reader.py (from KN5 Blender Tools, GPL-3.0-or-later; see that file).
"""
import argparse
import collections
import configparser
import gzip
import os
import re

from kn5_reader import parse

GAME_DEFAULTS = {'ROAD': True, 'GRASS': False, 'KERB': True, 'SAND': False}   # the game's system/data/surfaces.ini
HEADER = 'X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,SurfaceType\n'


def sensor_surface(name):
    upper = name.upper()
    if 'WALL' in upper or 'TIRE' in upper:
        return 'Wall'
    if 'GRASS' in upper:
        return 'Grass'
    if 'ROAD' in upper:
        return 'Road'
    return 'Unknown'


def read_ini(path):
    ini = configparser.ConfigParser(strict=False, interpolation=None, inline_comment_prefixes=(';', '//'))
    ini.optionxform = str.upper
    with open(path, encoding='latin-1') as f:
        ini.read_string(f.read())
    return ini


def surfaces(path):
    """{KEY: counts as track} from a surfaces.ini."""
    found = {}
    if path and os.path.exists(path):
        ini = read_ini(path)
        for section in ini.sections():
            if section.upper().startswith('SURFACE_') and ini.has_option(section, 'KEY'):
                key = ini.get(section, 'KEY').strip().upper()
                valid = ini.get(section, 'IS_VALID_TRACK', fallback='1').strip()
                found.setdefault(key, valid not in ('0', ''))
    return found


def find(folder, name):
    """The file in folder named name (which may hold sub-folders), ignoring case: tracks made on
    Windows often differ in case from their own model files, and on Linux that matters. None if
    it is not there."""
    path = folder
    for part in re.split(r'[\\/]', name):
        if not part:
            continue
        exact = os.path.join(path, part)
        if os.path.exists(exact):
            path = exact
            continue
        wanted = part.lower()
        found = [e for e in sorted(os.listdir(path)) if e.lower() == wanted] if os.path.isdir(path) else []
        if not found:
            return None
        path = os.path.join(path, found[0])
    return path


def layout_files(track, layout):
    """[(file name, (x, y, z) offset)] of the layout's model files."""
    for ini_name in ([f'models_{layout}.ini'] if layout else []) + ['models.ini']:
        path = find(track, ini_name)
        if path is None:
            continue
        ini = read_ini(path)
        files = []
        for section in ini.sections():
            if not ini.has_option(section, 'FILE'):
                continue
            offset = tuple((float(v) for v in ini.get(section, 'POSITION', fallback='0,0,0').split(',')[:3]))
            offset = offset + (0.0,) * (3 - len(offset))
            rotation = [float(v) for v in ini.get(section, 'ROTATION', fallback='0,0,0').split(',')]
            if any(rotation):
                print(f"  warning: {ini.get(section, 'FILE')} has ROTATION={rotation}, which is not applied")
            files.append((ini.get(section, 'FILE').strip(), offset))
        return files
    return [(os.path.basename(os.path.normpath(track)) + '.kn5', (0.0, 0.0, 0.0))]


def times(a, b):
    """4x4 product a.b (row-vector convention: a point p goes to p.a.b)."""
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def meshes(node, world, out):
    """Appends (mesh name, triangles in world coordinates) for node and its children."""
    if node.kind == 1:
        own = times([list(row) for row in node.matrix], world)
    else:
        own = world
        if node.kind == 2 and not node.name.startswith('AC_'):
            m = world
            p = [(m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0],
                  m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1],
                  m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2]) for x, y, z in node.mesh.positions]
            index = node.mesh.indices
            out.append((node.name, [(*p[index[i]], *p[index[i + 1]], *p[index[i + 2]])
                                    for i in range(0, len(index) - 2, 3)]))
    for child in node.children:
        meshes(child, own, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track')
    ap.add_argument('output')
    ap.add_argument('--layout', default=None)
    ap.add_argument('--name', default=None, help='file name prefix (default: the track folder name)')
    ap.add_argument('--game-surfaces', default=None, help="the game's system/data/surfaces.ini")
    args = ap.parse_args(argv)

    track = args.track
    name = args.name or os.path.basename(os.path.normpath(track))
    game_file = args.game_surfaces or find(os.path.join(track, '..', '..', '..'), 'system/data/surfaces.ini')
    keys = dict(GAME_DEFAULTS)
    keys.update(surfaces(game_file))
    layout_data = find(track, f'{args.layout}/data') if args.layout else None
    track_surfaces = find(layout_data, 'surfaces.ini') if layout_data and os.path.isdir(layout_data) \
        else find(track, 'data/surfaces.ini')
    keys.update(surfaces(track_surfaces))
    patterns = [(re.compile(r'^\d+' + re.escape(k)), k) for k in sorted(keys, key=len, reverse=True)]
    wall = re.compile(r'^\d+WALL')

    def collision_surface(mesh_name):
        """Wall / Road / Grass / Unknown for a colliding mesh, None for scenery."""
        upper = mesh_name.upper()
        if not upper[:1].isdigit():
            return None
        if wall.match(upper):
            return 'Wall'
        for pattern, key in patterns:
            if pattern.match(upper):
                return 'Road' if keys[key] else 'Grass'
        unplaced[re.sub(r'[._].*', '', upper)] += 1
        return 'Unknown'

    unplaced = collections.Counter()
    sensors, collision = [], []
    for file, (dx, dy, dz) in layout_files(track, args.layout):
        path = find(track, file)
        if path is None:
            print(f'  missing: {file} (listed by the layout)')
            continue
        found = []
        meshes(parse(path).root, [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)], found)
        before = len(sensors), len(collision)
        for mesh_name, tris in found:
            surface = collision_surface(mesh_name)
            moved = [(t[0] + dx, t[1] + dy, t[2] + dz, t[3] + dx, t[4] + dy, t[5] + dz,
                      t[6] + dx, t[7] + dy, t[8] + dz) for t in tris]
            if surface:
                collision.extend(t + (surface,) for t in moved)
            if surface in (None, 'Wall'):
                kind = sensor_surface(mesh_name)
                sensors.extend(t + (kind,) for t in moved)
        print(f'  {file}: {len(sensors) - before[0]} sensor, {len(collision) - before[1]} collision triangles')

    os.makedirs(args.output, exist_ok=True)
    for suffix, rows in (('triangles', sensors), ('collision', collision)):
        out_path = os.path.join(args.output, f'{name}_{suffix}.csv.gz')
        with gzip.open(out_path, 'wt', compresslevel=6) as out:
            out.write(HEADER)
            for row in rows:
                out.write(','.join(f'{v:.4f}' for v in row[:9]) + ',' + row[9] + '\n')
        by_type = collections.Counter(r[9] for r in rows)
        print(f'Wrote {len(rows)} triangles to {out_path}: '
              + ', '.join(f'{k} {v}' for k, v in sorted(by_type.items())))
    if unplaced:
        print('  numbered meshes with no known surface: ' + ', '.join(f'{k} ({v})' for k, v in unplaced.most_common(12)))
    if not any(r[9] != 'Wall' for r in collision):
        print('  warning: no ground found - the layout names no colliding <number><KEY> meshes')


if __name__ == '__main__':
    main()
