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
    (IS_VALID_TRACK=1), Grass for one that does not or whose FRICTION is under 0.9 x the main road's
    (ROAD's, or the most used track key's: verges a track lets count as track), Unknown for a key found
    in neither surfaces file.

<name>_collision_grip.csv.gz - one row per row of the collision file, in the same order: the
    surface's KEY and its grip, its FRICTION over the FRICTION of the road the layout has most of
    (so the main road is 1 and a slippery patch less; walls 1). The plugin's and the simulator's
    SurfaceGrip setting scales the tyres' grip by it. Plugins and trainers without that setting
    never read the file.

<name> is the track folder's name unless --name gives another. Each row: X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,
SurfaceType in world coordinates (to 0.1 mm), after one header line; gzip-compressed.

* The model files: those models_<layout>.ini (or models.ini) lists, each moved by its POSITION;
  without either file, <track>.kn5. File names are matched ignoring case.
* The game collides only with meshes named <number><KEY> (1ROAD, 2GRASS, 10WALL). KEY is a surface
  from the layout's data/surfaces.ini or the game's system/data/surfaces.ini; WALL is the game's
  wall. Meshes named AC_ (the game's helper objects) are left out of both files.

A layout whose model files are protected (every vertex moved by up to a few metres, which Content
Manager / Custom Shaders Patch undo when the game loads them) cannot be read: the ground comes out
as a crumpled sheet with layers metres apart. Such a layout is refused (SCRAMBLED_AGREEMENT).

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
# FRICTION when neither surfaces file gives one (the game's own file always does; these are only
# the fallback when it is not found): the same for all, so nothing differs from the road
DEFAULT_FRICTION = 1.0
HEADER = 'X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,SurfaceType\n'
GRIP_HEADER = 'Key,Grip\n'
# a track surface with less grip than this share of the main road's is the verge, not the course
OFF_ROAD_FRICTION_SHARE = 0.9
# A protected model file stores each vertex's normal as modelled but moves its position, so its
# triangles no longer face the way their vertices' normals say. |cos| between the two, median over
# the colliding road: 1.00 on a readable track (Spirit Peaks), about 0.6 on a protected one
# (Evergreen Speedway). Under this the layout is refused.
SCRAMBLED_AGREEMENT = 0.9


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
    return {k: valid for k, (valid, _) in surface_table(path).items()}


def surface_table(path):
    """{KEY: (counts as track, FRICTION or None)} from a surfaces.ini."""
    found = {}
    if path and os.path.exists(path):
        ini = read_ini(path)
        for section in ini.sections():
            if section.upper().startswith('SURFACE_') and ini.has_option(section, 'KEY'):
                key = ini.get(section, 'KEY').strip().upper()
                valid = ini.get(section, 'IS_VALID_TRACK', fallback='1').strip()
                try:
                    friction = float(ini.get(section, 'FRICTION', fallback='').strip())
                except ValueError:
                    friction = None
                found.setdefault(key, (valid not in ('0', ''), friction))
    return found


def triangle_area(t):
    ax, ay, az = t[3] - t[0], t[4] - t[1], t[5] - t[2]
    bx, by, bz = t[6] - t[0], t[7] - t[1], t[8] - t[2]
    cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    return 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5


def off_road_keys(friction, road_count):
    """The keys that count as track in the game but are not the road: their FRICTION is under
    OFF_ROAD_FRICTION_SHARE of the road's. The road is ROAD, the game's own road key, when the layout
    uses it, else the track key with the most triangles (a road is finely meshed, a verge or the
    ground beyond coarsely). Black Cat County's verges are OFF (0.70, gravel sound) and CUT (0.80),
    both IS_VALID_TRACK=1: as Road they made its course run on to the walls, 14 - 48 m wide round a
    road ~12 m wide."""
    if not road_count:
        return set()
    main = 'ROAD' if 'ROAD' in road_count else max(road_count, key=road_count.get)
    if main not in friction:
        return set()
    limit = OFF_ROAD_FRICTION_SHARE * friction[main]
    return {k for k in road_count if k in friction and friction[k] < limit}


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


def normal_agreement(mesh):
    """|cos| between each triangle's own normal and its vertices' stored normals (their mean), in
    the mesh's own coordinates: near 1 for every triangle of a readable mesh."""
    p, n, index = mesh.positions, mesh.normals, mesh.indices
    out = []
    for i in range(0, len(index) - 2, 3):
        a, b, c = p[index[i]], p[index[i + 1]], p[index[i + 2]]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        f = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        m = [n[index[i]][k] + n[index[i + 1]][k] + n[index[i + 2]][k] for k in range(3)]
        lf = (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5
        lm = (m[0] ** 2 + m[1] ** 2 + m[2] ** 2) ** 0.5
        if lf > 1e-9 and lm > 1e-9:
            out.append(abs(f[0] * m[0] + f[1] * m[1] + f[2] * m[2]) / (lf * lm))
    return out


def meshes(node, world, out, agreement=None):
    """Appends (mesh name, triangles in world coordinates) for node and its children; with
    `agreement` (a dict), also mesh name -> normal_agreement(mesh) for the colliding meshes."""
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
            if agreement is not None and node.name[:1].isdigit():
                agreement.setdefault(node.name, []).extend(normal_agreement(node.mesh))
    for child in node.children:
        meshes(child, own, out, agreement)


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
    friction = {k: DEFAULT_FRICTION for k in GAME_DEFAULTS}
    layout_data = find(track, f'{args.layout}/data') if args.layout else None
    track_surfaces = find(layout_data, 'surfaces.ini') if layout_data and os.path.isdir(layout_data) \
        else find(track, 'data/surfaces.ini')
    for source in (game_file, track_surfaces):
        for k, (valid, f) in surface_table(source).items():
            keys[k] = valid
            if f is not None:
                friction[k] = f
            friction.setdefault(k, DEFAULT_FRICTION)
    patterns = [(re.compile(r'^\d+' + re.escape(k)), k) for k in sorted(keys, key=len, reverse=True)]
    wall = re.compile(r'^\d+WALL')

    def collision_surface(mesh_name):
        """(Wall / Road / Grass / Unknown, surfaces.ini KEY or '') for a colliding mesh, (None, '')
        for scenery."""
        upper = mesh_name.upper()
        if not upper[:1].isdigit():
            return None, ''
        if wall.match(upper):
            return 'Wall', 'WALL'
        for pattern, key in patterns:
            if pattern.match(upper):
                return ('Road' if keys[key] else 'Grass'), key
        unplaced[re.sub(r'[._].*', '', upper)] += 1
        return 'Unknown', ''

    unplaced = collections.Counter()
    road_count = collections.Counter()
    road_agreement = []
    sensors, collision, collision_keys = [], [], []
    for file, (dx, dy, dz) in layout_files(track, args.layout):
        path = find(track, file)
        if path is None:
            print(f'  missing: {file} (listed by the layout)')
            continue
        found = []
        agreement = {}
        meshes(parse(path).root, [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)], found, agreement)
        before = len(sensors), len(collision)
        for mesh_name, tris in found:
            surface, key = collision_surface(mesh_name)
            moved = [(t[0] + dx, t[1] + dy, t[2] + dz, t[3] + dx, t[4] + dy, t[5] + dz,
                      t[6] + dx, t[7] + dy, t[8] + dz) for t in tris]
            if surface:
                collision.extend(t + (surface,) for t in moved)
                collision_keys.extend([key] * len(moved))
                if surface == 'Road':
                    road_count[key] += len(moved)
                    road_agreement.extend(agreement.get(mesh_name, []))
            if surface in (None, 'Wall'):
                kind = sensor_surface(mesh_name)
                sensors.extend(t + (kind,) for t in moved)
        print(f'  {file}: {len(sensors) - before[0]} sensor, {len(collision) - before[1]} collision triangles')

    if road_agreement:
        agree = sorted(road_agreement)[len(road_agreement) // 2]
        if agree < SCRAMBLED_AGREEMENT:
            raise SystemExit(f"{name}'s model files are protected (scrambled): its road's triangles do not face the "
                             f"way their stored normals say (median {agree:.2f}, a readable track 1.00), so its ground "
                             "and walls cannot be read from them. Drift bots cannot run on this layout.")

    off_road = off_road_keys(friction, road_count)
    if off_road:
        main = 'ROAD' if 'ROAD' in road_count else max(road_count, key=road_count.get)
        print(f"  counted as off the road (grass): {', '.join(f'{k} (FRICTION {friction[k]:g})' for k in sorted(off_road))}, "
              f"under {OFF_ROAD_FRICTION_SHARE:g} x the main road's ({main}, {friction[main]:g}) though the game counts them as track")
        collision = [r[:9] + ('Grass',) if k in off_road else r for r, k in zip(collision, collision_keys)]
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
    write_grip(os.path.join(args.output, f'{name}_collision_grip.csv.gz'), collision, collision_keys, friction)
    if unplaced:
        print('  numbered meshes with no known surface: ' + ', '.join(f'{k} ({v})' for k, v in unplaced.most_common(12)))
    if not any(r[9] != 'Wall' for r in collision):
        print('  warning: no ground found - the layout names no colliding <number><KEY> meshes')


def write_grip(out_path, collision, collision_keys, friction):
    """<name>_collision_grip.csv.gz: each collision row's KEY and FRICTION over the reference road's
    (the counted-as-track KEY with the most area). Walls and unknown keys are 1."""
    area = collections.Counter()
    for row, key in zip(collision, collision_keys):
        if row[9] == 'Road' and key:
            area[key] += triangle_area(row)
    reference = area.most_common(1)[0][0] if area else None
    ref_friction = friction.get(reference, DEFAULT_FRICTION) if reference else DEFAULT_FRICTION
    if ref_friction <= 0.0:
        ref_friction = DEFAULT_FRICTION

    def grip(row, key):
        if row[9] == 'Wall' or not key:
            return 1.0
        return friction.get(key, DEFAULT_FRICTION) / ref_friction
    with gzip.open(out_path, 'wt', compresslevel=6) as out:
        out.write(GRIP_HEADER)
        for row, key in zip(collision, collision_keys):
            out.write(f'{key or "-"},{grip(row, key):.4f}\n')
    total = sum(triangle_area(r) for r in collision if r[9] != 'Wall') or 1.0
    by_key = collections.Counter()
    for row, key in zip(collision, collision_keys):
        if row[9] != 'Wall':
            by_key[key or '-'] += triangle_area(row)
    print(f'Wrote surface grip to {out_path} (1 = {reference or "the road"}, FRICTION {ref_friction:g}): '
          + ', '.join(f'{k} {friction.get(k, DEFAULT_FRICTION) / ref_friction:.3f} ({a / total:.0%} of the ground)'
                      for k, a in by_key.most_common(8)))


if __name__ == '__main__':
    main()
