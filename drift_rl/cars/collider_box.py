"""The size of a car's body from its collider.kn5 (the mesh the game collides the car with), for the
plugin's BodyWallContact: how far the body reaches forward, back and to each side of the centre of
gravity, in metres.

The mesh is in the 3D model's frame; car.ini GRAPHICS_OFFSET is where that model sits relative to the
centre of gravity (the physics origin), so a point of the body is mesh point + offset. +Z is forward,
X is across. The file is read by tools/kn5_reader.py (from KN5 Blender Tools, GPL-3.0-or-later; see
that file), as tools/track_triangles.py reads track models."""
import os
import sys

import numpy as np

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools')
if os.path.isdir(_TOOLS) and _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def kn5_points(path):
    """Every mesh vertex of a .kn5 file, in its root frame: an (n, 3) array. Raises ValueError (the
    reader's KN5Error) on a file it can't read."""
    from kn5_reader import parse
    out = []

    def walk(node, world):
        if node.kind == 1:
            world = np.array(node.matrix, dtype=float) @ world
        elif node.mesh is not None and node.mesh.positions:
            p = np.asarray(node.mesh.positions, dtype=float)
            out.append((np.c_[p, np.ones(len(p))] @ world)[:, :3])
        for child in node.children:
            walk(child, world)

    walk(parse(path).root, np.eye(4))
    if not out:
        raise ValueError('no mesh in the file')
    return np.vstack(out)


def graphics_offset(car_ini_text):
    """car.ini [BASIC] GRAPHICS_OFFSET as (x, y, z); (0, 0, 0) when it is not there."""
    section = None
    for line in car_ini_text.splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].strip().upper()
        elif section == 'BASIC' and '=' in line:
            key, value = line.split('=', 1)
            if key.strip().upper() == 'GRAPHICS_OFFSET':
                parts = [float(x) for x in value.split(',')[:3]]
                if len(parts) == 3:
                    return tuple(parts)
    return (0.0, 0.0, 0.0)


def body_box(collider_kn5_path, car_ini_text):
    """{'BODY_FRONT_M', 'BODY_REAR_M', 'BODY_HALF_WIDTH_M'}: the body's reach from the centre of
    gravity. Raises ValueError if the mesh can't be read or gives a body no car has."""
    p = kn5_points(collider_kn5_path) + np.array(graphics_offset(car_ini_text))
    front = float(p[:, 2].max())
    rear = float(-p[:, 2].min())
    half = float(np.abs(p[:, 0]).max())
    if not (0.5 < front < 4.0 and 0.5 < rear < 4.0 and 0.4 < half < 1.5):
        raise ValueError(f'body {front:.2f} m forward, {rear:.2f} m back, {half:.2f} m to the side is not a car')
    return {'BODY_FRONT_M': round(front, 3), 'BODY_REAR_M': round(rear, 3), 'BODY_HALF_WIDTH_M': round(half, 3)}
