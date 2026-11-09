"""The car body read from a collider.kn5 (drift_rl/cars/collider_box.py) for the plugin's
BodyWallContact, on a made-up file in the game's layout."""
import struct
import sys
import os

import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'drift_rl'))
from cars import collider_box  # noqa: E402


def _text(s):
    b = s.encode('latin-1')
    return struct.pack('<i', len(b)) + b


def make_kn5(x_half, z_min, z_max, shift_z=0.0):
    """A collider.kn5: one material, a transform node (moving the mesh shift_z forward) holding a
    box mesh from -x_half to x_half across and z_min to z_max along."""
    out = b'sc6969' + struct.pack('<i', 5)
    out += struct.pack('<i', 0)                                    # textures
    out += struct.pack('<i', 1) + _text('collider') + _text('GL') + bytes([0, 0]) + struct.pack('<i', 0)
    out += struct.pack('<i', 0) + struct.pack('<i', 0)             # properties, samplers
    matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, shift_z, 1]
    out += struct.pack('<i', 1) + _text('root') + struct.pack('<i', 1) + bytes([1]) + struct.pack('<16f', *matrix)
    verts = [(x, y, z) for x in (-x_half, x_half) for y in (0.1, 1.2) for z in (z_min, z_max)]
    out += struct.pack('<i', 2) + _text('collision box') + struct.pack('<i', 0) + bytes([1, 1, 1, 0])
    out += struct.pack('<i', len(verts))
    for v in verts:
        out += struct.pack('<11f', *v, 0, 1, 0, 0, 0, 1, 0, 0)
    idx = [0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6]
    out += struct.pack('<i', len(idx)) + struct.pack(f'<{len(idx)}H', *idx)
    out += struct.pack('<ii', 0, 0) + struct.pack('<2f', 0, 1000) + struct.pack('<4f', 0, 0, 0, 3) + bytes([1])
    return out


CAR_INI = "[INFO]\nSCREEN_NAME=x\n[BASIC]\nGRAPHICS_OFFSET=0,-0.50,-0.16 ; model correction\nTOTALMASS=1137\n"


def _file(tmp_path, data):
    path = tmp_path / 'collider.kn5'
    path.write_bytes(data)
    return str(path)


def test_body_from_collider_and_graphics_offset(tmp_path):
    body = collider_box.body_box(_file(tmp_path, make_kn5(0.82, -1.96, 1.97)), CAR_INI)
    assert body == pytest.approx({'BODY_FRONT_M': 1.81, 'BODY_REAR_M': 2.12, 'BODY_HALF_WIDTH_M': 0.82}, abs=1e-3)


def test_transform_nodes_are_applied(tmp_path):
    body = collider_box.body_box(_file(tmp_path, make_kn5(0.8, -2.0, 2.0, shift_z=0.3)), "[BASIC]\nTOTALMASS=1\n")
    assert body['BODY_FRONT_M'] == pytest.approx(2.3) and body['BODY_REAR_M'] == pytest.approx(1.7)


def test_refuses_what_is_not_a_car(tmp_path):
    for data in (b'not a kn5 at all', make_kn5(5.0, -1.0, 1.0), make_kn5(0.8, -2.0, 2.0)[:120]):
        with pytest.raises(ValueError):
            collider_box.body_box(_file(tmp_path, data), CAR_INI)
