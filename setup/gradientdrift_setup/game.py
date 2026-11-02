"""What the Assetto Corsa install has for a track layout and a car."""
import math
import os
import sys
from dataclasses import dataclass, field

# ai_spline.py lives in drift_rl (the packaged program carries it too: gradient-drift-setup.spec)
_DRIFT_RL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'drift_rl')
if _DRIFT_RL not in sys.path:
    sys.path.insert(0, _DRIFT_RL)

from track_files import game_path  # noqa: E402  (game files by name, ignoring capitals)


@dataclass
class TrackFacts:
    folder: str
    exists: bool = False
    model_files: list = field(default_factory=list)   # what the layout's models*.ini lists
    ai_line: str = ''                                 # path of its fast_lane.ai, '' if none
    ai_points: int = 0                                # 0 = missing, empty or unreadable
    ai_note: str = ''
    lap_m: float = 0.0                                # the AI line's length
    span_m: float = 0.0                               # the diagonal of the box round the AI line, seen from above
    recorded: str = ''                                # a recorded lap (fast_lane.ai.candidate), if any
    sides: str = ''                                   # the data folder with side_l.csv and side_r.csv, if any


def track_facts(game, track, layout):
    folder = game_path(game, 'content', 'tracks', track)
    t = TrackFacts(folder=folder, exists=os.path.isdir(folder))
    if not t.exists:
        return t
    for ini in ([f'models_{layout}.ini'] if layout else []) + ['models.ini']:
        path = game_path(folder, ini)
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                t.model_files = [l.split('=', 1)[1].strip() for l in f if l.strip().upper().startswith('FILE=')]
            break
    else:
        kn5 = game_path(folder, f'{track}.kn5')
        t.model_files = [os.path.basename(kn5)] if os.path.exists(kn5) else []
    ai = game_path(folder, layout, 'ai', 'fast_lane.ai') if layout else game_path(folder, 'ai', 'fast_lane.ai')
    if os.path.exists(ai):
        t.ai_line = ai
        t.ai_points, t.ai_note, t.lap_m, t.span_m = ai_line_points(ai)
    else:
        candidates = os.listdir(os.path.dirname(ai)) if os.path.isdir(os.path.dirname(ai)) else []
        if 'fast_lane.ai.candidate' in [c.lower() for c in candidates]:
            t.recorded = game_path(os.path.dirname(ai), 'fast_lane.ai.candidate')
            t.lap_m, t.span_m = ai_line_points(t.recorded)[2:]
        t.ai_note = 'no ai/fast_lane.ai' + (f" (the folder has {', '.join(candidates)})" if candidates else '')
    if t.ai_points < 2 and not t.recorded:
        # no line to start from: setup_track.py makes one down the middle of the road from the
        # road's edges, which most tracks carry for the game's own AI
        for data in ([game_path(folder, layout, 'data')] if layout else []) + [game_path(folder, 'data')]:
            if all(os.path.exists(game_path(data, f'side_{s}.csv')) for s in 'lr'):
                t.sides = data
                t.lap_m, t.span_m = side_files_size(data)
                break
    return t


def side_files_size(data):
    """(length in metres, span in metres) of the road's left edge in side_l.csv (X, Y, Z, progress)."""
    try:
        with open(game_path(data, 'side_l.csv'), encoding='utf-8-sig', errors='replace') as f:
            rows = sorted((float(r[3]), float(r[0]), float(r[2])) for r in
                          (line.split(',') for line in f) if len(r) >= 4)
    except (OSError, ValueError):
        return 0.0, 0.0
    if len(rows) < 2:
        return 0.0, 0.0
    lap = sum(math.dist(rows[i][1:], rows[(i + 1) % len(rows)][1:]) for i in range(len(rows)))
    xs, zs = [r[1] for r in rows], [r[2] for r in rows]
    return lap, math.hypot(max(xs) - min(xs), max(zs) - min(zs))


def ai_line_points(path):
    """(number of points, note, length in metres, span in metres). AssettoServer needs a readable line
    with points in it. The span is the diagonal of the box round the line, seen from above."""
    from ai_spline import load_fast_lane
    try:
        points = load_fast_lane(path)
    except Exception as e:                      # a truncated or foreign file
        size = os.path.getsize(path)
        return 0, (f'empty ({size} bytes)' if size < 1024 else f'unreadable ({e.__class__.__name__})'), 0.0, 0.0
    if len(points) < 2:
        return len(points), f'empty ({os.path.getsize(path)} bytes)', 0.0, 0.0
    pos = [p.position for p in points]
    lap = sum(math.dist(pos[i], pos[(i + 1) % len(pos)]) for i in range(len(pos)))
    xs, zs = [q[0] for q in pos], [q[2] for q in pos]
    span = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
    return len(points), '', lap, span


@dataclass
class CarFacts:
    model: str
    exists: bool = False
    data: str = ''            # 'data.acd', 'data folder' or ''
    skins: list = field(default_factory=list)


def car_facts(game, model):
    folder = game_path(game, 'content', 'cars', model)
    c = CarFacts(model=model, exists=os.path.isdir(folder))
    if not c.exists:
        return c
    if os.path.exists(game_path(folder, 'data.acd')):
        c.data = 'data.acd'
    elif os.path.isdir(game_path(folder, 'data')):
        c.data = 'data folder'
    skins = game_path(folder, 'skins')
    if os.path.isdir(skins):
        c.skins = sorted(d for d in os.listdir(skins) if os.path.isdir(os.path.join(skins, d)))
    return c
