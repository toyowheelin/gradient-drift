"""The plugin's car files: an any-car policy drives each bot with its own car's numbers, which the
plugin reads from plugins/GradientDriftPlugin/cars/<car model>.json. Made here from the game
install's car (its data.acd, or its unpacked data folder) with the trainer's own tools
(drift_rl/cars), so the bot drives with exactly the numbers the policy was trained on."""
import json
import os
import shutil
import tempfile

from . import game  # puts drift_rl on the path
from .plan import PACE_MODEL

# the server's own track files the pace is worked out on when setup has no copy of the track's data
SERVER_SENSES = ('triangles', 'collision')
SERVER_EXTRAS = ('road_mesh.csv', 'course_edges.csv')


def car_file_text(game_folder, model):
    """(the car file's JSON text, None) or (None, the reason it cannot be made)."""
    from cars import acd_to_ini
    from cars.car_numbers import car_numbers, car_param_values
    folder = game.game_path(game_folder, 'content', 'cars', model)
    if not os.path.isdir(folder):
        return None, f"the game install has no content/cars/{model}"
    tmp = tempfile.mkdtemp(prefix='gradient-drift-car-')
    try:
        unpacked = os.path.join(tmp, model)       # car_numbers names the car after its folder
        acd = game.game_path(folder, 'data.acd')
        if os.path.exists(acd):
            os.makedirs(unpacked)
            # the key comes from the car's folder name as it is on disk, which can differ in capitals from MODEL=
            for name, content in acd_to_ini.decrypt_acd(acd, acd_to_ini.get_key(os.path.basename(folder))):
                with open(os.path.join(unpacked, os.path.basename(name)), 'wb') as f:
                    f.write(content)
        elif os.path.isdir(game.game_path(folder, 'data')):
            shutil.copytree(game.game_path(folder, 'data'), unpacked)
        else:
            return None, f"content/cars/{model} has neither data.acd nor a data folder"
        try:
            numbers = car_numbers(unpacked)
        except (KeyError, ValueError, OSError) as e:
            return None, f"could not read the car's data ({e.__class__.__name__}: {e})"
        if str(numbers.get('driven_wheels', '')).upper() != 'RWD':
            return None, f"{model} is not rear-wheel drive ({numbers.get('driven_wheels') or 'unknown'}); the policy drives rear-drive cars"
        values = car_param_values(numbers)
        values['_car'] = model
        values['_note'] = ("made by gradient-drift-setup from the game's car files; every bot drives it as a drift "
                           "car (welded diff, angle kit). PACE_SCALE 1 = not measured for this car")
        values['PACE_SCALE'] = 1.0
        return json.dumps(values, indent=1, sort_keys=True) + '\n', None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pace_source(policy, track_data, server_folder, track, made_here=False, server_line=None):
    """(how to work out the cars' PACE_SCALE on this track, None) or (None, why it cannot be): the
    server's own copy of the track's data (its AI line server_line and its track files), which is
    what its bots run on, else the track's data folder (drift_rl/pace_model.py reads its AI line).
    made_here: this run makes or copies the track's data before the cars' scales are worked out, so
    the track data folder's copy is the one the server will run."""
    tag = f"{PACE_MODEL} on {track}"
    if (not made_here and server_line and os.path.exists(server_line)
            and all(os.path.exists(os.path.join(server_folder, f'{track}_{kind}.csv.gz')) for kind in SERVER_SENSES)):
        return (policy, server_folder, track, server_line, tag), None
    if track_data and (made_here or os.path.exists(os.path.join(track_data, track, f'{track}_fast_lane.ai'))):
        return (policy, track_data, track, None, tag), None
    where = (f"{track_data} has no {track} folder with its AI line ({track}_fast_lane.ai)" if track_data
             else f"setup has no copy of {track}'s data")
    return None, (f"{where}, and the server has no copy of its own, so it cannot work out each car's pace there. "
                  "Setup makes that data itself when the server lacks it; if the server got it from elsewhere, "
                  "pick the folder holding it (a folder per track) under Track files")


def same_car_file(old_text, new_text, tag):
    """Whether the server's car file already is new_text with a PACE_SCALE measured as tag says."""
    try:
        old, new = json.loads(old_text), json.loads(new_text)
    except ValueError:
        return False
    if old.get('_pace') != tag:
        return False
    for d in (old, new):
        for k in ('PACE_SCALE', '_pace', '_note'):
            d.pop(k, None)
    return old == new


def measure_pace_scale(text, source):
    """The car file text with PACE_SCALE worked out on the track (drift_env.car_pace_scale: pace_model)."""
    import fast_sim as fs
    import drift_env as de
    _policy, track_data, track, server_line, tag = source
    if server_line:
        world = _server_world(track_data, track, server_line)
    else:
        world = fs.load_world(track_data, track=track)
    values = json.loads(text)
    values['PACE_SCALE'] = round(de.car_pace_scale([world], values), 4)
    values['_pace'] = tag
    values['_note'] = values['_note'].replace("PACE_SCALE 1 = not measured for this car",
                                              "PACE_SCALE: its speed on this track over the reference car's")
    return json.dumps(values, indent=1, sort_keys=True) + '\n'


def _server_world(server_folder, track, server_line):
    """The track as the server's bots see it (its AI line and track files), loaded from a scratch
    copy laid out as a track data folder."""
    import fast_sim as fs
    tmp = tempfile.mkdtemp(prefix='gradientdrift-pace-')
    try:
        folder = os.path.join(tmp, track)
        os.makedirs(folder)
        shutil.copyfile(server_line, os.path.join(folder, f'{track}_fast_lane.ai'))
        for name in [f'{kind}.csv.gz' for kind in SERVER_SENSES] + list(SERVER_EXTRAS):
            there = os.path.join(server_folder, f'{track}_{name}')
            if os.path.exists(there):
                shutil.copyfile(there, os.path.join(folder, f'{track}_{name}'))
        return fs.load_world(tmp, track=track)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
