"""
Reads the unpacked data of Assetto Corsa cars (the folders acd_to_ini.py writes) and prints,
for each car, the numbers the drift plugin's vehicle model uses.

Usage:
    python car_numbers.py <unpacked folder> [<unpacked folder> ...] [--json out.json]

Each folder must be named after the car (it is what acd_to_ini.py's output folder is called).

Where each number comes from:
    mass                 car.ini          [BASIC] TOTALMASS (kg, with driver)
    yaw inertia          car.ini          [BASIC] INERTIA = width, height, length of a uniform
                                          box: mass / 12 x (width^2 + length^2)
    wheelbase, front %   suspensions.ini  [BASIC] WHEELBASE, CG_LOCATION (share of the weight
                                          on the FRONT axle)
    steering lock        car.ini          [CONTROLS] STEER_LOCK / STEER_RATIO (degrees at the
                                          road wheel, centre to lock)
    tires                tyres.ini        the compound [COMPOUND_DEFAULT] INDEX selects:
                                          DY0 or DY_REF (peak sideways grip), DY1 (grip lost
                                          per kN on the tire, the older tyre model), and
                                          FRICTION_LIMIT_ANGLE (slip angle at that peak, deg)
    CG height            suspensions.ini  -[FRONT] BASEY (wheel centre below the CG) + the front
                                          tyre RADIUS
    roll stiffness       suspensions.ini  springs x track^2 / 2 + [ARB] x track^2, each axle;
    front share                           the front's share of the two
    driven wheels        drivetrain.ini   [TRACTION] TYPE
    engine               engine.ini       [HEADER] POWER_CURVE (power.lut): torque at the crank;
                                          [TURBO_n] MAX_BOOST, capped by WASTEGATE: the curve is
                                          multiplied by 1 + boost; [ENGINE_DATA] LIMITER
    gearing              drivetrain.ini   [GEARS] GEAR_n, FINAL; tyres.ini rear RADIUS
"""
import json
import math
import os
import sys

GRAVITY = 9.80665


def _any_case(path):
    """path, or the file beside it whose name differs only in capitals (car files are made on Windows,
    where CAR.INI is car.ini; on Linux it is not)."""
    folder, name = os.path.split(path)
    if os.path.exists(path) or not os.path.isdir(folder or '.'):
        return path
    return next((os.path.join(folder, e) for e in sorted(os.listdir(folder or '.')) if e.lower() == name.lower()), path)


def read_ini(path):
    """Section -> {key: value}. Comments (';' or '//') dropped; a repeated key keeps its
    first value, which is what Assetto Corsa itself does."""
    sections = {}
    current = None
    with open(_any_case(path), 'r', encoding='latin-1') as f:
        for line in f:
            line = line.split(';')[0].split('//')[0].strip()
            if not line:
                continue
            if line.startswith('['):
                name = line[1:line.index(']')].strip().upper() if ']' in line else line[1:].upper()
                current = sections.setdefault(name, {})
            elif '=' in line and current is not None:
                key, value = line.split('=', 1)
                current.setdefault(key.strip().upper(), value.strip())
    return sections


def number(section, key):
    value = section.get(key)
    if value is None:
        return None
    try:
        return float(value.split()[0].rstrip(','))
    except (ValueError, IndexError):
        return None


def torque_curve(path):
    """[(rpm, Nm)] from a .lut, in rpm order; [] if the file is missing."""
    points = []
    path = _any_case(path)
    if not os.path.exists(path):
        return points
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.split(';')[0].strip()
            if '|' not in line:
                continue
            try:
                rpm, torque = (float(x) for x in line.split('|')[:2])
            except ValueError:
                continue
            points.append((rpm, torque))
    return sorted(points)


def boost(engine):
    """Total turbo boost at full load: each [TURBO_n]'s MAX_BOOST, capped by its WASTEGATE when
    that is set. The game multiplies the torque curve by (1 + boost)."""
    total = 0.0
    for name in sorted(engine):
        if name.startswith('TURBO_'):
            value = number(engine[name], 'MAX_BOOST') or 0.0
            gate = number(engine[name], 'WASTEGATE')
            total += min(value, gate) if gate else value
    return total


# The drive-force curve: speeds it is given at, and the share of the engine's torque that reaches
# the wheels (the files do not say; a typical manual rear-drive drivetrain loses about 15%).
DRIVE_CURVE_STEP_MS = 2.5
DRIVE_CURVE_POINTS = 33                 # 0 - 80 m/s
DRIVETRAIN_EFFICIENCY = 0.85


def drive_force_curve(curve, gain, limiter, ratios, final, radius):
    """The most force the engine can put on the road at each speed (DRIVE_CURVE_STEP_MS apart,
    from 0): the best gear's torque at the rpm that speed gives, through the gearbox, the final
    drive and the tyre radius. Below the curve's lowest rpm the clutch slips and that torque is
    used; above the limiter a gear gives nothing. [] when the files lack any of it."""
    if not curve or not ratios or not final or not radius or any(r is None for r in ratios):
        return []
    rpms = [r for r, _ in curve]
    torques = [tq for _, tq in curve]

    def torque_at(rpm):
        if rpm <= rpms[0]:
            return torques[0]
        for i in range(1, len(rpms)):
            if rpm <= rpms[i]:
                w = (rpm - rpms[i - 1]) / max(rpms[i] - rpms[i - 1], 1e-9)
                return torques[i - 1] + w * (torques[i] - torques[i - 1])
        return torques[-1]
    top = limiter or rpms[-1]
    out = []
    for k in range(DRIVE_CURVE_POINTS):
        v = k * DRIVE_CURVE_STEP_MS
        best = 0.0
        for g in ratios:
            rpm = v / radius * g * final * 60.0 / (2.0 * math.pi)
            if rpm > top:
                continue
            best = max(best, max(0.0, torque_at(rpm)) * gain * g * final / radius * DRIVETRAIN_EFFICIENCY)
        out.append(best)
    return out


def car_numbers(folder):
    name = os.path.basename(os.path.normpath(folder))
    car = read_ini(os.path.join(folder, 'car.ini'))
    susp = read_ini(os.path.join(folder, 'suspensions.ini'))
    tyres = read_ini(os.path.join(folder, 'tyres.ini'))
    drive = read_ini(os.path.join(folder, 'drivetrain.ini'))
    engine = read_ini(os.path.join(folder, 'engine.ini'))

    mass = number(car['BASIC'], 'TOTALMASS')
    box = [float(x) for x in car['BASIC']['INERTIA'].split(',')[:3]]
    width, height, length = box
    wheelbase = number(susp['BASIC'], 'WHEELBASE')
    front_share = number(susp['BASIC'], 'CG_LOCATION')
    steer_lock = number(car['CONTROLS'], 'STEER_LOCK')
    steer_ratio = number(car['CONTROLS'], 'STEER_RATIO')

    index = int(number(tyres.get('COMPOUND_DEFAULT', {}), 'INDEX') or 0)
    suffix = '' if index == 0 else f'_{index}'
    front, rear = tyres['FRONT' + suffix], tyres['REAR' + suffix]

    tyre_version = int(number(tyres.get('HEADER', {}), 'VERSION') or 1)

    def grip(tyre):
        """The peak the model uses: DY0 for the older tyre model (versions below 5: grip of a tire
        with no load, DY1 off per kN); DY_REF for the newer one (the grip at the reference load
        FZ0, scaled by (load / FZ0)^(LS_EXPY - 1))."""
        if tyre_version >= 5 and number(tyre, 'DY_REF') is not None:
            return number(tyre, 'DY_REF')
        value = number(tyre, 'DY0')
        return value if value is not None else number(tyre, 'DY_REF')

    compounds = sorted(k for k in tyres if k == 'FRONT' or (k.startswith('FRONT_') and k[6:].isdigit()))
    curve_file = engine.get('HEADER', {}).get('POWER_CURVE', 'power.lut')
    curve = torque_curve(os.path.join(folder, curve_file))
    torque = max(curve, key=lambda p: p[1]) if curve else None
    turbo = [number(engine[k], 'MAX_BOOST') for k in sorted(engine) if k.startswith('TURBO_')]
    gain = 1.0 + boost(engine)
    limiter = number(engine.get('ENGINE_DATA', {}), 'LIMITER')
    in_range = [(r, t) for r, t in curve if limiter is None or r <= limiter]
    power_kw = max((t * gain * r * 2.0 * math.pi / 60.0 / 1000.0 for r, t in in_range), default=None)
    gears = drive.get('GEARS', {})
    count = int(number(gears, 'COUNT') or 0)
    ratios = [number(gears, f'GEAR_{i}') for i in range(1, count + 1)]
    final = number(gears, 'FINAL')
    rear_radius = number(rear, 'RADIUS')

    weight = mass * GRAVITY
    front_track, rear_track = number(susp['FRONT'], 'TRACK'), number(susp['REAR'], 'TRACK')
    arb = susp.get('ARB', {})

    def roll_stiffness(axle, track):
        spring = number(susp[axle], 'SPRING_RATE') or 0.0
        return (spring * track * track / 2.0 + (number(arb, axle) or 0.0) * track * track) if track else 0.0
    roll_f, roll_r = roll_stiffness('FRONT', front_track), roll_stiffness('REAR', rear_track)
    base_y, front_radius = number(susp['FRONT'], 'BASEY'), number(front, 'RADIUS')
    return {
        'car': name,
        'screen_name': car.get('INFO', {}).get('SCREEN_NAME', ''),
        'mass_kg': mass,
        'inertia_box_m': box,
        'yaw_inertia_kgm2': mass / 12.0 * (width ** 2 + length ** 2),
        'wheelbase_m': wheelbase,
        'front_weight_share': front_share,
        'cg_to_front_axle_m': (1.0 - front_share) * wheelbase,
        'cg_to_rear_axle_m': front_share * wheelbase,
        'front_static_load_n': front_share * weight,
        'rear_static_load_n': (1.0 - front_share) * weight,
        'front_track_m': front_track,
        'rear_track_m': rear_track,
        'cg_height_m': (front_radius - base_y) if base_y is not None and front_radius else None,
        'roll_stiffness_front_share': roll_f / (roll_f + roll_r) if roll_f + roll_r > 0 else None,
        'steer_lock_deg': steer_lock,
        'steer_ratio': steer_ratio,
        'max_steer_deg': steer_lock / abs(steer_ratio),
        'tyre_compound_index': index,
        'tyre_compound_count': len(compounds),
        'tyre_name': front.get('NAME', ''),
        'front_peak_friction': grip(front),
        'rear_peak_friction': grip(rear),
        'front_load_sensitivity_per_kn': number(front, 'DY1'),
        'rear_load_sensitivity_per_kn': number(rear, 'DY1'),
        'front_peak_slip_deg': number(front, 'FRICTION_LIMIT_ANGLE'),
        'rear_peak_slip_deg': number(rear, 'FRICTION_LIMIT_ANGLE'),
        'tyre_version': tyre_version,
        'front_ls_expy': number(front, 'LS_EXPY') if tyre_version >= 5 else None,
        'rear_ls_expy': number(rear, 'LS_EXPY') if tyre_version >= 5 else None,
        'front_fz0_n': number(front, 'FZ0'),
        'rear_fz0_n': number(rear, 'FZ0'),
        'front_tyre_width_m': number(front, 'WIDTH'),
        'rear_tyre_width_m': number(rear, 'WIDTH'),
        'driven_wheels': drive.get('TRACTION', {}).get('TYPE', ''),
        'peak_torque_nm': torque[1] if torque else None,
        'peak_torque_rpm': torque[0] if torque else None,
        'turbo_max_boost': [b for b in turbo if b is not None],
        'peak_torque_boosted_nm': torque[1] * gain if torque else None,
        'peak_power_kw': power_kw,
        'power_per_tonne_kw': power_kw / (mass / 1000.0) if power_kw else None,
        'rev_limit_rpm': limiter,
        'gear_ratios': ratios,
        'final_drive': final,
        'rear_tyre_radius_m': rear_radius,
        'drive_force_curve_n': drive_force_curve(curve, gain, limiter, ratios, final, rear_radius),
        'peak_wheel_force_first_gear_n': (torque[1] * gain * ratios[0] * final / rear_radius
                                          if torque and ratios and ratios[0] and final and rear_radius else None),
    }


# What car_param_values() reads from a cars/car_numbers.py entry, by the simulator's names.
_CAR_NUMBER_PARAMS = {
    'YAW_INERTIA_KGM2': 'yaw_inertia_kgm2', 'DIST_CG_FRONT_M': 'cg_to_front_axle_m',
    'DIST_CG_REAR_M': 'cg_to_rear_axle_m', 'FRONT_STATIC_LOAD_N': 'front_static_load_n',
    'REAR_STATIC_LOAD_N': 'rear_static_load_n', 'LAT_FRONT_PEAK_FRICTION': 'front_peak_friction',
    'LAT_REAR_PEAK_FRICTION': 'rear_peak_friction', 'CG_HEIGHT_M': 'cg_height_m',
    'FRONT_TRACK_M': 'front_track_m', 'REAR_TRACK_M': 'rear_track_m',
    'ROLL_STIFFNESS_FRONT_SHARE': 'roll_stiffness_front_share',
}


# Every bot drives as a drift car whatever its files say (the owner, 2026-10-12): a welded
# rear diff (the model's rear axle is one solid axle already) and an angle kit giving at least
# this much lock at the road wheels. Weight, power, gearing, tyres and geometry stay the car's own.
DRIFT_STEER_LOCK_DEG = 50.0


def car_param_values(car, steer_lock_deg=None, pace_scale=1.0):
    """The simulator parameters (fast_sim names) of one car from its cars/car_numbers.py entry:
    geometry, loads, tyres (the older model's DY1 or the newer one's LS_EXPY and FZ0), steering
    lock (the car's own, but at least DRIFT_STEER_LOCK_DEG; or steer_lock_deg), and its engine
    (EnableEngine, the drive-force curve)."""
    v = {name: float(car[key]) for name, key in _CAR_NUMBER_PARAMS.items() if car.get(key) is not None}
    v['FRONT_PEAK_SLIP_RAD'] = math.radians(car['front_peak_slip_deg'])
    v['REAR_PEAK_SLIP_RAD'] = math.radians(car['rear_peak_slip_deg'])
    v['MAX_STEER_RAD'] = math.radians(steer_lock_deg if steer_lock_deg
                                       else max(DRIFT_STEER_LOCK_DEG, car['max_steer_deg']))
    newer = (car.get('tyre_version') or 1) >= 5 and car.get('front_ls_expy') and car.get('front_fz0_n')
    for axle in ('FRONT', 'REAR'):
        a = axle.lower()
        if newer:
            v[f'{axle}_LS_EXPY'] = float(car[f'{a}_ls_expy'])
            v[f'{axle}_FZ0_N'] = float(car[f'{a}_fz0_n'])
            v[f'{axle}_LOAD_SENSITIVITY_PER_KN'] = 0.0
        else:
            v[f'{axle}_LS_EXPY'] = 0.0
            v[f'{axle}_LOAD_SENSITIVITY_PER_KN'] = float(car.get(f'{a}_load_sensitivity_per_kn') or 0.0)
    curve = car.get('drive_force_curve_n') or []
    v['EXT_ENGINE'] = 1.0 if len(curve) == DRIVE_CURVE_POINTS else 0.0
    for k, force in enumerate(curve[:DRIVE_CURVE_POINTS]):
        v[f'ENGINE_FORCE_{k:02d}'] = float(force)
    v['PACE_SCALE'] = float(pace_scale)
    return v



def main():
    args = sys.argv[1:]
    out = None
    if '--json' in args:
        i = args.index('--json')
        out = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(1)

    cars = [car_numbers(folder) for folder in args]

    rows = [
        ('mass, kg', lambda c: f"{c['mass_kg']:.0f}"),
        ('wheelbase, m', lambda c: f"{c['wheelbase_m']:.3f}"),
        ('weight on front axle', lambda c: f"{100 * c['front_weight_share']:.1f}%"),
        ('CG to front axle, m', lambda c: f"{c['cg_to_front_axle_m']:.3f}"),
        ('CG to rear axle, m', lambda c: f"{c['cg_to_rear_axle_m']:.3f}"),
        ('front static load, N', lambda c: f"{c['front_static_load_n']:.1f}"),
        ('rear static load, N', lambda c: f"{c['rear_static_load_n']:.1f}"),
        ('yaw inertia, kg m2', lambda c: f"{c['yaw_inertia_kgm2']:.1f}"),
        ('max steer at the wheel, deg', lambda c: f"{c['max_steer_deg']:.2f}"),
        ('tire compound', lambda c: f"{c['tyre_compound_index']} of {c['tyre_compound_count']}"),
        ('front peak grip', lambda c: f"{c['front_peak_friction']:.4f}"),
        ('rear peak grip', lambda c: f"{c['rear_peak_friction']:.4f}"),
        ('front peak slip, deg', lambda c: f"{c['front_peak_slip_deg']:.1f}"),
        ('rear peak slip, deg', lambda c: f"{c['rear_peak_slip_deg']:.1f}"),
        ('driven wheels', lambda c: c['driven_wheels']),
        ('peak torque, Nm (no boost)', lambda c: '-' if c['peak_torque_nm'] is None else f"{c['peak_torque_nm']:.0f}"),
        ('turbo max boost', lambda c: '+'.join(f'{b:g}' for b in c['turbo_max_boost']) or '-'),
        ('peak power with boost, kW', lambda c: '-' if c['peak_power_kw'] is None else f"{c['peak_power_kw']:.0f}"),
        ('power per tonne, kW', lambda c: '-' if c['power_per_tonne_kw'] is None else f"{c['power_per_tonne_kw']:.0f}"),
        ('rev limit, rpm', lambda c: '-' if c['rev_limit_rpm'] is None else f"{c['rev_limit_rpm']:.0f}"),
        ('gears, final drive', lambda c: ', '.join(f'{g:g}' for g in c['gear_ratios'] if g) + f"; {c['final_drive']}"),
    ]
    for c in cars:
        print(f"\n{c['car']}  ({c['screen_name']}; tires: {c['tyre_name']})")
        for label, fmt in rows:
            print(f"    {label:30s}{fmt(c)}")

    if out:
        with open(out, 'w') as f:
            json.dump(cars, f, indent=2)
        print(f"\nwritten: {out}")


if __name__ == '__main__':
    main()
