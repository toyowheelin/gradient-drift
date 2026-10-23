"""
AI Car Monitor - an in-game Assetto Corsa app that logs what this client sees of every car on
the server (bots included, and optionally the player's own): position, the four tire contact
points, and whatever the game's Python API will give for the car's motion and tires. Each row
carries the car's index on this client, its driver name and its model.

Cars are not picked by number: the game numbers them on each client with the player as 0 and the
others shifted around the player's own slot, so a bot's index is not its server slot (a fixed
"car 16" logged nothing when the player joined in a later slot).

Install: copy this file to  assettocorsa/apps/python/AI Car Monitor/AI Car Monitor.py  and turn
the app on in the game. Press Record to start, Stop to finish. Each recording is written as it
goes to  Documents/Assetto Corsa/telemetry_exports/ai_car_monitor_<date>_<time>.csv.

The game returns zero for many values of a remote car, so a zero here does not mean a client
never got it.
"""
import ac
import acsys
import csv
import os
import time

PLAYER_CAR_ID = 0        # the game's index for this client's own car
LOG_PLAYER_CAR = True    # the player's car in the same columns, rows told apart by CarId
DECIMALS = 4

SCALARS = ["SpeedKMH", "RPM", "Gas", "Brake", "Gear", "Steer", "DriveTrainSpeed", "CGHeight"]
VECTORS = ["Velocity", "LocalVelocity", "LocalAngularVelocity"]
PER_WHEEL = ["WheelAngularSpeed", "TyreSlip", "NdSlip", "SlipAngle", "SlipRatio", "Load",
             "SuspensionTravel", "TyreDirtyLevel", "TyreRadius", "TyreLoadedRadius"]
WHEELS = ["FL", "FR", "RL", "RR"]


def columns():
    names = ["ClientTimeMs", "CarId", "DriverName", "CarModel", "X", "Y", "Z"]
    for wheel in WHEELS:
        names += ["Wheel{}_{}".format(wheel, axis) for axis in "XYZ"]
    names += SCALARS
    for name in VECTORS:
        names += ["{}_{}".format(name, axis) for axis in "XYZ"]
    for name in PER_WHEEL:
        names += ["{}_{}".format(name, wheel) for wheel in WHEELS]
    return names


class Recorder:
    def __init__(self):
        self.file = None
        self.writer = None
        self.rows = 0
        self.wheel_reads = {}   # per-wheel value -> "all" (one call, four values), "each", or None

    @property
    def active(self):
        return self.file is not None

    def start(self):
        folder = os.path.join(os.path.expanduser("~"), "Documents", "Assetto Corsa", "telemetry_exports")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        path = os.path.join(folder, "ai_car_monitor_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns())
        self.rows = 0
        self.wheel_reads = {name: self.how_to_read(name) for name in PER_WHEEL}
        ac.log("AI Car Monitor: writing {}".format(path))

    def stop(self):
        if self.file is not None:
            self.file.close()
            ac.log("AI Car Monitor: {} rows written".format(self.rows))
        self.file = None
        self.writer = None

    def how_to_read(self, name):
        key = getattr(acsys.CS, name, None)
        if key is None:
            return None
        try:
            if len(list(ac.getCarState(PLAYER_CAR_ID, key))) == 4:
                return "all"
        except Exception:
            pass
        try:
            float(ac.getCarState(PLAYER_CAR_ID, key, acsys.WHEELS.FL))
            return "each"
        except Exception:
            return None

    def sample(self, car_id, now_ms):
        row = [now_ms, car_id, ac.getDriverName(car_id), ac.getCarName(car_id)] \
            + list(ac.getCarState(car_id, acsys.CS.WorldPosition))
        for wheel in (acsys.WHEELS.FL, acsys.WHEELS.FR, acsys.WHEELS.RL, acsys.WHEELS.RR):
            row += list(ac.getCarState(car_id, acsys.CS.TyreContactPoint, wheel))
        for name in SCALARS:
            row.append(number(lambda: ac.getCarState(car_id, getattr(acsys.CS, name))))
        for name in VECTORS:
            row += numbers(lambda: ac.getCarState(car_id, getattr(acsys.CS, name)), 3)
        for name in PER_WHEEL:
            how = self.wheel_reads.get(name)
            if how == "all":
                row += numbers(lambda: ac.getCarState(car_id, getattr(acsys.CS, name)), 4)
            elif how == "each":
                row += [number(lambda w=w: ac.getCarState(car_id, getattr(acsys.CS, name), w))
                        for w in (acsys.WHEELS.FL, acsys.WHEELS.FR, acsys.WHEELS.RL, acsys.WHEELS.RR)]
            else:
                row += [""] * 4
        self.writer.writerow(row)
        self.rows += 1


def number(read):
    try:
        return round(float(read()), DECIMALS)
    except Exception:
        return ""


def numbers(read, count):
    try:
        values = [round(float(v), DECIMALS) for v in read()]
    except Exception:
        return [""] * count
    return values if len(values) == count else [""] * count


recorder = Recorder()
ui = {}


def refresh():
    ac.setText(ui["toggle"], "Stop" if recorder.active else "Record")
    ac.setText(ui["state"], "{} rows".format(recorder.rows) if recorder.active else "idle")


def on_toggle(*args):
    if recorder.active:
        recorder.stop()
    else:
        recorder.start()
    refresh()


def acMain(ac_version):
    window = ac.newApp("AI Car Monitor")
    ac.setSize(window, 150, 70)
    ui["state"] = ac.addLabel(window, "")
    ac.setPosition(ui["state"], 10, 30)
    ui["toggle"] = ac.addButton(window, "")
    ac.setPosition(ui["toggle"], 80, 28)
    ac.setSize(ui["toggle"], 60, 24)
    ac.addOnClickedListener(ui["toggle"], on_toggle)
    refresh()
    return "AI Car Monitor"


def acUpdate(delta_t):
    if not recorder.active:
        return
    now_ms = int(time.time() * 1000)
    for car_id in range(ac.getCarsCount()):
        if (car_id == PLAYER_CAR_ID and not LOG_PLAYER_CAR) or not ac.isConnected(car_id):
            continue
        try:
            recorder.sample(car_id, now_ms)
        except Exception:
            pass
    if recorder.rows % 200 == 0:
        recorder.file.flush()
        refresh()


def acShutdown():
    recorder.stop()
