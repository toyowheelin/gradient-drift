"""
AI Car Monitor - an in-game Assetto Corsa app that logs what this client sees of every car on
the server (bots included, and optionally the player's own): position, the four tire contact
points, speed, pedals, gear, rpm, steering, velocity and whether the rear wheels spin. Each row
carries the car's index on this client, its driver name and its model.

Cars are not picked by number: the game numbers them on each client with the player as 0 and the
others shifted around the player's own slot, so a bot's index is not its server slot (a fixed
"car 16" logged nothing when the player joined in a later slot).

Install: copy this file to  assettocorsa/apps/python/AI Car Monitor/AI Car Monitor.py  and turn
the app on in the game. Press Record to start, Stop to finish. Each recording is written as it
goes to  Documents/Assetto Corsa/telemetry_exports/ai_car_monitor_<date>_<time>.csv.gz
(a plain .csv if the game's Python has no gzip). pandas reads either directly.

Every car is sampled at SAMPLE_HZ (20 a second), all cars at the same instant, whatever the frame
rate. Only the values the game really sends for a remote car are logged: the rest (local
velocities, tyre slip and angle, loads, suspension, radii, CG height) came back as zero or one
constant for every remote car, so they were dropped. A full server records ~1 MB a minute.
"""
import ac
import acsys
import csv
import os
import time

try:
    import gzip
except Exception:          # some game installs ship a cut-down Python
    gzip = None

PLAYER_CAR_ID = 0        # the game's index for this client's own car
LOG_PLAYER_CAR = True    # the player's car in the same columns, rows told apart by CarId
SAMPLE_HZ = 20.0         # samples a second for every car (the frame rate caps it)
FLUSH_SECONDS = 5.0      # how often the file is pushed to disk while recording
DECIMALS = 3             # mm for positions, 0.001 for pedals and steering

WHEELS = ["FL", "FR", "RL", "RR"]
SCALARS = ["SpeedKMH", "Gas", "Brake", "Steer"]
SLIP_WHEELS = ["RL", "RR"]   # SlipRatio: the only per-wheel value a remote car sends (on/off)


def columns():
    names = ["ClientTimeMs", "CarId", "DriverName", "CarModel", "X", "Y", "Z"]
    for wheel in WHEELS:
        names += ["Wheel{}_{}".format(wheel, axis) for axis in "XYZ"]
    names += SCALARS + ["RPM", "Gear"]
    names += ["Velocity_{}".format(axis) for axis in "XYZ"]
    names += ["SlipRatio_{}".format(wheel) for wheel in SLIP_WHEELS]
    return names


class Recorder:
    def __init__(self):
        self.file = None
        self.writer = None
        self.path = ""
        self.rows = 0
        self.slip_read = None   # "all" (one call, four values), "each", or None
        self.next_sample = 0.0
        self.next_flush = 0.0

    @property
    def active(self):
        return self.file is not None

    def start(self):
        folder = os.path.join(os.path.expanduser("~"), "Documents", "Assetto Corsa", "telemetry_exports")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        base = os.path.join(folder, "ai_car_monitor_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
        self.file = None
        if gzip is not None:
            try:
                self.file = gzip.open(base + ".gz", "wt", newline="")
                self.path = base + ".gz"
            except Exception:
                self.file = None
        if self.file is None:
            self.file = open(base, "w", newline="")
            self.path = base
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns())
        self.rows = 0
        self.slip_read = how_to_read("SlipRatio")
        self.next_sample = 0.0
        self.next_flush = time.time() + FLUSH_SECONDS
        ac.log("AI Car Monitor: writing {}".format(self.path))

    def stop(self):
        if self.file is not None:
            self.file.close()   # a .gz file is only whole once closed (a crash leaves it readable up to the last flush)
            ac.log("AI Car Monitor: {} rows written to {}".format(self.rows, self.path))
        self.file = None
        self.writer = None

    def sample(self, car_id, now_ms):
        row = [now_ms, car_id, ac.getDriverName(car_id), ac.getCarName(car_id)] \
            + rounded(ac.getCarState(car_id, acsys.CS.WorldPosition))
        for wheel in (acsys.WHEELS.FL, acsys.WHEELS.FR, acsys.WHEELS.RL, acsys.WHEELS.RR):
            row += rounded(ac.getCarState(car_id, acsys.CS.TyreContactPoint, wheel))
        for name in SCALARS:
            row.append(number(lambda: ac.getCarState(car_id, getattr(acsys.CS, name))))
        row.append(number(lambda: ac.getCarState(car_id, acsys.CS.RPM), 0))
        row.append(number(lambda: ac.getCarState(car_id, acsys.CS.Gear), 0))
        try:
            row += rounded(ac.getCarState(car_id, acsys.CS.Velocity))
        except Exception:
            row += ["", "", ""]
        if self.slip_read == "all":
            try:
                values = list(ac.getCarState(car_id, acsys.CS.SlipRatio))
                row += [round(float(values[2]), DECIMALS), round(float(values[3]), DECIMALS)]
            except Exception:
                row += ["", ""]
        elif self.slip_read == "each":
            row += [number(lambda w=w: ac.getCarState(car_id, acsys.CS.SlipRatio, w))
                    for w in (acsys.WHEELS.RL, acsys.WHEELS.RR)]
        else:
            row += ["", ""]
        self.writer.writerow(row)
        self.rows += 1


def how_to_read(name):
    """How the game hands out a per-wheel value: all four in one call, or one wheel per call."""
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


def rounded(values):
    return [round(float(v), DECIMALS) for v in values]


def number(read, decimals=DECIMALS):
    try:
        value = round(float(read()), decimals)
        return int(value) if decimals == 0 else value
    except Exception:
        return ""


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
    now = time.time()
    if now < recorder.next_sample:
        return
    # the next sample is due one step after this one was due, so a frame rate that does not divide
    # evenly still averages SAMPLE_HZ; after a stall (or the first frame) it counts from now
    step = 1.0 / SAMPLE_HZ
    behind = now - recorder.next_sample
    recorder.next_sample = recorder.next_sample + step if behind < step else now + step
    now_ms = int(now * 1000)
    for car_id in range(ac.getCarsCount()):
        if (car_id == PLAYER_CAR_ID and not LOG_PLAYER_CAR) or not ac.isConnected(car_id):
            continue
        try:
            recorder.sample(car_id, now_ms)
        except Exception:
            pass
    if now >= recorder.next_flush:
        recorder.file.flush()
        recorder.next_flush = now + FLUSH_SECONDS
        refresh()


def acShutdown():
    recorder.stop()
