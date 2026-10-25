# drift_rl: the trainer

Trains the small neural-network policy the plugin drives with. Training happens **offline**, in a
fast copy of the plugin's own vehicle model. The result is one frozen file, `policy.json`, which
the plugin loads; nothing learns on the live server.

> [!NOTE]
> Training a policy of your own is optional: the plugin ships with one
> (`GradientDriftPlugin/GradientDriftPlugin.policy.json`).

What it can train:

- **Several tracks and generated courses at once**, so one policy drives tracks it has never seen.
- **A pool of cars.** Every car drives as a drift car (welded diff, an angle kit of at least 50°),
  with its weight, geometry, tyres and engine from its own files (`cars/car_numbers.py`). The policy
  reads those numbers, so one policy drives any rear-drive car.
- **Exactly the physics the plugin will run.** Parts of the vehicle model that came later (the
  tyres' load sensitivity, the steering rate, the rear friction circle) are `--plugin-setting`s a
  run is trained with and writes into its policy file. The plugin turns on exactly those, so a
  policy always drives in the car it learned in.

**Contents:** [Install](#install) · [1. Track data](#1-track-data) · [2. Train](#2-train) ·
[3. See what it does](#3-see-what-it-does) · [4. Use it on a server](#4-use-it-on-a-server) ·
[5. Check a server run](#5-check-a-server-run) ·
[Polishing the shipped policy for one track](#polishing-the-shipped-policy-for-one-track) · [Cars](#cars) ·
[What the policy does](#what-the-policy-does) · [Files](#files) ·
[How far to trust the simulator](#how-far-to-trust-the-simulator)

## Install

Python 3.12 or newer (3.12 is what the GitHub builds use; 3.13 and 3.14 run it too).

```
pip install -r requirements.txt torch
python check_plugin_policy.py --self-test --policy ../GradientDriftPlugin/GradientDriftPlugin.policy.json
```

The self-test should end `ALL CHECKS PASSED`.

- **NVIDIA GPU:** PyTorch needs its CUDA build for the network. On Windows a plain
  `pip install torch` is CPU-only; take the command from https://pytorch.org/get-started/locally/.
- **The simulator runs on the CPU cores** (`--sim cpu`). A GPU version of it exists (`gpu_env.py`;
  it needs `pip install "numba-cuda[cu12]"` and `python check_gpu.py`), but on a mid-range card it
  is slower than 16 CPU threads.
- After `fast_sim.py` or `drift_env.py` change, the first start recompiles (about a minute).
- Ctrl-C stops a run cleanly; its checkpoint and policy are saved first.

## 1. Track data

None ships with this project (see the top-level README). The trainer reads it from
`tracks/<track>/` in this folder, and you make it from the game installed on your own machine,
all of it in one go:

```
python setup_track.py <game>/content/tracks/<track>
```

It ends by checking what the data asks of the bots (`check_track.py`; `--policy` names the policy
that will drive, for its sensors' height): speed targets the car cannot drive, the AI line running
into a wall, and walls beside the course too low for the sensors to feel. On a problem it stops and
says where; `--skip-check` goes on anyway. `python check_track.py --track <track>` runs the check on
its own.

`<track>` is the track's folder name under `content/tracks/`. `--track NAME` picks the track for
any tool (repeat it to train on several); `--tracks FOLDER` points any tool at another folder of
tracks.

<details>
<summary>The files it makes, one by one</summary>

| file | where it comes from |
|---|---|
| `<track>_fast_lane.ai` | the track's AI line: `content/tracks/<track>/ai/fast_lane.ai` in the game, renamed |
| `<track>_collision.csv.gz`, `<track>_visual.csv.gz` | the track's model files, read the game's way: what the game collides with (its ground and walls) and what the model shows. `python ../tools/track_triangles.py <game>/content/tracks/<track> tracks/<track>` writes the second as `<track>_triangles.csv.gz`; rename it |
| `<track>_triangles.csv.gz` | what the bot's sensors see: the walls the game collides with plus the course's markers, `python sensor_view.py --track <track>` (after `build_course.py`). Things the model shows that the car cannot hit are left out. |
| `<track>_road_mesh.csv` | the road-mesh points the plugin reads, from the AssettoServer folder |
| `<track>_course_edges.csv`, `<track>_course_markers.csv` | `python build_course.py` (the course's edges, from its markers where it has them) |
| `<track>_course_zones.csv` | `python build_zones.py` (outside zones and the long-corner stretches) |
| `<track>_pace_reference.csv` | made by the trainer the first time it trains the track with paces (or by the setup program, the same way, for a track the policy never trained on): how fast a reference policy drives each point flat out, which a pace of 1.0 asks for. The plugin needs it in the server folder for a policy that reads its pace error. |

The files keep the track's name in theirs inside the track's folder, because they are copied to
the server folder as they are. The server folder needs `<track>_triangles.csv.gz`,
`<track>_collision.csv.gz`, `<track>_road_mesh.csv`, `<track>_course_edges.csv` and, for a policy
with paces, `<track>_pace_reference.csv`; and the AI line, as
`content/tracks/<track>/ai/fast_lane.ai`.

</details>

## 2. Train

Start from the shipped policy, or continue a run of your own:

```
python train_ppo_gpu.py --sim cpu --init-policy ../GradientDriftPlugin/GradientDriftPlugin.policy.json --track <track> --steps 20000000 --name mine
python train_ppo_gpu.py --sim cpu --resume runs/mine/checkpoint.pt --steps 40000000 --name mine2
```

| option | what it does |
|---|---|
| `--track A --track B --generated 100` | several tracks plus generated courses at once |
| `--pace 0.8 1.0 --pace-reference P` | paces (a share of each course's flat-out speed, as driven by policy P), for spacing bots |
| `--car-pool cars/car_pool.json` | train on many cars (see [Cars](#cars)); `--car-exclude CAR` holds one out |
| `--skill 0 1` | train the skill input (the plugin's `BotSkill`): 0 = careful, 1 = best |
| `--chase 0.5` | (tandem branch, CPU simulator only) half the episodes chase a ghost leader, a recorded run replayed ahead of the car (`record_ghosts.py` makes them); `--chase-gap LOW HIGH` the asked gaps, `--chase-gap-anneal-ticks N` starting them in the top half of that range and widening it to LOW over N ticks. While chasing, the pace input comes from the Intelligent Driver Model (with the lead's plan for half the episodes, a bot lead), and the policy sees the lead's path 0.5, 1 and 2 s ahead. `--residual UNITS` keeps the starting policy frozen and trains a second network added to its action only while there is a lead (residual policy learning), so solo driving stays exactly as it was. Gaps are between the cars' outlines; the gap paid for opens 2.5 m through the lead's direction changes (breathing, as the TRI / Stanford autonomous tandem cars drop back on transitions) and the line is paid from 0.5 m inside the lead's in turns. `--chase-scenarios` starts some chase episodes in hard situations (catch-ups from 30 - 70 m, alongside into a transition, a lead that bogs down), as GT Sophy trained; `--privileged-critic` lets the value network (training only) see the lead with no delay, the breathing gap and the lead's true path ahead (asymmetric actor-critic). The plugin cannot drive such a policy yet |
| `--reward NAME=VALUE` | changes one reward term (see `RewardConfig` in `drift_env.py`) |
| `--reward-from runs/<run>/config.json` | starts from an earlier run's reward |
| `--reward weave_penalty=5` | charges flicks the course does not ask for (weaving to shed speed) |
| `--reward swing_penalty=4` | charges reversals of the rotation the course does not ask for (swinging the angle to shed speed) |
| `--envs N` | trades ticks per second against learning per tick |

`--init-policy` keeps the old policy's observation scaling and gives the new value network 60
updates before the driving network moves, so expect the evaluation return to dip for the first
few million ticks.

`--init-policy` also keeps the old policy's paces, sliding grip and plugin settings. It does not keep
its car pool, its skill range or its reward (the policy file does not record the reward): give
those again, as the shipped policy's are in
[Polishing the shipped policy for one track](#polishing-the-shipped-policy-for-one-track).

## 3. See what it does

```
python flips.py runs/mine/policy.json --grip 0.5                  # how it changes direction
python dial_table.py runs/mine/policy.json                        # speed, drift time, zones ... at each RearSlidingGrip
python pace_report.py runs/mine/policy.json --track <track>       # with paces: does it drive the pace asked?
python drive_report.py runs/a/policy.json runs/b/policy.json --track <track> --human <recording>.csv   # side by side
python car_report.py runs/mine/policy.json <track>,<track> 0.9 out.json --cars cars/car_pool.json      # every car in the pool
```

`car_report.py` asks each car for its own speed on each track (its pace scale, measured as setup
does).

## 4. Use it on a server

Copy `runs/mine/policy.json` into the server's `plugins/GradientDriftPlugin/` folder as
`GradientDriftPlugin.policy.json` (keep the old one), and restart the server. No rebuild is needed.

> [!IMPORTANT]
> A policy with paces also needs `tracks/<track>/<track>_pace_reference.csv` in the server folder,
> and one trained on a car pool needs each bot car's file
> (`plugins/GradientDriftPlugin/cars/<model>.json`). The setup program writes both. It also
> measures each car's pace scale on the server's track (how fast it goes flat out there against
> the reference car), so a slow car is not asked for a fast one's speed.

## 5. Check a server run

Turn on `EnableTelemetryLogging: true` in the plugin's `extra_cfg.yml` block (and
`LogHumanCarForCalibration: true` for a human driver to compare against). Each server start writes
a new telemetry file:

```
python check_plugin_policy.py drift_physics_telemetry_<date>_<time>.csv --policy runs/mine/policy.json   # same network, same inputs?
python replay_live.py drift_physics_telemetry_<date>_<time>.csv --policy runs/mine/policy.json           # tick by tick against the simulator
python drive_report.py --telemetry drift_physics_telemetry_<date>_<time>.csv --human <recording>.csv     # the bots against a human
```

## Polishing the shipped policy for one track

The shipped policy is trained on several tracks and generated courses so that it drives tracks it
has never seen. Where it drives one of them badly, training it further on that track usually makes
it better there. Three things decide whether it works.

- **The track data has to be right.** Training only fixes what the simulator can see. If a wall is
  missing from the track data, or the AI line or the course edges are wrong, more training learns
  the wrong track. Look at the shipped policy on the track first (step 2): it should go wrong in
  the simulator where it goes wrong on the server.
- **It forgets other tracks.** A policy polished on one track drives others worse, so it belongs on
  that track's server only. Keep the shipped policy everywhere else.
- **The simulator is not the server.** Check the result with a server run (step 5) before you keep
  it.

**1. Make the track's data** ([1. Track data](#1-track-data)), and a car pool from your cars if
the server runs more than one car model ([Cars](#cars)).

**2. See how the shipped policy drives it**, to compare against later:

```
python drive_report.py ../GradientDriftPlugin/GradientDriftPlugin.policy.json --track <track>
```

**3. Train from the shipped policy** on that track:

```
python train_ppo_gpu.py --sim cpu --init-policy ../GradientDriftPlugin/GradientDriftPlugin.policy.json \
    --track <track> --generated 20 --car-pool cars/car_pool.json --skill 0 1 \
    --reward weave_penalty=5 --reward swing_penalty=4 --steps 30000000 --name <track>_polished
```

- `--car-pool` and `--skill 0 1` keep what the shipped policy can do: drive any car and follow the
  plugin's `BotSkill`. Without them the new policy drives only the plugin's configured car and has
  no skill dial. Leave `--car-pool` out only if every bot is that one car.
- `--reward weave_penalty=5 --reward swing_penalty=4` are the reward changes the project's own
  policies train with. Without them the run uses the trainer's default reward, which lets the car
  weave to shed speed.
- `--generated 20` keeps some practice on other shapes in the run. 30M ticks is what the project's
  own one-track trials use; Ctrl-C stops a run early and keeps what it has.

**4. Compare** the new policy with the shipped one on the track, on the same measures as step 2:

```
python drive_report.py ../GradientDriftPlugin/GradientDriftPlugin.policy.json runs/<track>_polished/policy.json --track <track>
```

Keep it only if it is better where the shipped one went wrong and no worse off the course, on the
walls or spinning. Otherwise stay with the shipped policy and look at the track data again.

**5. Put it on that track's server only** ([4. Use it on a server](#4-use-it-on-a-server)): copy
`runs/<track>_polished/policy.json` into the server's `plugins/GradientDriftPlugin/` as
`GradientDriftPlugin.policy.json` (keep the old one), and copy `tracks/<track>/<track>_pace_reference.csv`
into the server folder too, since the trainer may have remade it. Restart, check the log's
`GradientDrift: policy loaded from` line, and run the server with telemetry to check it
([5. Check a server run](#5-check-a-server-run)). A later plugin update copies the shipped policy
over it again: copy yours back afterwards.

## Cars

A car pool is the numbers of the cars you train on, made from your own game install (no car data
ships with this project). Unpack each car's `data.acd`, then read them all into one file:

```
python cars/acd_to_ini.py <game>/content/cars/<car>/data.acd <car> cars/unpacked/<car>     # once per car
python cars/car_numbers.py cars/unpacked/* --json cars/car_pool.json
```

Only rear-drive cars are kept. Hold one or two out of training (`--car-exclude`) to check the
policy on cars it has never driven.

## What the policy does

The top-level README draws it ("How the bot drives"). Each server tick the plugin hands the
network numbers it already has: speed, slip, yaw rate, where the racing line goes next, ray
distances to walls and course markers, the room to the course's edges, its last action, the pace
asked of it and how far over that pace it has been. The network sets three things:

| action | effect | range |
|---|---|---|
| steering | the steering command, as a fraction of full lock (the policy file's `MaxSteerAngleDeg`) | −1 … +1 |
| pedal | the share of thrust (above 0) or of the brakes (below 0), through the rear tyres' friction circle | −1 … +1 |
| front brake | a second, independent brake on the front axle only, added to whatever the pedal does (at most the front's share of the car's braking, and taking from the front tyres' grip for steering). With the throttle on, it is left-foot braking: speed comes off while the rear keeps the angle | 0 … 1 |

Policies from before an input or output was added read fewer inputs (46, 52 or 54 of the 55) and
have no brake output; the plugin and these tools run any of them. The plugin's safety nets keep
running underneath: wrong-way recovery, wall push-out, and the correctors for glitches in the
nearest racing-line point and its angle.

**The network** is two hidden layers (tanh; 128 units in the newest policies, 64 before), trained
with PPO (Schulman et al., 2017).

**The reward** pays for progress along the racing line, more while sliding at a drift angle, and
for linking drifts; with a pace, for no more distance than the pace covers, and less the further
the car strays from it. It costs for:

- walls, leaving the course, and hitting the course's markers (cones, tyre stacks);
- spinning and jerky steering;
- corner stretches driven not sliding, or with the tail on the wrong side;
- flicking from one side to the other, or swinging the angle open and shut, where the course does
  not ask for it (to shed speed).

`RewardConfig` in `drift_env.py` has every term. Each run's `config.json` records the values it
was trained with.

## Files

**Simulator and training**

| file | what it is |
|---|---|
| `fast_sim.py` | The plugin's vehicle model and sensing, compiled with numba: the nearest racing-line point and its corrections, the policy's steering and pedal, the car's dynamics, wrong-way recovery, wall push-out, car height, ray casts. |
| `drift_env.py` | The training environment: observations, reward (`RewardConfig`), episode endings, tick timing, starts. `DriftVecEnv` steps many cars at once. |
| `train_ppo_gpu.py` | The PPO trainer (PyTorch). |
| `gpu_env.py`, `sim_target.py`, `check_gpu.py` | The simulator compiled for an NVIDIA GPU from the same source, and its check. |
| `policy.py` | The network and the `policy.json` reader and writer. |
| `seed_pace.py` | Gives a policy a head start on driving a pace (it copies a scripted driver that brakes when too fast). |

**Tracks and cars**

| file | what it is |
|---|---|
| `setup_track.py` | A track's data in one command: its geometry, road mesh, racing line (checked, and rebuilt down the middle of the road where it needs it), course, zones and what the sensors see, then `check_track.py`. `--server` copies the plugin's files into a server folder. |
| `check_track.py` | Checks a track's data before training: speed targets the car cannot drive, the racing line too near a wall, walls too low for the policy's sensors. |
| `build_course.py`, `build_zones.py`, `build_road_mesh.py`, `line_tools.py`, `sensor_view.py` | The steps `setup_track.py` runs: the course (edges, markers), the outside zones, the road-mesh points, checking or rebuilding the racing line, and what the sensors see (the walls the game collides with and the markers). |
| `track_files.py`, `ai_spline.py` | Where each track's files are (`tracks/<track>/`); the racing-line reader, a port of AssettoServer's. |
| `generate_track.py` | Random closed courses (walled, tyre stacks on a pad, or grass-edged; hairpins, switchbacks, hills and camber) with every file a real track has, for training a policy to drive courses it has not seen. `--generated` makes them itself. |
| `migrate_tracks.py` | Copies track data from the layout before 2026-10-11 (`closed_loop_sweep_bundle/data/`) into `tracks/`. |
| `cars/acd_to_ini.py`, `cars/car_numbers.py` | Unpack a car's `data.acd` and read the vehicle-model numbers out of it (mass, geometry, tyres, steering, engine and gearing). What they extract stays on your machine. |

**Reports and checks**

| file | what it is |
|---|---|
| `drive_report.py` | Policies side by side and against a human recording: pace, sliding, rotation (yaw rate, yaw acceleration, reversals, measured the same way for both), sideways g, weaving, and time off the course, on the walls or spinning, in server-style runs. `--pace-reference` measures each course's flat-out speed in each policy's own physics; `--set NAME=VALUE` tries a plugin setting. |
| `flips.py` | How a policy changes direction: flips a minute, how fast, steering rate, the median flip sample by sample. `--set NAME=VALUE` tries a plugin setting. |
| `dial_table.py` | Speed, drift time, zones, steering calm, front tyres and wrong-side time at each `RearSlidingGrip` setting. |
| `pace_report.py` | How closely a policy drives the pace it is asked for. |
| `car_report.py` | Every car in a pool on each track, each asked for its own speed. |
| `check_plugin_policy.py` | After a server run: is the plugin running the same network on the same inputs, and is the run inside what the simulator predicts? |
| `replay_live.py` | After a server run: replays its telemetry tick by tick through the simulator and reports every quantity that disagrees. |
| `record_ghosts.py` | Records a policy's runs on a track as ghost leaders for chase training (`<track>_ghosts.npz`). |
| `chase_report.py` | How policies chase a ghost leader in the simulator (gap, angle and line against it, hits) and how they drive alone. |
| `tandem_report.py` | How the tandem-train followers in a plugin telemetry file behaved. |
| `judge.py` | Drivers side by side on the same course (bots from the plugin's telemetry, a human from the in-game recorder), lap by lap and zone by zone, with a rough Formula Drift score (line 40, angle 40, style 20, measured rather than judged). |
| `human_recording.py` | Speed, sideways acceleration, slip, RPM, gear and throttle of a human driver, from an in-game telemetry recording. |

## How far to trust the simulator

- **Tick by tick:** replaying a real server run, each tick predicted from the real one before it,
  the simulator lands within millimetres of the server's position (median 3 mm) and within
  thousandths of a degree of its heading.
- **Whole runs:** runs on the server have come out inside the range the simulator predicts for
  the same policy, usually within a percent or two on speed and drift time.
- **Known gaps:** a few per cent of ray readings that graze the ground, and how far the car is
  pushed out of thin walls.

Run `replay_live.py` after any server run that surprises you.
