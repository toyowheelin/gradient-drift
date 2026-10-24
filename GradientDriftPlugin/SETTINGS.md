# Gradient Drift plugin settings

The plugin reads its settings from a block in the server's `cfg/extra_cfg.yml`:

```yaml
EnablePlugins: [GradientDriftPlugin]
--- !GradientDriftConfiguration
EnableLearnedPolicy: true
BotSkill: 1
BotPacePercent: 90
TrainSize: 3
```

Every setting has a working default, so the block can hold only what you want to change. The setup
program writes `EnablePlugins`, `EnableLearnedPolicy`, `TrainSize` and `BotPacePercent` for you.
Settings are read when the server starts: restart it after a change.

The settings come in two kinds:

- **Operator settings** (below): how many bots, how fast, how good, how they look, what gets logged.
  Change these freely.
- **Developer settings** (at the end): the vehicle model and sensors the policy was trained with.
  Leave them alone. A policy only drives well in the car it learned in, so changing one of these
  makes the bots worse, not different. Most of them are overridden anyway, by the policy file or by
  the car files setup writes.

## Operator settings

### Turning it on

| Setting | Default | What it does |
|---|---|---|
| `EnableLearnedPolicy` | `true` | Lets the plugin drive the AI cars. Off, or no usable policy file, and the server's own AI drives them as before. The shipped policy drives any track, but the plugin stands aside (and says why in the log) if the track's data files are missing from the server folder (setup puts them there: the track's geometry, course edges and pace reference, made for the server's AI line), or if the policy needs a feature this build of the plugin doesn't have. An older policy trained only on real tracks also stands aside on a track it wasn't trained on. |
| `LearnedPolicyPath` | empty | The policy file to run. Empty means `GradientDriftPlugin.policy.json` beside the plugin's DLL, which is where the release puts it. |

### How the bots drive

| Setting | Default | What it does |
|---|---|---|
| `BotSkill` | `1` | How good a driver the bots are, 0 to 1. **1** is the best the policy can do. **0** is a driver a track day would still let out with others: holds a drift and rarely spins, but at a shallower angle, on a more cautious line, with slower and sloppier hands. Only for a policy trained with a skill input (gen9 and later); others ignore it. Kept inside the range the policy was trained for. |
| `BotPacePercent` | `0` | How fast the bots are asked to go, as a percentage of the track's flat-out drift speed at each point of the lap. 90 means 90% everywhere, on any track. Lower means calmer bots that brake more into corners and slide a little less. Kept inside the policy's trained range (80 to 100 for current policies); **0** means the top of that range (100 for current policies). Needs the track's `<track>_pace_reference.csv` in the server folder (setup puts it there). |
| `BotPaceMinPercent` | `85` | With several bots out, spacing speeds some up and slows others down (see `HoldBotSpacing`). This is the slowest it may ask for. Below about 85% the bots slide noticeably less, so the default keeps them above that. 0 means the bottom of the policy's range. |
| `HoldBotSpacing` | `true` | With more than one bot (or train) out, spreads them evenly round the lap. A bot the plugin takes over is placed in the middle of the biggest gap near a player, then each bot's pace is nudged up or down a little to keep the gaps even. The bots don't see each other: if one spins, the car behind can drive through it. |
| `RearSlidingGrip` | `0.5` | How much sideways grip the rear tyres keep once sliding. Current policies are trained at exactly 0.55, so for them any value is moved to 0.55 (with a warning in the log). Only matters for an older policy trained over a range of grips. |

### Tandem trains

| Setting | Default | What it does |
|---|---|---|
| `TrainSize` | `1` | Bots per train, in the order of their entries in `entry_list.ini`. The first of each train is driven by the policy; the others follow it, replaying what it did a moment earlier, so they hold its line door to door. **1** means every bot drives alone. |
| `TandemGapSeconds` | `0.35` | How far each follower runs behind the car in front, in seconds. 0.35 s is a little over a car length at drift speed. |
| `TandemGapVariation` | `0.25` | The gap wanders by up to this share either way (0.25 = ±25%), slowly and differently for each car, so a train doesn't look robotic. |
| `TandemMinGapMeters` | `4.8` | Never closer than this to the car in front, nose to nose. A car is about 4.4 m long. |
| `TandemLateralOffsetMeters` | `0.6` | How far a follower may sit to one side of the lead's line. Each wanders between about a fifth of this and all of it, on a side of its own. |

### How the bots look (cosmetic only)

None of these change how a bot drives; they change what players see and hear.

| Setting | Default | What it does |
|---|---|---|
| `EnableSteeringVisual` | `true` | Shows the bot's real steering: front wheels and steering wheel turn on screen. The server's own AI always shows them straight. |
| `SteeringVisualMaxRateDegS` | `360` | Fastest the shown steering may move, degrees a second, so the wheels don't snap from lock to lock. 0 = no limit. |
| `EnableDrivetrainVisual` | `true` | Sends wheel speeds, engine RPM, gear and throttle that match what the car is doing, so it sounds like it's on the limiter in a slide, with the rear wheels spinning. |
| `BodyRollDegPerG` | `0.85` | How far the body leans outward in a turn, degrees per g of sideways acceleration. 0 = none. |
| `BodyPitchDegPerG` | `0.57` | How far the nose lifts under power, degrees per g. 0 = none. |

Note: the drivetrain visual uses one gearbox and rev range for every bot (the Miata's, from the
developer settings), whatever car it is. Each car's own gearing is not used for the sound yet.

### Logging

The plugin writes no files unless you turn on telemetry. Everything else goes to the server's log
and console, and only when there is something to say: a line or two at start (the policy loaded), a warning
when the plugin cannot drive the bots (the stock AI drives then) or when a bot is stuck or held
stopped for 30 seconds, and a line when it moves again. Nothing while the bots drive normally.

| Setting | Default | What it does |
|---|---|---|
| `EnableTelemetryLogging` | `false` | Writes `drift_physics_telemetry_<date>_<time>.csv` in the server folder: one row per bot per tick. For checking a policy on a server with `drift_rl/check_plugin_policy.py` and `drive_report.py --telemetry`. About 110 MB an hour with six bots. |
| `TelemetryMaxFileMegabytes` | `500` | The telemetry file stops growing at this size. 0 = no limit. It also stops, with a line in the log, if a write fails. |
| `LogLearnedPolicyObservations` | `true` | Adds the network's inputs to every telemetry row (about 330 bytes a row). The check tools need them. |
| `LogHumanCarForCalibration` | `false` | Adds the first connected player's car to the telemetry: position, steering, RPM, gear, throttle, wheel speeds. No name, IP or Steam ID. Used to compare the bots with a human driver. |
| `VerboseLogging` | `false` | The plugin's routine lines as well: what it loaded, each setting the policy file brings, each bot car's numbers, spawns and placements, and the timing line every 60 s. Off, the log has warnings only. |
| `LogTimingIntervalSeconds` | `0` | Every this many seconds, a log line with how long the plugin took per server tick (network, sensors, telemetry, followers), the worst tick, and the share of one CPU core. 0 = off (every 60 s with `VerboseLogging`). |

## Developer settings

These describe the car, the sensors and the safety net the policy was trained with. The trainer
(`drift_rl`) simulates exactly these numbers, so a change here means the policy is driving a car it
has never seen. To change the car or the sensors, change them in the trainer and train a new policy.

Many are replaced at start, without anything in the YAML:

- **By the policy file** (`plugin_settings` in `policy.json`): the steering lock and steering speed,
  the rear thrust, brake and friction-circle models, tyre load sensitivity and its numbers, and the
  wrong-way recovery thresholds. The plugin switches on only the features a policy was trained with,
  so an older policy keeps driving the car it knew.
- **By the car files** (`plugins/GradientDriftPlugin/cars/<model>.json`, written by setup) for a
  policy that drives any car (gen8 and later): each bot's weight, axle geometry, yaw inertia, tyre
  grip and load sensitivity, track width, CG height, steering lock and engine. A car without a file
  falls back to the values below, with a warning in the log.

### The car (a two-axle "bicycle" model; defaults are the GravyGarage Miata)

| Setting | Default | |
|---|---|---|
| `YawInertiaKgM2` | `2199.6` | Car file |
| `DistanceCgToFrontAxleM` | `1.035` | Car file |
| `DistanceCgToRearAxleM` | `1.265` | Car file |
| `FrontStaticLoadN` | `6132.6` | Car file |
| `RearStaticLoadN` | `5017.6` | Car file |
| `LateralDynamicsFrontPeakFriction` | `1.3745` | Car file |
| `LateralDynamicsRearPeakFriction` | `1.3385` | Car file |
| `FrontPeakSlipAngleRadians` | 11.7° | Car file |
| `RearPeakSlipAngleRadians` | 12.0° | Car file |
| `MaxSteerAngleRadians` | 28.125° | Policy file (`MaxSteerAngleDeg`), car file |
| `MaxSteerRateDegS` | `0` (no limit) | Policy file |
| `EnableTireLoadSensitivity` | `false` | Policy file |
| `FrontTireLoadSensitivityPerKn` | `-0.044` | Policy file, car file |
| `RearTireLoadSensitivityPerKn` | `-0.049` | Policy file, car file |
| `FrontTrackMeters` | `1.67` | Policy file, car file |
| `RearTrackMeters` | `1.50` | Policy file, car file |
| `RollStiffnessFrontShare` | `0.67` | Policy file, car file |
| `CgHeightMeters` | `0.40` | Policy file, car file |
| `EnableRearThrustLimit` | `false` | Policy file |
| `EnableRearBrakeLimit` | `false` | Policy file |
| `EnableRearFrictionCircle` | `false` | Policy file |
| `DisableBrakes` | `false` | Policy file. With `EnableRearFrictionCircle`: no pedal braking and no brake action, so the car slows only by lifting and with its angle (gen18 on) |
| `FrontBrakeOnly` | `false` | Policy file. With `EnableRearFrictionCircle`: no pedal braking, but the brake action (the front axle, with the throttle) still works, so the car can scrub speed and keep its angle |
| `MaxLongitudinalDecelMs2` | `5.6` | Hardest braking, m/s² |
| `MinSlipAngleSpeedMs` | `3.0` | Speed floor in the slip-angle formula |
| `MaxLateralDynamicsSubStepSeconds` | `0.05` | Longest physics sub-step |
| `MaxRealDtSeconds` | `0.2` | A longer server tick is treated as this long |

### Wrong-way recovery (the one safety net)

If a car points more than the threshold away from the racing line for a few ticks, its rotation and
sideways speed are damped until it's straight again. The policy is trained with this in place.

| Setting | Default | |
|---|---|---|
| `EnableWrongWayRecovery` | `true` | |
| `WrongWayEntryThresholdDeg` | `67` | Policy file |
| `WrongWayEntryTicks` | `3` | Policy file |
| `WrongWayExitSlipDeg` | `30` | Policy file |
| `WrongWayExitYawRateDegS` | `20` | Policy file |
| `WrongWayRecoveryTimeConstantSeconds` | `0.3` | |

### Sensors, walls and ride height

The policy reads these sensors' outputs directly, so their geometry has to match training.

| Setting | Default | |
|---|---|---|
| `RaycastFanRayCount` | `15` | Rays in the fan ahead (the policy reads 15) |
| `RaycastFanAngleDegrees` | `120` | Width of the fan |
| `RaycastFanMaxDistanceMeters` | `60` | |
| `RaycastHeightOffsetMeters` | `0.5` | |
| `ProximityPinMaxDistanceMeters` | `12` | The six short rays fixed to the body |
| `ProximityPinHeightOffsetMeters` | `0.15` | Above the car's position (itself `GroundRayHeightOffsetM` above the road), for the six pins and the contact rays. Policy file: gen14 on lowers them to feel tire walls under 0.55 m tall (Spirit Peaks' straight) |
| `RaycastGridTargetTrianglesPerCell` | `8` | Speed only; doesn't change what the rays hit |
| `CollisionContactDistanceMeters` | `1.6` | A wall this close along a short ray pushes the car back out (only walls stop these rays, not the road) |
| `CollisionPushOutMarginMeters` | `0.3` | |
| `CarOutlineContact` | `false` | Policy file. Walls touch the car's own outline (a rectangle turned with its heading: each axle's distance from the centre plus `CarOutlineOverhangMeters`, half the wider track plus `CarOutlineSideMarginMeters`) instead of the contact distance's circle, so a sliding car's tail can't pass through a wall |
| `CarOutlineOverhangMeters` | `0.8` | |
| `CarOutlineSideMarginMeters` | `0.1` | |
| `GroundRayHeightOffsetM` | `0.40` | Ride height above the ground found below the car (the mean of the ground under its centre and, where found, under both axles and both sides, so short bumps don't make it hop) |
| `GroundRayStartAboveMeters` | `1.5` | |
| `GroundRayMaxDistanceMeters` | `6.0` | |
| `GroundRayHeightSmoothingFactor` | `1.0` | |
| `GroundRayMissHoldSeconds` | `0.5` | Where the ground ray finds nothing (a crack in the track's ground mesh), the last height is held this long before the fallback below |
| `RoadMeshHeightOffsetM` | `0.44` | Fallback height when the ground ray finds nothing |
| `CrossTrackBlendStartM` | `5.0` | |
| `CrossTrackBlendEndM` | `12.0` | |
| `MaxTrustedMeshDistanceM` | `8.0` | |
| `HeightSmoothingFactor` | `0.15` | |
| `HeightSnapThresholdMeters` | `0.15` | |
| `PitchRollSmoothingFactor` | `0.5` | |
| `MaxLateralAccelMs2` | `13.3` | Bounds how fast the racing line's direction may jump between ticks |

### Visual calibration

Measured from a human driver's client; only change them if the wheels or sound look wrong on a new
version of the game.

| Setting | Default | |
|---|---|---|
| `SteeringVisualSign` | `1` | Direction of the shown steering |
| `SteeringVisualWheelByteUnitsPerDegree` | `2` | Encoding of the road-wheel angle |
| `DrivetrainVisualGearRatios` | Miata: 3.321, 1.902, 1.408, 1.150, 0.889 | |
| `DrivetrainVisualFinalDrive` | `3.8` | |
| `DrivetrainVisualIdleRpm` | `1000` | |
| `DrivetrainVisualRevLimitRpm` | `8750` | |
| `DrivetrainVisualTyreDiameterMeters` | `0.52` | |
| `DrivetrainVisualUpshiftRpm` | `6000` | |
| `DrivetrainVisualDownshiftRpm` | `2500` | |
| `DrivetrainVisualWheelSpeedLagSeconds` | `0.15` | How quickly wheel speed and RPM follow wheelspin |

### Older policies only

| Setting | Default | |
|---|---|---|
| `BotPaceKmh` | `0` | The pace for a gen4 policy, whose paces are km/h rather than a share of flat out. 0 = the top of its range. |
