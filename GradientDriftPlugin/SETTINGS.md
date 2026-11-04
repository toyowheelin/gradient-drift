# Gradient Drift plugin settings

The plugin reads its settings from a block in the server's `cfg/extra_cfg.yml`:

```yaml
EnablePlugins: [GradientDriftPlugin]
--- !GradientDriftConfiguration
EnableLearnedPolicy: true
BotSkill: 1
BotPacePercent: 90
ChaseBots: false
```

Every setting has a working default, so the block can hold only what you want to change. The setup
program writes `EnablePlugins`, `EnableLearnedPolicy` and `BotPacePercent` for you, turns
`HoldBotSpacing` back on for several bots, and takes the retired `TrainSize` out.
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
| `LearnedPolicyPath` | empty | The policy file to run. Empty means `GradientDriftPlugin.policy.json` beside the plugin's DLL, which is where the release puts it, or its copy in `GradientDriftPlugin.bundle.zip` there when there is no loose file. |

### How the bots drive

| Setting | Default | What it does |
|---|---|---|
| `BotSkill` | `1` | How good a driver the bots are, 0 to 1. **1** is the best the policy can do. **0** is a driver a track day would still let out with others: holds a drift and rarely spins, but at a shallower angle, on a more cautious line, with slower and sloppier hands. Only for a policy trained with a skill input (gen9 and later); others ignore it. Kept inside the range the policy was trained for. |
| `BotPacePercent` | `0` | How fast the bots are asked to go, as a percentage of the track's flat-out drift speed at each point of the lap. 90 means 90% everywhere, on any track. Lower means calmer bots that brake more into corners and slide a little less. Kept inside the policy's trained range (80 to 120 for the newest policies, 80 to 100 for older ones); **0** (or not set) means 100, held inside that range; before v0.3.0 it meant the top of the range. Spacing and leading a player only ever slow a bot from this pace, never speed it past it. Needs the track's `<track>_pace_reference.csv` in the server folder (setup puts it there). |
| `BotPaceMinPercent` | `85` | With several bots out, spacing speeds some up and slows others down (see `HoldBotSpacing`). This is the slowest it may ask for. Below about 85% the bots slide noticeably less, so the default keeps them above that. 0 means the bottom of the policy's range. |
| `HoldBotSpacing` | `true` | With more than one bot out, spreads them evenly round the lap. A bot the plugin takes over is placed in the middle of the biggest gap near a player, then a bot that is catching the one ahead is slowed a little to keep the gaps even (never sped past `BotPacePercent`). Spaced bots don't see each other (only a chasing bot sees the car it chases), and cars don't collide: if one spins, the car behind can drive through it. |
| `RearSlidingGrip` | `0.5` | How much sideways grip the rear tyres keep once sliding. Current policies are trained at exactly 0.55, so for them any value is moved to 0.55 (with a warning in the log). Only matters for an older policy trained over a range of grips. |

### Tandem

| Setting | Default | What it does |
|---|---|---|
| `TrainSize`, `TandemGapSeconds`, `TandemGapVariation`, `TandemMinGapMeters`, `TandemLateralOffsetMeters` | | **Retired in v0.3.0.** Trains of followers replaying a lead car are gone: bots form trains by chasing each other, each driving itself (`ChasePlayers` with `ChaseBots`, below). The plugin still accepts these settings, so an old `extra_cfg.yml` starts, but ignores them (a `TrainSize` above 1 gets a warning in the log). Setup takes `TrainSize` out. |
| `LeadPlayers` | `false` | A bot with a player chasing close behind waits for them: its own pace (`BotPacePercent`) with the player right on its tail, slowing to `BotPaceMinPercent` as they drop back to 40 m. Further back, the bots space themselves as usual. Only for a policy that reads its pace (every current one). |
| `ChasePlayers` | `true` | A bot chases a player who pulls in within `ChaseOptInMeters` (7.5 m, about 25 ft) in front of its nose, until the player gets `ChaseOptOutMeters` (40 m) ahead. Its pace comes from the Intelligent Driver Model, the standard car-following model of traffic simulation, aiming `ChaseGapMeters` behind the player and easing off early when it closes fast. Only for a policy that reads its pace (every current one). With a tandem policy (one trained to chase, as the shipped one is: with the car-ahead inputs, in one network or through an added residual network) the bot also sees the player, as in training: where they are, their gap between the car outlines, line, slip, steering, throttle and brake lights, and its pace takes in the player's acceleration and may drop to the chase floor the policy was trained with. A policy that reads its pace but was not trained to chase follows by pace alone, blind to the car ahead. On by default since v0.3.0; it does nothing for an older policy that doesn't read its pace. |
| `ChaseGapMeters` | `2` | With `ChasePlayers`: the gap the chasing bot aims for behind the player, bumper to bumper, before the speed-dependent margin the model adds. |
| `ChaseOptInMeters` | `7.5` | With `ChasePlayers`: how far ahead of a bot's nose, along the line, a player has to be for the bot to start chasing them. Bigger has bots join a tandem from further back. |
| `ChaseOptOutMeters` | `40` | With `ChasePlayers`: how far ahead the player has to get before the bot stops chasing. |
| `ChaseTimeGapSeconds` | `0.2` | With `ChasePlayers`: the speed-dependent margin, in seconds of the bot's speed added to `ChaseGapMeters`. Lower sits closer at speed but leaves less room when the player slows. |
| `ChaseResponseSeconds` | `0.6` | With `ChasePlayers`: how quickly the bot's pace answers the gap, in seconds. Lower closes up and backs off faster. |
| `ChasePaceMaxPercent` | `110` | With `ChasePlayers`: the most pace the chasing bot may ask, as a percentage of the policy's top pace. Over 100 gives it some speed in hand to catch a player who drives at the bots' top pace. |
| `ChaseBots` | `true` | With `ChasePlayers`: a bot also chases a bot that comes within `ChaseOptInMeters` ahead of it, so drift trains of bots form on their own. Whatever this is, every car (player or bot) has at most one bot chasing it: a second bot coming up behind a chaser waits at its usual pace, or with this on chases the chaser. On by default since v0.3.0: set it to `false` to keep bots spaced round the lap and chasing players only. |
| `ChaseTrainMax` | `3` | With `ChaseBots`: the most bots chasing in one line, behind a player or a bot that is not chasing. Longer trains don't form. |
| `ChaseBotRunSeconds` | `60` | With `ChaseBots`: how long a bot chases another bot before it drops back (0 = no limit). Chasing a player is never timed. |
| `ChaseBotCooldownSeconds` | `30` | With `ChaseBots`: after a bot-on-bot chase ends, how long that bot takes up no other bot, so trains break up and form again. |
| `CarContactPush` | `false` | Bots feel other cars' mass. A bot that touches a player's car or another bot takes its share of the collision: it is pushed back (or along, if hit from behind), and an off-centre hit turns it. When two bots touch, both are pushed. The player's own game already gives the player its share. With this off, a bot drives through a hit as if nothing were there. Policies trained with the push (`--chase-touch-push`) expect it on. |

### Each bot's own style

So a server full of bots doesn't look like clones. Each bot can have its own **aggression**, **chase gap**
and **skill**. These are all things the policy already reads (its pace, the gap it is asked to hold, its
skill) or chase rules the plugin applies, so no retraining is needed, and every value stays inside the
range the policy was trained for. A bot keeps its style through respawns and server restarts. With
`VerboseLogging` on, the log shows each bot's style the first time it drives.

Aggression runs from 0 to 1. **0.5** means the settings above, unchanged. Going up from there the bot asks for
more pace (up to the policy's top), takes up a car to chase from further back (up to 1.6 × `ChaseOptInMeters`)
and may chase up to 10% faster than `ChasePaceMaxPercent`. It also drops the time gap: at 1 it holds just its
chase gap at any speed (no `ChaseTimeGapSeconds`), which is what gets a chaser door to door. Going down it asks
for less pace (down to `BotPaceMinPercent`), only takes up a car close ahead (down to 0.4 × `ChaseOptInMeters`),
chases up to 10% slower, and keeps up to twice `ChaseTimeGapSeconds`.

| Setting | Default | What it does |
|---|---|---|
| `BotStyleSpread` | `0` | How far apart the bots' styles are drawn, 0 to 1. **0** means every bot drives with the settings above, as before. **1** means aggression anywhere from 0 to 1 and chase gaps from half `ChaseGapMeters` to twice it. |
| `BotSkillSpread` | `0` | Each bot's skill is `BotSkill` minus a random amount up to this (0 to 1). **0** means every bot is at `BotSkill`. |
| `BotStyleSeed` | `0` | Change it to draw a new set of styles. |
| `BotStyles` | empty | Styles set by hand, which win over the draws. Each entry names a bot by its `DRIVERNAME` in `entry_list.ini`. An entry with no `Name` applies to the AI entries in order. Any of `Aggression` (0 to 1), `ChaseGapMeters` and `Skill` can be set; anything left out is drawn or taken from the settings. |

```yaml
BotStyleSpread: 0.6
BotStyles:
  - Name: Kenji
    Aggression: 0.9
    ChaseGapMeters: 1
  - Name: Mia
    Aggression: 0.2
    Skill: 0.6
```

### Fitting in with the players

With `AdaptToPlayers` on, the bots drive more like whoever is drifting with them, for as long as
they are on. The plugin watches the players and moves each bot's **pace** toward the players' pace
(their speed along the line against the track's pace reference) and, with a tandem policy, its
**chase gap** toward the gap players keep when they follow a car. Nothing is learned and nothing is
kept: these are numbers the policy already reads, held inside the range it was trained for, and when
the server is empty the bots ease back to the settings and the plugin forgets what it saw. Each
bot keeps its own style (above): every bot moves by the same amount.

Only players drifting on the line count: sliding at 12 to 75 degrees, within 5 m of the AI line,
driving its way. A gap counts only while a player follows a car closely (under 15 m) at about its
speed. Players spinning, crashing, off the line or cruising are left out. With `VerboseLogging` on,
the log says when the bots start fitting in, the players' pace and gap once a minute, and when the
server empties.

| Setting | Default | What it does |
|---|---|---|
| `AdaptToPlayers` | `false` | Turns it on. |
| `AdaptToPlayersStrength` | `1` | How far the bots move toward the players, 0 to 1. **1** means all the way, **0.5** halfway. |
| `AdaptToPlayersLearnSeconds` | `60` | Seconds of drifting players seen before the bots move all the way. They move part of the way sooner. |
| `AdaptToPlayersForgetSeconds` | `120` | Seconds with nobody on the server before the bots are back on the settings. |
| `AdaptToPlayersAverageSeconds` | `60` | Seconds the players' pace and gap are averaged over. |

Note: the pace can rise to the top of the policy's range when fast players are on, which is faster
than spacing and leading ever ask for. If bots run wide at the top end, lower `AdaptToPlayersStrength`.
Drift angle does not adapt: the policy has no input for it yet.

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

The plugin writes no files unless you turn on telemetry or the scorecard. Everything else goes to the server's log
and console, and only when there is something to say: a line or two at start (the policy loaded), a warning
when the plugin cannot drive the bots (the stock AI drives then) or when a bot is stuck or held
stopped for 30 seconds, and a line when it moves again. Nothing while the bots drive normally.

| Setting | Default | What it does |
|---|---|---|
| `EnableTelemetryLogging` | `false` | Writes `drift_physics_telemetry_<date>_<time>.csv` in the server folder: one row per bot per tick. For checking a policy on a server with `drift_rl/check_plugin_policy.py` and `drive_report.py --telemetry`. About 110 MB an hour with six bots. |
| `TelemetryMaxFileMegabytes` | `500` | The telemetry file stops growing at this size. 0 = no limit. It also stops, with a line in the log, if a write fails. |
| `EnableScorecard` | `false` | A short report card per server session of how the bots drove: slide and spin share, wall time and wall hits, car hits (bots and players), chase time and gap (p10/p50/p90, share within 2 m of `ChaseGapMeters`), how often the server's own AI had them, stuck episodes, wrong-way recoveries. Only time with a player on the server counts: once the last one leaves, the bots still out are being taken away and are left out (the card says how many bot-minutes that was). Written as `<start>_<track>_<session>.txt` (a table) and `.json` into `ScorecardFolder`; `drift_rl/scorecard_report.py` puts several side by side. A few KB a session. |
| `ScorecardFolder` | `gradientdrift_scorecards` | Where the scorecards go, relative to the server folder. |
| `ScorecardWriteMinutes` | `10` | The scorecard is rewritten this often while a session runs (so a crash loses at most this much), and once more when the session ends or the server stops. |
| `LogLearnedPolicyObservations` | `true` | Adds the network's inputs to every telemetry row (about 330 bytes a row). The check tools need them. |
| `LogHumanCarForCalibration` | `false` | Adds the first connected player's car to the telemetry: position, steering, RPM, gear, throttle, wheel speeds. No name, IP or Steam ID. Used to compare the bots with a human driver. |
| `VerboseLogging` | `false` | The plugin's routine lines as well: what it loaded, each setting the policy file brings, each bot car's numbers, spawns and placements, and the timing line every 60 s. Off, the log has warnings only. |
| `ShowEntryListNames` | `true` | Show each bot in game under its entry list `DRIVERNAME` instead of AssettoServer's `<NamePrefix> <slot>` (`Traffic 16`). A bot with no `DRIVERNAME` keeps AssettoServer's name. |
| `BotNation` | `""` | Country code shown with the bots' names (e.g. `ITA`); empty = none. |
| `JoinMessage` | `Drift bots by Gradient Drift https://github.com/toyowheelin/gradient-drift` | A line added to the welcome message every player gets on joining, just above AssettoServer's "Powered by AssettoServer" line; the game shows it in chat. Empty (`JoinMessage: ""`) = no line. |
| `LogTimingIntervalSeconds` | `0` | Every this many seconds, a log line with how long the plugin took per server tick (network, sensors, telemetry, followers), the worst tick, and the share of one CPU core. 0 = off (every 60 s with `VerboseLogging`). |

## Developer settings

These describe the car, the sensors and the safety net the policy was trained with. The trainer
(`drift_rl`) simulates exactly these numbers, so a change here means the policy is driving a car it
has never seen. To change the car or the sensors, change them in the trainer and train a new policy.

Many are replaced at start, without anything in the YAML:

- **By the policy file** (`plugin_settings` in `policy.json`): the steering lock and steering speed,
  the brakes (`FrontBrakeOnly`, `DisableBrakes`), the rear thrust, brake and friction-circle models
  and `RearSpinGripShare`, tyre load sensitivity and its numbers, the tyre curve (`RoundTireCurve`),
  `SurfaceGrip`, the height of the wall sensors, and the wrong-way recovery thresholds. The plugin
  switches on only the features a policy was trained with, so an older policy keeps driving the car
  it knew.

  A new setting a policy file can carry goes in `DriftPolicy.CanApplySetting` and the apply switch of `ApplyPolicySettings`; then run
  `python tools/plugin_features.py`, which writes `GradientDriftPlugin.features.json` (what policy
  files this build can run) from the source. Setup checks a policy against that file before it
  installs the pair, and CI fails while the file is out of date.
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
| `SurfaceGrip` | `false` | Policy file. Each axle's peak friction times the grip of the ground under it: its surface's `FRICTION` in the track's `surfaces.ini` over the main road's, from `<track>_collision_grip.csv.gz` (setup writes it beside the other track files; without it every surface grips 1, with a warning) |
| `RearSpinGripShare` | `1` (off) | Policy file. With `EnableRearFrictionCircle`: the rear friction circle while the throttle spins the tires, as a share of their peak; it shrinks toward this with the share of full-throttle thrust used |
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
| `RoundTireCurve` | `false` | Policy file. Each axle's sideways force follows the brush (Fiala) curve: three times as steep as the straight line at small slip angles, rounding over into full sliding at the tyres' `FRICTION_LIMIT_ANGLE` plus `FLEX` x the static load on one tire (car file `FRONT_TIRE_FLEX` / `REAR_TIRE_FLEX`, else `FrontTireFlex` / `RearTireFlex`). Matches Assetto Corsa's front tyre, which gives 75 - 90% of its peak by 4 - 8 deg |
| `FrontTireFlex`, `RearTireFlex` | `0.000852` | tyres.ini `FLEX`, degrees of slip angle per N, for cars without a car file |
| `GroundRayHeightOffsetM` | `0.40` | Ride height above the ground found below the car. The bot drives and senses at this height; clients are sent each car at its own CG height (`CG_HEIGHT_M` in its car file) instead |
| `GroundRayStartAboveMeters` | `1.5` | |
| `GroundRayMaxDistanceMeters` | `6.0` | |
| `GroundRayHeightSmoothingFactor` | `1.0` | |
| `RideHeightSpanMeters` | `3.0` | The car rides on the straight line best fitting the ground this far ahead of and behind it along its path (27 points: 9 along each of its centre line and both sides): bumps shorter than about twice this smooth out, hills are followed exactly. `0` = the ground straight below only |
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
