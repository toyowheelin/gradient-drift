<p align="center"><img src="docs/logo.jpg" alt="Gradient Drift: reinforcement-learning drift bots for AssettoServer" width="100%"></p>

<h1 align="center">gradient-drift</h1>

<p align="center">
  <b>Reinforcement-learning drift bots for <a href="https://github.com/compujuckel/AssettoServer">AssettoServer</a> (Assetto Corsa)</b>
</p>

<p align="center">
  <a href="https://github.com/toyowheelin/gradient-drift/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/toyowheelin/gradient-drift?style=plastic&logo=github&label=release"></a>
  <a href="https://github.com/toyowheelin/gradient-drift/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/toyowheelin/gradient-drift/total?style=plastic&logo=github&color=brightgreen"></a>
  <a href="https://github.com/toyowheelin/gradient-drift/actions/workflows/build-plugin.yml"><img alt="Plugin build" src="https://img.shields.io/github/actions/workflow/status/toyowheelin/gradient-drift/build-plugin.yml?branch=main&style=plastic&logo=githubactions&logoColor=white&label=plugin%20build"></a>
  <a href="LICENSE"><img alt="Licence: AGPL-3.0" src="https://img.shields.io/badge/licence-AGPL--3.0-blue?style=plastic&logo=gnu&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://github.com/compujuckel/AssettoServer"><img alt="AssettoServer v0.0.54" src="https://img.shields.io/badge/AssettoServer-v0.0.54-EA141E?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAEQAAAAqCAYAAAAK7f2YAAAACXBIWXMAAC4jAAAuIwF4pT92AAAGWWlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNi4wLWMwMDIgNzkuMTY0MzUyLCAyMDIwLzAxLzMwLTE1OjUwOjM4ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgMjEuMSAoV2luZG93cykiIHhtcDpDcmVhdGVEYXRlPSIyMDIyLTA5LTI3VDE1OjE4OjQ1KzAyOjAwIiB4bXA6TW9kaWZ5RGF0ZT0iMjAyMi0xMi0xNlQxODoxMTozNyswMTowMCIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAyMi0xMi0xNlQxODoxMTozNyswMTowMCIgZGM6Zm9ybWF0PSJpbWFnZS9wbmciIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiIHBob3Rvc2hvcDpJQ0NQcm9maWxlPSJzUkdCIElFQzYxOTY2LTIuMSIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDpiNjU4ZmU5Mi0wMGQ4LWY0NGQtYmE3ZC1jM2RmMWExMDc4OTIiIHhtcE1NOkRvY3VtZW50SUQ9ImFkb2JlOmRvY2lkOnBob3Rvc2hvcDo1MjZiMGNiZC1lNTY2LTM5NDMtOGJhNS1hZTI5NGVlODJiYjYiIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDplMmViNWVlNC05YjUzLTkyNDctYTQ3OS0zNzE0YmNlYjRkM2YiPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE%2BIDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJjcmVhdGVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOmUyZWI1ZWU0LTliNTMtOTI0Ny1hNDc5LTM3MTRiY2ViNGQzZiIgc3RFdnQ6d2hlbj0iMjAyMi0wOS0yN1QxNToxODo0NSswMjowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDIxLjEgKFdpbmRvd3MpIi8%2BIDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJjb252ZXJ0ZWQiIHN0RXZ0OnBhcmFtZXRlcnM9ImZyb20gYXBwbGljYXRpb24vdm5kLmFkb2JlLnBob3Rvc2hvcCB0byBpbWFnZS9wbmciLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOmI2NThmZTkyLTAwZDgtZjQ0ZC1iYTdkLWMzZGYxYTEwNzg5MiIgc3RFdnQ6d2hlbj0iMjAyMi0xMi0xNlQxODoxMTozNyswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDIxLjEgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8%2BIDwvcmRmOlNlcT4gPC94bXBNTTpIaXN0b3J5PiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI%2FPmn8iCQAAAGjSURBVGje5dnREcIgDABQR%2BiH387gCI7SERyhG3QERnCEjtAR%2FPenIyC946OHQElIMG24y%2FlR9MqzhEAvF4Htc73dXSwuHtDvWmuNxTcjGcNCQc6OAQLRgFEMogWjCEQTxi6INowsiEaMJIhWjCiIZowfEDegUTNGDGTSjEEFIh5j%2FezXwW6iL6lDECDiMWYXXaYfJYjBJOPWT0YXXDcupk0sRCAGuzK1zBljcP3JlEPQGGwgiQR6r61DCkCqMFhAEoNdgj4DwypTjUEOkvnnw%2BnS%2BQRLBUKCQQqSwTCJ%2FiHKO4JUAkKGQQYCxcj8zgsIQopBAkKB4Z%2BWEThlyDGqQSAYvu%2Fg47WpPWZEDmHBqAJBYED3MSkQNgw0SCOMGEjPiYECaYjBeYRIA9IYI3eE2PvpkwtekD9gWMyrzCaFWSXGnFhSWUDYz1wJMDrA3gVd1B0GA7iZOz8GAuTcGECQ82MAQHRgFIIcCmONAYtRAHI4jHDFAGHsgBwSw5ZAIUDkYfibZcVIgMjE8DfLihEBkYuBBAFhBCCyMRAgYIwNiHwM175VQuLr4gB4bgAAAABJRU5ErkJggg%3D%3D"></a>
  <a href="https://dotnet.microsoft.com/download/dotnet/8.0"><img alt="Plugin: .NET 8" src="https://img.shields.io/badge/plugin-.NET%208-512BD4?style=plastic&logo=dotnet&logoColor=white"></a>
  <a href="https://www.python.org/downloads/"><img alt="Trainer: Python 3.12+" src="https://img.shields.io/badge/trainer-Python%203.12%2B-3776AB?style=plastic&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-PPO-EE4C2C?style=plastic&logo=pytorch&logoColor=white"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-the-bot-drives">How the bot drives</a> ·
  <a href="GradientDriftPlugin/SETTINGS.md">Settings</a> ·
  <a href="setup/README.md">Setup program</a> ·
  <a href="drift_rl/README.md">Training</a> ·
  <a href="CREDITS.md">Credits</a>
</p>

A server plugin takes over the server's AI cars and drives them sideways: linked drifts, and tandem
trains where bots chase players and each other, on the server's own track. The driving comes from a small neural network
trained offline, by reinforcement learning, in a simulator of the plugin's own vehicle model. It
ships as one `policy.json`; nothing learns on the live server.

> [!NOTE]
> **Vibe coded.** Practically all of this project was written by Claude, Anthropic's AI assistant,
> working with the owner in conversation: the plugin, the simulator and trainer, the setup program,
> the tests and these docs. The owner set the direction, ran the training and the server tests,
> and judged the driving. Every change is checked against the simulator, and the plugin has driven
> on real servers, but the code has not had a line-by-line human review. Read it, and run it on
> your own server, with that in mind.

## What it does

| | |
|---|---|
| 🏁 **Any track** | The shipped policy was trained on seven real tracks and 100 randomly generated courses, so it drives tracks it has never seen. The setup program makes a track's data from your own game install. |
| 🚗 **Any rear-drive car** | Every bot drives as a drift car (welded diff, an angle kit of at least 50°), with its weight, geometry, tyres, power and gearing from the car's own files. |
| 🚦 **Several bots** | Chasing players and each other in tandem trains (`ChasePlayers`, `ChaseBots`, on by default), or spread round the lap. |
| 🎛️ **Dials, no retraining** | How fast (a share of the track's flat-out drift speed), how good a driver (from a careful novice who still holds a drift to the best the policy can do), and how close they chase. Each bot can have its own style (`BotStyleSpread`, `BotStyles`), and with `AdaptToPlayers` the bots match the pace and chase gap of the players drifting with them. |
| 🧑 **Close to a human** | On the owner's server the bots rotate the car about as hard and as often as a human drifting the same track (yaw acceleration, direction changes, sideways g, time sliding). |

**Limits:** tested on Windows servers only (the same plugin download should run on Linux too, as
a .NET plugin with no native code, but nobody has tried it yet); rear-drive cars only;
no tracks whose model files are protected (DRM, which Custom Shaders Patch decodes as the game
loads the track: setup cannot read their ground or walls and turns them down);
and bots feel other cars only with `CarContactPush` on (off by default): then a bot that touches a
player or another bot is pushed by its share of the hit, by the two cars' masses; with it off, a bot
drives through a car it hits as if nothing were there.

## Quick start

**You need** a server running AssettoServer v0.0.54, and a PC with the game installed to run the
setup program on (setup reads the track and the cars from the game's files). They don't have to be
the same machine:

- **Your own server** (on the PC with the game, or in a server folder that PC can open): setup
  makes the changes in the server folder itself.
- **A hosted server** (a hosting plan, or any machine without the game): download the server's
  `cfg` folder to your PC, and setup exports the files to upload through your host's file manager
  or FTP, with the config edits written out too. See
  [A server on a hosting plan](setup/README.md#a-server-on-a-hosting-plan-or-any-machine-without-the-game).

1. **Download the release**: one zip, `gradient-drift-<version>.zip`, from the
   [latest release](https://github.com/toyowheelin/gradient-drift/releases/latest). Unzip it
   anywhere. It has the setup program for Windows (`windows/`) and for Linux (`linux/`), each with
   the plugin already in its `GradientDriftPlugin` folder, where setup finds it. Each release is
   built for one AssettoServer version, named in its title and README; on another version the
   plugin says so in the server log at start. To build the
   plugin yourself instead, see [Building the plugin](#building-the-plugin).
2. **Run the setup program** ([`setup/`](setup/README.md)) on the PC with the game. Point it at the
   game install and the server folder (for a hosted server, the folder you downloaded its `cfg`
   into), then choose the bot car, how many bots and pace. It shows every change before and after.
   It copies in the plugin, writes each bot car's numbers, adds the bot entries, sets the AI
   settings AssettoServer needs, and makes the track's data if it's missing.
   - **Your own server:** it makes the changes on a yes, keeps the originals, and can undo.
   - **A hosted server:** click **Export for a hosted server...** on the last page (or run
     `export` from the command line). Stop the server and upload everything in the export's
     `upload` folder into the server's folder, keeping the folders. For the three config files you
     can instead make the edits `HOW_TO_INSTALL.txt` lists by hand in your host's panel. Setup
     can't read a hosted server's AssettoServer version, so check your host runs v0.0.54.
3. **Start the server** and drive. The plugin's log lines start `GradientDrift:`: one when the
   policy loads, then nothing while the bots drive well. It warns only when something is wrong:
   why, if it stands aside and leaves the stock AI driving, or which bot is stuck and where.
4. **Tune it** in the server's `extra_cfg.yml`, then restart the server.
   [`SETTINGS.md`](GradientDriftPlugin/SETTINGS.md) lists every setting; the ones you'll want are
   `BotSkill`, `BotPacePercent`, `ChasePlayers`, `ChaseBots`, `BotStyleSpread` and `AdaptToPlayers`.
   `EnableScorecard: true` writes a short report of how the bots drove each session, to tune by.
   The [Claude Code plugin](claude-plugin/README.md) can do the tuning with you.

> [!TIP]
> **Trying a policy of your own?** Copy it over
> `plugins/GradientDriftPlugin/GradientDriftPlugin.policy.json` (keep the old one) and restart. No
> rebuild is needed. A loose policy there is read instead of the one in setup's
> `GradientDriftPlugin.bundle.zip`, and the server log says so at start. Setup checks that the
> plugin can run a policy (`GradientDriftPlugin.features.json`) before it installs either.

## How the bot drives

![The policy network: 130 inputs describing how the car is moving, the racing line, its sensors, the asked pace, the car itself, the asked skill, the car ahead and the last few ticks are scaled, pass through three layers of 256 units, and come out as steering, pedal and brake, which the plugin turns into the car's next position every server tick.](docs/policy_network.svg)

Every server tick (18 a second on a typical server), for every bot it drives, the plugin does the
same four things:

1. **Describe the car's situation as 130 numbers.**
   - how the car is moving: speed, sideways speed, how fast it is rotating, its slip angle;
   - the racing line: where it is and how it bends over the next stretch;
   - its sensors: what 15 rays fanned out ahead and 6 short rays fixed to the body hit (the walls
     the game collides with, and the course's tire stacks), and how much room there is to the
     course's edges;
   - what it did last tick, the pace it has been asked to drive and how far over that it has been
     lately;
   - what car it is (weight, wheelbase, grip, power), and how good a driver it has been asked to be;
   - the car ahead, when it is following one in tandem: the gap, closing speed, where that car is
     on the line and what it is doing with its wheel and pedals;
   - a short memory: nine of these numbers (its speeds, rotation, slip, last actions and how hard
     each axle was working) as they were over the last few ticks.
2. **Run them through the network.** Each number is scaled the way it was in training, then
   multiplied through three layers of 256 units (about 166,000 numbers in all, the whole content of
   `policy.json`).
3. **Read three numbers out:**

   | output | what it does |
   |---|---|
   | steering | how far to turn the wheel |
   | pedal | throttle above zero, all four brakes below zero (off in the shipped policy, which brakes with the front brake only: `FrontBrakeOnly`) |
   | front brake | 0 to 1, a separate front-axle brake applied on top of whatever the pedal is doing |

   The network decides both pedals each tick, so it can use them together: throttle to keep the
   rear spinning and the angle held, front brake to scrub off speed (left-foot braking, the way a
   drifter keeps a gap without straightening up). On a server run the front brake was in use about
   a fifth of the time, almost always with the throttle on.
4. **Move the car itself.** The plugin's own vehicle model turns those into tire forces (each
   axle's grip from the car's tyre file, less as the tyres carry more load), a slide, wall contact
   and ride height. The server sends players the car's new position, speed and rotation, with
   wheels, steering, engine and gear to match. On the next tick the rays, the line and the speeds
   all read differently, and the loop goes round again.

**Where the driving comes from.** The network never sees a "drift" instruction. It learned what to
do in `drift_rl`'s simulator, which runs the same vehicle model and the same sensors as the
plugin, by driving hundreds of millions of ticks on real and randomly generated courses, in 28
different cars, and being rewarded for progress made sideways at a good angle, inside the course,
without hitting anything (reinforcement learning, PPO). The result is frozen into `policy.json`;
the server only runs it.

The plugin also runs older, smaller policies (46 to 94 inputs). The parts of the vehicle
model a policy was trained with are listed in its file and the plugin turns on exactly those, so
each policy drives in the car it learned in.

## What is here

| folder | what it is |
|---|---|
| [`GradientDriftPlugin/`](GradientDriftPlugin/) | The server plugin (C#). `GradientDriftPlugin.policy.json` is the trained policy it runs; [`SETTINGS.md`](GradientDriftPlugin/SETTINGS.md) lists its settings; [`HISTORY.md`](GradientDriftPlugin/HISTORY.md) is how each part of it came about, and the server tests. |
| [`setup/`](setup/README.md) | The setup program for server operators: reads a server, says what it needs for drift bots, and makes those changes on a yes, with backups and an undo. |
| [`drift_rl/`](drift_rl/README.md) | The simulator, trainer and evaluation tools (Python), for training a policy of your own. |
| [`claude-plugin/`](claude-plugin/README.md) | A Claude Code plugin for server owners: a skill that lets Claude tune the bots to your server, reading the plugin's log and scorecards and changing settings one at a time with backups. |
| [`tests/`](tests/README.md) | The regression tests CI runs on every change: golden physics traces, the C# plugin against the Python policy, short drive checks and setup on a made-up server. |
| `tools/` | `track_triangles.py` writes a track layout's geometry as two triangle files (what its model shows, and what the game collides with) and each colliding surface's grip, using `kn5_reader.py` from KN5 Blender Tools. `centre_line.py` makes an AI line down the middle of a track from its edge files. `ai_car_monitor.py` is an in-game app that logs what a client sees of every car on the server. `plugin_features.py` writes `GradientDriftPlugin.features.json` from the plugin's source, and `ship_policy.py` puts a policy in place for a release. |

## What the plugin does on a server

- **No ports of its own; with drift scoring on, a few read-only addresses and chat commands.** It
  reads the other cars' positions and speeds to place the bots, chase or wait for a car, (with
  `AdaptToPlayers`) match the players' pace and gap, and (with `ZoneScoring`, on by default) judge
  players' drifting. Scoring adds three GET addresses on AssettoServer's own HTTP port, which the
  HUD reads: `/gradientdrift/zones/<car>` (that car's live points), `/gradientdrift/board` and the
  board as a web page at `/gradientdrift/leaderboard` (names encoded; the page runs and loads
  nothing). It also adds the chat commands `/zones` and `/top`. None of them takes any input beyond
  a car number or changes anything. `ZoneScoring: false` turns them all off.
- **What it sends players.** Besides the bots' cars and their entry-list names, players get a line
  in the welcome message on joining, "Drift bots by Gradient Drift" with a link here (`JoinMessage`;
  `JoinMessage: ""` turns it off), and a chat line for each scored run (`ZoneScoreChat`). Players
  with Custom Shaders Patch also get short scripts: one paints the track's drift zones on the road
  (`ShowZones`), and one draws the drift HUD (`ZoneHud`). The scripts are plain text built at start
  (the zones' corner points, and lines that draw with CSP's track paint and UI). AssettoServer
  serves them from its own scripts address, as it does for its other client scripts. The paint
  script reads nothing and sends nothing back. The HUD only reads the addresses above.
- **Drift records, on by default with scoring**: each scored run goes into a SQLite file in the
  server folder (`ZoneRecordsFile`, `gradientdrift_zone_records.db`): time, track, the player's name
  and Steam ID, car, score and its parts. It feeds the board, stays on your server, and is never
  sent anywhere. Delete the file with the server stopped to clear the board; `ZoneScoring: false` stops it.
- **Otherwise reads numbers, writes nothing** unless you turn it on (below). At start it reads its policy,
  the bot cars' files and the track's data, all from one file setup writes
  (`plugins/GradientDriftPlugin/GradientDriftPlugin.bundle.zip`), or loose
  (`plugins/GradientDriftPlugin/GradientDriftPlugin.policy.json`, `plugins/GradientDriftPlugin/cars/`,
  the track's data files in the server folder; a loose file is read instead of the bundle's copy):
  numbers only, nothing in them is run. What it has to say goes to the server's own log and console.
- **Telemetry, only if you turn it on** (`EnableTelemetryLogging: true`, for checking a policy with
  `drift_rl`): `drift_physics_telemetry_<date>_<time>.csv` in the server folder, every AI car every
  tick and, with `LogHumanCarForCalibration: true`, the first connected player's car (position,
  speed, inputs; no name, IP or Steam ID). About 110 MB an hour with six bots; it stops at
  `TelemetryMaxFileMegabytes` (500) or the first failed write.
- **A scorecard, only if you turn it on** (`EnableScorecard: true`): a few KB a session in
  `gradientdrift_scorecards/` in the server folder, a table and a .json of how the bots drove (slide,
  walls, car hits, chase gap, how often the stock AI had them, stuck bots). Counts only, no player
  names; touches with a player are counted under the bot.
- **It cannot take the server down.** AssettoServer shuts down after 10 tick updates in a row
  fail; an error inside the plugin is logged instead, and after 5 in a row it stops and the stock
  AI drives.

## No game content included

No car or track data ships with this project, now or in a release. Each user generates what the
tools need from the game content installed on their own machine.

- **Track files** (`drift_rl/tracks/<track>/`): the track's AI line and mesh, the course edges,
  markers and zones generated from them, and pictures of them. The trainer needs them;
  [`drift_rl/README.md`](drift_rl/README.md#1-track-data) says how to make them from your own copy
  of the track.
- **Car files**, and the numbers extracted from them: `drift_rl/cars/` has the extraction tools only.
- **Telemetry and recordings.**

## Building the plugin

Normally you don't need this: the [release](https://github.com/toyowheelin/gradient-drift/releases/latest)
carries the plugin built, and setup installs it for you along with everything else (the
[Quick start](#quick-start)). Build it yourself only to run changes of your own or an unreleased
version. It compiles against AssettoServer's own source, so you need that too.

> [!NOTE]
> Building gives you the plugin only. A server needs more than the plugin for bots to drive: AI
> entries in `entry_list.ini`, AI turned on with the `AiParams` bots need, each bot car's file,
> and the track's data, all made by the [setup program](setup/README.md). On a new server, build,
> then run setup pointed at your build (step 4) and it does all of that. Copying the files in by
> hand only fits a server setup has already prepared, when you are updating its plugin.

**You need:** the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) and
[git](https://git-scm.com/).

**1. Get AssettoServer's source at v0.0.54**, the version the plugin is built and tested against.
Clone it whole, not with `--depth 1`, because its build reads its version from git:

```
git clone --branch v0.0.54 https://github.com/compujuckel/AssettoServer.git
```

**2. Put the plugin inside it.** Copy this repository's `GradientDriftPlugin/` folder into the
AssettoServer folder, next to its own projects. The plugin's `.csproj` finds them by relative path,
so it has to sit exactly here:

```
AssettoServer/
├── AssettoServer/              (AssettoServer's own projects)
├── AssettoServer.Shared/
└── GradientDriftPlugin/        <- this repository's folder
    ├── GradientDriftPlugin.csproj
    ├── GradientDriftPlugin.policy.json
    └── *.cs
```

The plugin is one folder of this repository, so git can't clone it into that spot on its own. To
build from a clone you can update with `git pull`, clone this repository somewhere else and link its
folder into the AssettoServer tree instead of copying it. On Windows (PowerShell, from the folder
holding both clones):

```
git clone https://github.com/toyowheelin/gradient-drift.git
New-Item -ItemType Junction -Path AssettoServer\GradientDriftPlugin -Target (Resolve-Path gradient-drift\GradientDriftPlugin)
```

On Linux: `ln -s ../gradient-drift/GradientDriftPlugin AssettoServer/GradientDriftPlugin`.

**3. Build it:**

```
cd AssettoServer/GradientDriftPlugin
dotnet build
```

The first build also builds the AssettoServer projects it references, so it takes a minute. The
result is in `bin/Debug/net8.0/`. The files the server needs are:

| file | what it is |
|---|---|
| `GradientDriftPlugin.dll` | the plugin |
| `GradientDriftPlugin.deps.json` | what it depends on, for .NET to load it |
| `GradientDriftPlugin.policy.json` | the trained policy, copied in from the source folder |
| `GradientDriftPlugin.features.json` | what policies this build can run, for setup's check (the plugin itself does not read it) |

**4. Install it on the server.** Run the [setup program](setup/README.md) with its `--plugin` option
(or the window's **Plugin build** field) pointed at the `bin/Debug/net8.0/` folder. On a new server
this is the way to do it: setup installs your build along with everything else the bots need (see
the note above), and on a server it already set up it replaces the plugin with your build.

To update the plugin by hand on a server setup has already prepared, stop the server first
(Windows keeps the files locked while it runs), then copy the files into the server's
`plugins/GradientDriftPlugin/` folder. On Windows, from the AssettoServer folder:

```
copy GradientDriftPlugin\bin\Debug\net8.0\GradientDriftPlugin.* "D:\servers\my server\plugins\GradientDriftPlugin\"
```

Setup has already turned the plugin on in `cfg/extra_cfg.yml` on such a server
(`EnablePlugins: [GradientDriftPlugin]` and a `--- !GradientDriftConfiguration` block; see
[`SETTINGS.md`](GradientDriftPlugin/SETTINGS.md)). Setting up a server entirely by hand, without
setup, is not covered here: [What it changes](setup/README.md#what-it-changes) lists every change
setup makes, if you want to follow it.

**5. Check it loaded.** Start the server and look for `GradientDrift: policy loaded from ...` in its
log. If the plugin stands aside, a warning there says why, and the stock AI drives. After that the
plugin stays quiet while the bots drive: it writes to the log only when something is wrong, such as a
bot stuck for 30 seconds. `VerboseLogging: true` in the plugin's settings brings back the full
output: what it loaded, each bot car, spawns and timing.

> [!TIP]
> **Updating.** After pulling a newer version of this repository, copy its `GradientDriftPlugin/`
> folder over the one inside AssettoServer again, rebuild, and copy the files in again. A new
> policy on its own needs no rebuild: copying the `.policy.json` is enough. Keep the build folder's
> copy of the policy in step with the server's, or the next copy from the build puts the old policy
> back.

`.github/workflows/build-plugin.yml` runs these same steps on every change to the plugin, so a
build that breaks shows up there first.

A release ships one policy, the same on every server. `python tools/ship_policy.py <policy.json>
--release <tag>` puts a policy in place as `GradientDriftPlugin/GradientDriftPlugin.policy.json`
after checking the plugin can run it, and adds it to setup's list of shipped policies; a `v*` tag
then builds the one release zip (`release.yml`), which refuses a policy that is not on that list or
does not fit the plugin. The release notes are the tag's section of [CHANGELOG.md](CHANGELOG.md).

## Training your own policy

Optional: the plugin ships with a policy. The full guide is [`drift_rl/README.md`](drift_rl/README.md);
in short, from `drift_rl/`:

```
# the track's data, once
python setup_track.py <game>/content/tracks/<track>

# train, starting from the shipped policy
python train_ppo_gpu.py --sim cpu --init-policy ../GradientDriftPlugin/GradientDriftPlugin.policy.json --hidden 256 --layers 3 --history-frames 4 --history-every 2 --track <track> --car-pool cars/car_pool.json --skill 0 1 --reward weave_penalty=5 --reward swing_penalty=4 --steps 30000000 --name <run>

# how it drives (list more policies to compare)
python drive_report.py runs/<run>/policy.json --track <track>
```

This keeps the shipped policy's network (`--hidden`, `--layers`, `--history-*`) but not its chase
training: a run without `--chase` makes a policy blind to the car ahead, whose bots follow a car
by pace alone. To make the bots drive one track
better, train the shipped policy further on that track, chasing included, and put the result on
that track's server only: the steps and what to check are in
[Polishing the shipped policy for one track](drift_rl/README.md#polishing-the-shipped-policy-for-one-track).

<details>
<summary><h2>The name</h2></summary>

**gradient-drift** is a pun on gradient descent, the method that trains the network: training
works out which way each of the network's ~166,000 numbers should move to do better and nudges
them that way, millions of times (PPO, the algorithm used, is a *policy-gradient* method). Claude
suggested it: asked for a name with a bit of flavour that kept the machine-learning angle, it gave
a short list and picked this one as its favourite, and the owner chose it on 2026-10-07. It reads
as a drift name to car people and as a machine-learning name to machine-learning people, which is
who this project is for.

The server plugin follows AssettoServer's own naming, so it reads like the plugins that ship with
it (`RandomWeatherPlugin` is configured by a `--- !RandomWeatherConfiguration` block):

```yaml
EnablePlugins: [GradientDriftPlugin]       # loads plugins/GradientDriftPlugin/GradientDriftPlugin.dll
--- !GradientDriftConfiguration
EnableLearnedPolicy: true
```

AssettoServer finds a plugin by its folder name, which has to match its DLL, and reads the
settings block by the settings class's name. All of these are in every server's config, so the
name was settled before the first release (the plugin was called DriftAiPlugin while it was being
built).

</details>

## Research behind it

> *"The project sits right at the intersection of Elite Engineering and Applied Doctoral Research.
> While the core machine learning algorithm (PPO) is standard industry tech, the synthesis of
> modern literature (Residual RL, Asymmetric Actor-Critics, GT Sophy sportsmanship bounds, TRI
> tandem physics) makes this project significantly more sophisticated than standard software
> projects. It is essentially an open-source implementation of multi-million dollar robotics
> research conducted by Stanford, Toyota, and Sony AI."*
> — Gemini

Every paper it names is linked in [CREDITS.md](CREDITS.md). The current policy is trained with
PPO and drives in tandem: an asymmetric (privileged) critic, chase practice in hard situations as
GT Sophy trained, and a gap that opens through the lead's transitions as the TRI / Stanford tandem
cars do. Residual RL drove the first policies. The project uses those groups' published
methods and is not affiliated with them.

## Licence

Copyright (C) 2026 the gradient-drift contributors.

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See [`LICENSE`](LICENSE) for the full text.

`tools/kn5_reader.py` is from KN5 Blender Tools and stays under its own GPL-3.0-or-later licence
(see the notice in that file). What this project is built on and owes credit to:
[`CREDITS.md`](CREDITS.md).
