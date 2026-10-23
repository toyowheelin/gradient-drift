# Gradient Drift setup

Walks a server operator through getting drift bots onto an AssettoServer server. It finds the
server and the game install, checks the track and the car, works out every change the server
needs, and makes them with backups. It's a Windows window (Qt) over a shared core, with the
command line as the fallback on Windows and the only form on Linux.

**Contents:** [What it does](#what-it-does) · [Download and run](#download-and-run) ·
[Using it](#using-it) · [A server on a hosting plan](#a-server-on-a-hosting-plan-or-any-machine-without-the-game) ·
[What it changes](#what-it-changes) · [Running from source](#running-from-source) ·
[Building the package](#building-the-package) · [How it's built](#how-its-built)

## What it does

- **Checks a server and makes the changes for drift bots**, with every changed file kept and an undo.
- **Knows AssettoServer from the game's own dedicated server.** It looks inside the server program,
  since the file name proves nothing (AssettoServer is often renamed `acServer.exe`), and reads
  AssettoServer's version.
- **Makes a track's data when it's missing**, from the game install. It takes a few minutes: the
  track's geometry, road mesh, AI line, course and zones, the same as `drift_rl/setup_track.py`.
  For a track with no AI line of its own it makes one from a recorded lap
  (`ai/fast_lane.ai.candidate`).
- **Refuses a track whose data looks wrong**, before it writes anything to the server: speed
  targets the car cannot drive (more than it can corner at, or a kink in the AI line), the AI line
  running into a wall, or a wall too low for the policy's sensors to feel (`drift_rl/check_track.py`).
  It says where on the lap, and what to do about it.
- **Updates an installed plugin.** Run it again with a newer download and it replaces the plugin
  on the server, and the policy with it (an older plugin turns down a newer policy's settings and
  leaves the cars to the stock AI). The old files go to the backup like any other change. Stop the
  server first: setup will not replace a plugin AssettoServer has loaded. A policy beside the same
  plugin that differs from the download's is kept, since it may have been polished for the track;
  setup says so and where the download's is.
- **Exports instead, for a server it cannot reach** (a hosting plan): the files to upload and the
  config edits ([below](#a-server-on-a-hosting-plan-or-any-machine-without-the-game)).

It does not train a policy for a track. The trainer can polish the shipped policy for one track:
[Polishing the shipped policy for one track](../drift_rl/README.md#polishing-the-shipped-policy-for-one-track).

## Download and run

Download the zip for your system, unzip it anywhere, and run what's inside. It carries its own
Python and Qt; nothing to install.

### Windows

`gradient-drift-setup-windows.zip`, about 140 MB (most of it is the track tools).

- **The window:** double-click `gradient-drift-setup.exe`.
- **The command line:** run the same program from a Command Prompt or PowerShell
  (`gradient-drift-setup.exe check ...`, as [below](#using-it)). With no arguments it opens the
  window there too.

> [!WARNING]
> The program isn't code-signed yet, so the first time Windows shows "Windows protected your PC".
> Click **More info**, then **Run anyway**. If Windows marks the downloaded zip as blocked,
> right-click it > Properties > **Unblock** before you unzip it.

### Linux

`gradient-drift-setup-linux.zip`, about 145 MB, command line only:

```
./gradient-drift-setup/gradient-drift-setup check ...
```

It's built on Debian 11, so it runs on that and anything newer (glibc 2.31+). Not yet tried on a
real Linux server. It finds the game's files whatever their capitals (a Steam / Proton install
keeps a mod's `AI/` or `Track.KN5` as made on Windows); the server's own folders must still match
its config exactly, as AssettoServer itself needs.

### What's in the zip

Beside the program are its licence (`LICENSE`, AGPL-3.0-or-later), `CREDITS.md`, and
`THIRD_PARTY_NOTICES.txt` with `licenses/`: what else is inside (Python, Qt for Python, numpy,
scipy, pandas, numba and the others), each one's licence, and its full text.

## Using it

In the window, the pages walk you through it: the game install, the server, the track check, the
bots and their cars, and the change list. The bots' page has a row per bot slot (the server's
existing AI entries first, then the new ones) with its car and skin, and "Every bot in" to put them
all in one car. Under "Track files" on the last page, pick the folder setup_track.py writes into,
with a folder per track (`tracks\` beside the packaged program, or `drift_rl\tracks`); setup
turns down a track's own folder in the game or server content.

On the command line (the packaged program is `gradient-drift-setup.exe`; from source it is
`py -m gradientdrift_setup`):

```
gradient-drift-setup.exe check "D:\servers\my server" --game "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
gradient-drift-setup.exe check "D:\servers\my server" --game "..." --details      # each change as before / after
gradient-drift-setup.exe apply "D:\servers\my server" --game "..." --bots 4       # shows the changes, asks, makes them
gradient-drift-setup.exe undo  "D:\servers\my server"                             # puts back what the last apply changed
```

Several server folders can be given at once (not with `export`).

| option | what it sets |
|---|---|
| `--bots N` | bots in all, existing ones included (setup never takes a bot away) |
| `--car MODEL`, `--skin NAME` | every bot's car and skin, the existing AI entries put in it too. Without `--car` an existing bot keeps its car and a new one takes an existing bot's, else the car the shipped policy was trained in, else the server's first car |
| `--car A,B,...`, `--skin X,Y,...` | a car (and skin) per bot slot, the existing AI entries first in the entry list's order; an empty one keeps that bot's car; bots past the list take its last car |
| `--train-size N` | run the bots as tandem trains of N: the first car of each drives, the others follow it door to door (1 = every bot on its own) |
| `--pace PERCENT` | how fast they drive, as a share of the track's reference speed round the lap (a policy trained with paces; lower is calmer) |
| `--plugin FOLDER` | the plugin's build (default: `plugin/` beside the packaged program, else the repository's build folder) |
| `--track-data FOLDER` | the tracks' files, a folder per track (default: `tracks/` beside it, else the repository's `drift_rl/tracks`) |

## A server on a hosting plan (or any machine without the game)

Setup needs the game's files, so it runs on your own PC; `export` then gives you the files to put on
the server yourself, through your host's file manager or FTP:

1. On your host, start the server once (so AssettoServer writes `cfg/extra_cfg.yml`), stop it, and
   download its `cfg` folder (`server_cfg.ini`, `entry_list.ini`, `extra_cfg.yml`) into an empty
   folder on your PC, e.g. `D:\my host\cfg\`.
2. In the window, pick `D:\my host` as the server, choose the car, bots and pace as usual, and on the
   last page click **Export for a hosted server...** and pick an empty folder. From the command line:

   ```
   gradient-drift-setup.exe export "D:\my host" --out "D:\my host export" --game "C:\...\assettocorsa" --bots 4 --pace 90
   ```

   Making the track's data takes a few minutes, as with `apply`. The folder
   you downloaded is not changed.
3. The export folder has `upload\`, every file the server needs added or changed at its place in
   the server folder (`plugins\GradientDriftPlugin\...`, the track's data files beside `cfg`, the
   track's AI line under `content\tracks\`, the edited `cfg\` files), and `HOW_TO_INSTALL.txt`, which
   lists each file and, for the three config files, the same edits as lines to make by hand in your
   host's panel instead of uploading them.
4. Stop the server, upload everything in `upload\` into the server's folder keeping the folders,
   and start it. Keep the file names exactly as they are: most hosts run Linux, where
   capitals matter. The plugin has been run on Windows servers only so far.

> [!CAUTION]
> Setup cannot tell the server's AssettoServer version from the config files alone: the plugin is
> built for v0.0.54, so check your host runs that. There is no undo for an export; the files you
> downloaded are your backup.

## What it changes

Each change is shown before and after.

**`entry_list.ini`:** a `[CAR_n]` per new bot, with `AI=fixed` (a slot only a bot drives; with
`AI=auto` a player who picks that car takes the bot's slot, and the server has a bot fewer), and
`MODEL=`/`SKIN=` of an existing AI entry whose slot was given another car. Player slots are never
changed, and no bot is taken away.

**`server_cfg.ini`:** the bots' cars in `CARS=`, and `MAX_CLIENTS` raised to cover the entries.

**`extra_cfg.yml`:**

- `EnableAi`, and the plugin in `EnablePlugins`;
- the `AiParams` bots need. AssettoServer only spawns AI for a player near the AI line who is not
  idle, and a drift player is often well off it, or parked watching, so setup sets
  `MaxPlayerCount` (the player slots), `MaxPlayerDistanceToAiSplineMeters` at least 100 and
  `PlayerAfkTimeoutSeconds` at least 600. With several bots it also sets `PlayerRadiusMeters` to at
  least 5/8 of a lap (AssettoServer takes away AI cars further than that from every player);
- the plugin's own section with `EnableLearnedPolicy: true`, and `TrainSize` and `BotPacePercent`
  when given.

**Made first when they are not there:** the track's data, into the track files folder (`tracks/`
beside the packaged program, else the repository's `drift_rl/tracks`), never into the server; then,
for a policy with paces, the track's pace reference (its reference speed at each point, what a pace
of 100% asks for), worked out from the track's shape (`drift_rl/pace_model.py`) exactly as training
makes it, so a track the policy never trained on needs no training run. A server's older pace
reference (measured by a policy driving the track, before 2026-10-18) is replaced the same way.

**Checked before anything is written,** whenever the server gets a track's data: what the speed
targets ask of the car in each corner, and the walls beside the course against the height of the
policy's wall sensors (`drift_rl/check_track.py`). The speed targets checked are the ones the server
will run: its own pace reference unless setup is replacing it. A problem stops the run with nothing
changed.

**Copied in:**

- the plugin's files;
- the track's data files (also when the server's copy differs from the one in the track files,
  e.g. a pace reference made by newer training; a copy that differs only in line endings, as a
  copy made on Windows can, counts as the same);
- the track's AI line: the one the track's data was made for, saved beside it, which is the game's
  or a rebuilt one. A server line that differs from it is replaced, because the course edges and
  pace reference go point by point along it.

### Backups and undo

- Only the lines it means to change are touched (comments, order and line endings stay), and each
  edited file is read back before anything is written.
- The originals go to `<server>/gradient-drift-setup-backups/<date_time>/`. Undo restores them and
  moves the files the run added into the same folder, so nothing is deleted.
- It will not apply anything while a check fails (the game's own server, no plugin build, no track
  data, a policy trained for another track).

## Running from source

```
py -m pip install ruamel.yaml PySide6-Essentials
cd setup
py -m gradientdrift_setup                    # the walk-through, in a window
py -m gradientdrift_setup check ...          # or any command above
```

## Building the package

The zips come from the "build setup" workflow (`.github/workflows/build-setup.yml`): run it from
the Actions tab, or push a `setup-v*` tag to get a draft release with both attached. To build one
yourself, from the repo root:

```
pip install -r setup/requirements-build.txt
pyinstaller --noconfirm setup/gradient-drift-setup.spec
```

Set `GRADIENTDRIFT_GUI=0` for the command-line-only build. The result goes in
`dist/gradient-drift-setup/`.

## How it's built

| file | what it does |
|---|---|
| `gradientdrift_setup/server.py` | Reads a server folder: `server_cfg.ini`, `entry_list.ini`, `extra_cfg.yml` (all its documents, comments kept), the plugin folder; whether AssettoServer is running from it. |
| `gradientdrift_setup/game.py` | What the game install has for the layout (its model files, its AI line and whether it has points) and for a car. |
| `gradientdrift_setup/identify.py` | AssettoServer or the game's own server, and AssettoServer's version. |
| `gradientdrift_setup/plan.py` | The change list for the bots (each change carrying its edit), and what setup cannot fix by editing files. |
| `gradientdrift_setup/edits.py` | Line-preserving edits to `.ini` and `extra_cfg.yml`. |
| `gradientdrift_setup/apply.py` | Makes a plan's changes with backups; undo. |
| `gradientdrift_setup/export.py` | For a hosted server: makes the changes into a copy of its downloaded `cfg/`, and writes the files to upload and `HOW_TO_INSTALL.txt`. |
| `gradientdrift_setup/cli.py` | The command line; with no arguments it opens the window, or falls back to the command line where it cannot (no Qt, or Linux without a desktop). |
| `gradientdrift_setup/gui.py` | The window (Qt, PySide6): game install, server, track check, the bots' cars slot by slot, the change list. |
| `gradientdrift_setup/console.py` | Windows: tells a double-click from a terminal start, and hides the console for the window. |
| `gradientdrift_setup/trackdata.py` | `gradient-drift-setup selftest` (not in the help): the packaged program's own track tools make a generated course's data. The build workflow runs it on Windows and Linux. |
| `gradientdrift_setup/find.py` | Finds the game (Steam libraries, the usual folders), the server folders in or beside it, and the track files folder. |
| `gradientdrift_setup/acini.py` | Assetto Corsa .ini reading the way the game does. |
| `gradientdrift_setup/carfile.py` | For a policy that drives any car: each bot car's numbers, from the game's car files with the trainer's own tools (`drift_rl/cars`), written to `plugins/GradientDriftPlugin/cars/<model>.json`, with the car's pace scale on the server's track (`PACE_SCALE`: its speed there against the reference car, from `drift_rl/pace_model.py`; 1 when the track's data is missing). |
| `gradient-drift-setup.spec`, `launcher.py`, `requirements-build.txt` | The packaged build (PyInstaller, one folder). |

Confirmed on the owner's servers (2026-10-12): `AiParams.MaxPlayerCount` equals the number of player
entries; AssettoServer reads the AI line from the server's own `content/tracks/<track>[/<layout>]/ai/fast_lane.ai`;
an `AI=auto` bot entry gives its slot to a player who picks that car, so setup writes `AI=fixed`.
