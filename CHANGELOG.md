# Changelog

What changed in each release, newest first. A `v*` tag's release notes are its section here:
the release workflow copies the section whose heading is the tag into the draft release, and
refuses the tag if there is none. Add to `## Unreleased` as changes merge; rename it to the tag
when releasing.

Settings named here are described in
[GradientDriftPlugin/SETTINGS.md](GradientDriftPlugin/SETTINGS.md).

## Unreleased

- Bots can feel other cars' mass (`CarContactPush`, off by default): a bot that touches a player
  or another bot takes its share of the collision, pushed back or along and turned by an
  off-centre hit; two bots that touch are both pushed. The hit is shared by the cars' real masses,
  players' cars included: setup now writes a car file for every car in the entry list.
- Bot aggression (`BotStyles`, `BotStyleSpread`) now also sets the time gap a chasing bot holds:
  the pushiest bots (aggression 1) drop `ChaseTimeGapSeconds` and follow door to door, the calmest
  keep twice it.
- Players joining a server with the bots see "Drift bots by Gradient Drift" and a link to the
  project in chat, just above AssettoServer's own "Powered by AssettoServer" line
  (`JoinMessage`; set it to `""` to turn it off, or to any text of your own).
- The bots can fit in with the players drifting with them (`AdaptToPlayers`, off by default): each
  bot's pace and chase gap move toward the players' while they are on, inside the policy's trained
  range, and go back to the settings when the server is empty. Nothing is learned or kept.
- A Claude Code plugin for server owners (`claude-plugin/`, skill `tune-drift-bots`): Claude reads
  the plugin's log and scorecards and tunes `extra_cfg.yml` one setting at a time, with a backup
  each time. See [claude-plugin/README.md](claude-plugin/README.md).
- Trainer reports for choosing and checking policies: `pick_policy.py` ranks candidate policies on
  one fixed set of scenarios, `chase_report.py --by-lead` splits a chase by who drove the lead, and
  `replay_live.py` replays every bot of a server run, each in its own car.

## v0.3.0

The largest release so far: a new policy that drives in tandem, bots that can chase you and each
other, one file per server for the bots' data, and setup that checks the policy and plugin fit
before installing them.

### Upgrading

- Run the new setup on each server (`gradient-drift-setup apply <server>`). It replaces the plugin
  and the policy together, makes each track's surface grip file, and packs the bots' data into
  one file. It also takes out `TrainSize`, now retired. Stop the server first. `undo` puts back
  what it changed.
- The bots now drive at 100% pace when `BotPacePercent` is not set. Before, an unset pace meant
  the top of the policy's range, which is 120% for the new policy.
- A policy you trained or polished yourself is kept, as before. If the new plugin cannot run it,
  setup now stops and says why, in place of the bots silently falling back to the stock AI.
  `--shipped-policy` puts the release's policy in its place.

### The new policy

- One policy for every server, trained from scratch with tandem in mind: 130 inputs (including the
  car ahead and a short memory of the last few ticks), three layers of 256.
- Trained with every technique that tested well: smoother actions, hidden tyre and sensor
  variation, practice against copies of itself and recorded human laps, a style reward from a
  human's driving, mirror symmetry, and an automatic curriculum (see
  [CREDITS.md](CREDITS.md)).
- Physics it was trained with, which the plugin now runs:
  - `RoundTireCurve`: a rounder tyre curve (brush model)
  - `RearSpinGripShare`: less rear grip while the throttle spins the tyres
  - `SurfaceGrip`: grip scaled by the surface under each axle, from the track's own surfaces
- Tested on three servers. Bots slid 86 - 87% of the time at about 35° and spun 0% of the time.
  Chasing a player, they held a median gap of about 7 m.

### Tandem

- `ChasePlayers`: a bot chases a player who pulls in just ahead of it, holding a gap with a
  car-following model and the policy's own chase inputs. `ChaseOptInMeters` and
  `ChaseOptOutMeters` set how close a player has to be to start a chase and how far ahead they
  get before it ends.
- `ChaseBots`: bots chase bots too, so drift trains form on their own. Each car has at most one
  chaser, and `ChaseTrainMax` caps a train's length.
- Both are on by default, so a server gets tandem without a settings change. Set `ChaseBots: false`
  to keep bots spaced round the lap and chasing players only, or `ChasePlayers: false` for no chasing.
- `LeadPlayers`: a bot with a player close behind waits for them.
- **Retired:** `TrainSize` and its `Tandem*` settings. Followers that replayed a lead car's path
  are gone: chasing bots drive themselves and form trains on their own. The plugin ignores these
  settings (an old config still starts), and setup takes `TrainSize` out. Setup's `--train-size`
  option and its trains question are gone.

### Bots

- Each bot can have its own style: aggression, chase gap and skill (`BotStyleSpread`,
  `BotStyleSeed`, `BotStyles`). A server full of bots no longer drives like clones.
- Spacing several bots round the lap never pushes a bot above its own pace.

### For server owners

- **One data file per server.** The policy, the bots' car files and the track's data go into
  `plugins/GradientDriftPlugin/GradientDriftPlugin.bundle.zip`. A loose file you copy in by hand
  still wins, and the server log says which loose files it is reading.
- **Setup checks the pair.** Each plugin build comes with a list of the policies it can run
  (`GradientDriftPlugin.features.json`). Setup refuses a policy the plugin would turn down.
- **Scorecard.** `EnableScorecard` logs a short report card per session: slide and spin share,
  wall and car hits, chase time and gap, and time on the stock AI. Counting stops while no player
  is on.
- **Setup works on more tracks.** Big tracks fit in memory, a track without an AI line gets one
  from its side files, and low-grip verges the game counts as track are kept off the course (Black
  Cat County's course used to run out to the walls).
- `gradient-drift-setup bundle <server>` only packs a server's loose data files.

### For developers

- Regression tests in CI: golden physics traces, the plugin's network against the trainer's, drive
  checks on generated courses, and setup on a mock server.
- `tools/plugin_features.py` writes the features file from the plugin source. CI fails while it is
  out of date.
- `tools/ship_policy.py` makes a policy the release's and records its checksum. A release must ship
  exactly one policy.
- Trainer: deeper networks, kickstarting from a teacher, mid-drift and ghost starts, chase
  episodes with recorded leads, a privileged critic, population-based training (`pbt.py`), and
  switches for each technique above. Tools for reading chase runs: `chase_report.py`,
  `telemetry_ghosts.py`, `speed_limit.py`.
