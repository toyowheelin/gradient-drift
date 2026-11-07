# Changelog

What changed in each release, newest first. A `v*` tag's release notes are its section here:
the release workflow copies the section whose heading is the tag into the draft release, and
refuses the tag if there is none. Add to `## Unreleased` as changes merge; rename it to the tag
when releasing.

## Unreleased

- Setup puts the AI line where AssettoServer 0.0.54 reads it on a track with layouts:
  `content/tracks/<track>/ai/fast_lane.ai`. It used to go in the layout's own `ai` folder, which
  0.0.54 never reads, so the server would not start ("No ai folder found").
- Setup refuses a track whose model files are protected (every vertex moved, which the game's
  Content Manager / Custom Shaders Patch undo on load): its ground came out as a crumpled sheet with
  layers metres apart, and the bots climbed it into the air.

## v0.3.1

Bots that feel other cars, fit in with the players on the server, and say hello. The policy is
the same as v0.3.0's (combo_all2), so the bots drive as before until you turn the new settings on.
The only change you will see without touching a setting is the join message.

### Upgrading

- Run the new setup on each server (`gradient-drift-setup apply <server>`, with the server
  stopped), or `export` again for a hosted server. It replaces the plugin and now also writes a
  car file for every car model in the entry list, players' cars included, so `CarContactPush`
  knows each car's mass. The policy is unchanged; a policy of your own is kept as before.
  `undo` puts back what it changed.
- Players now see a line about the bots when they join (see below). Set `JoinMessage: ""` in the
  plugin settings to turn it off.
- Everything else new is off by default or keeps the v0.3.0 behaviour at its default.

### Fixed

- Setup stopped with "the speed targets have N points and the AI line M" when it rebuilt a
  track's data (for example to make the surface grip file a v0.3.0 policy needs) and the rebuilt AI
  line came out a point longer or shorter than before. It now makes the pace reference again for
  the new line and installs both together; the old one is kept beside it as `.before_new_line`.
- A bot slowed almost to a stop (by a push, say) could twist violently back and forth for a moment:
  below walking-ish speeds the car model let the front wheels swing the car round as if it were
  still moving, and the policy sawing at the wheel shook it. The steering's effect now fades with
  speed below 3 m/s (11 km/h), in the plugin and the trainer's simulator alike; nothing changes
  above that.

### Bots feel other cars

- **`CarContactPush`** (off by default). Before, a bot drove through a car-on-car hit as if
  nothing were there. Now a bot that touches a player or another bot takes its share of the
  collision: it is pushed back, or along if hit from behind, and an off-centre hit turns it. Two
  bots that touch are both pushed. The hit is shared by the two cars' real masses and yaw
  inertia, from their car files, so a light car is pushed more than a heavy one. The player's own
  game already gives the player their share. A push held on keeps moving the bot: while the cars
  stay in contact it is kept moving away at least as fast as the player drives into it. Run on the
  project's own three bot servers with the shipped policy before this release.
- **Bot aggression now sets how close a chaser follows.** With `BotStyles` or `BotStyleSpread`,
  the pushiest bots (aggression 1) drop `ChaseTimeGapSeconds` and hold just their chase gap at any
  speed, door to door; the calmest (aggression 0) keep twice it. At the default (no spread, 0.5)
  nothing changes. The time gap is what holds a chaser about 7 m back at drifting speed, so this is
  the dial for closer tandems.

### Bots fit in with the players

- **`AdaptToPlayers`** (off by default). While players drift on the server, each bot's pace and
  chase gap move toward theirs: pace measured against the track's pace reference, the gap from
  players following a car closely. Only players drifting on the line count (sliding 12 - 75
  degrees, within 5 m of the AI line, its way); spinning, crashing or cruising players are left
  out. It stays inside the policy's trained range and goes back to the settings when the server
  is empty. Nothing is learned or kept. `AdaptToPlayersStrength`, `...LearnSeconds`,
  `...ForgetSeconds` and `...AverageSeconds` tune it; with `VerboseLogging` the log shows what it
  measured once a minute. See "Fitting in with the players" in
  [SETTINGS.md](https://github.com/toyowheelin/gradient-drift/blob/main/GradientDriftPlugin/SETTINGS.md).

### Players see who drives the bots

- **`JoinMessage`** (on by default). Players joining get "Drift bots by Gradient Drift" and a
  link to the project in their welcome message, just above AssettoServer's own "Powered by
  AssettoServer" line. Set it to any text of your own, or to `""` to turn it off.

### Tune the bots with Claude

- A **Claude Code plugin** for server owners (`claude-plugin/`, skill `tune-drift-bots`). Claude
  checks the plugin's log, reads the bots' scorecards and changes `extra_cfg.yml` one setting at
  a time, with a backup each time; you judge the driving. Install it in Claude Code with
  `/plugin marketplace add toyowheelin/gradient-drift` then
  `/plugin install gradient-drift-tuning@gradient-drift`, or upload the skill folder on
  claude.ai. See
  [claude-plugin/README.md](https://github.com/toyowheelin/gradient-drift/blob/main/claude-plugin/README.md).
  Not made or endorsed by Anthropic.

### For people training their own policies

- `train_ppo_gpu.py --chase-touch-push`: cars have mass in the simulator too. A touch moves both
  cars, the lead is knocked off its run and eases back, and the reward charges what a touch does
  to the lead (yaw kick past a free amount, a spun lead, sitting where the lead's rear is about to
  swing), not the touch itself. With it, the automatic curriculum closes the gap on upsets, not
  every touch. Policies trained with it expect `CarContactPush` on.
- `gap_traces.py` (new): per-tick chase traces from the simulator, each episode sorted by why
  the gap is what it is (holds, held back, pinned, overshoots, oscillates), with a plot.
- `pick_policy.py` (new) ranks candidate policies on one fixed set of scenarios;
  `chase_report.py --by-lead` splits a chase by who drove the lead; `--touch-push` prints lead
  kicks, upsets, spins and swing-zone time.
- `replay_live.py` replays every bot of a server run, each in its own car, and reads policies
  with a history block (it silently broke on v0.3.0's policy before).
- Fixed: `--init-policy` from a policy with history inputs (such as v0.3.0's) crashed.

### Docs

- The README, the trainer README and SETTINGS.md were checked against the code after v0.3.0:
  the dials, what the bots send players, how a policy without chase training follows, and how
  bots get their names. The quick start covers hosted servers (setup's `export`), and the release
  notes say so too.

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
