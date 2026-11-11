# Changelog

What changed in each release, newest first. A `v*` tag's release notes are its section here:
the release workflow copies the section whose heading is the tag into the draft release, and
refuses the tag if there is none. Add to `## Unreleased` as changes merge; rename it to the tag
when releasing.

## Unreleased

- Drift line: the AI line moves sideways at most 0.2 m per metre (a short zone far from the racing
  line gets part of the move instead of a kink: Spirit Peaks' zone 8 had made a 9 m corner the track
  check refused), each point's corner radius is worked out again for the moved line, and generated
  courses' lines (version -1) can be moved too. Setup works out a new pace reference when it puts a
  new AI line in the server; before, the server kept the old line's.
- The drift HUD shows speed in mph or km/h. It starts in the game's units where the player's CSP
  says which, and a right click on the panel switches it (remembered on their PC).
- Arcade drift scoring: `ZoneScoreMode: Arcade` scores players freestyle, with no zones (and no
  zone paint), so it works on any track. A chain builds points every moment the car slides, from
  angle x speed, times a combo that goes up with each transition and a bonus of up to x2 for
  sliding close behind another car. Stopping sliding for 1.5 s banks the chain, and a spin or a
  wall hit loses it. The HUD shows the chain, its multiplier and a tandem bar, with popups for
  transitions and for banked and lost chains. Chat, `/zones`, `/top` and the board page show
  chains too, and arcade chains keep their own board. `Zones` (the default) scores through the
  zones as before.
- Players' drift runs are scored through the track's zones, like a drift judge would: each zone
  gets up to 100 for line (rear within 3 m of the zone's edge while sliding), angle and speed, a
  tandem bonus for sliding close behind another car, 0 for a spin, and a run (first zone to last)
  loses 5 for each wall hit. The server judges from the car status it already receives. Players
  with Custom Shaders Patch get a HUD with live zone points, popups for each zone, spin, wall and
  run, and the server's top five; everyone gets a chat line per run and `/zones` and `/top`. Each
  server keeps its own runs in a SQLite file (`gradientdrift_zone_records.db`, the system's own
  SQLite, no extra files). On by default (`ZoneScoring`, `ZoneHud`, `ZoneScoreChat`,
  `ZoneRecordsFile`, `ZoneBoardSize`).
- Setup has a zone editor: **Zones...** on its last page, or `gradient-drift-setup zones <server or
  track>`. It shows a track's drift zones as yellow striped boxes on a map of the course; drag a
  zone's ends, draw new ones, move one to the other edge, change its kind, delete, or put back the
  automatic ones. Load a recording (AI Car Monitor, plugin telemetry or the game's recorder) to
  see each driver's rear-axle path and how well it fills each zone. Saved zones are marked as
  edited and kept (with their ends following the course) when setup or `build_zones.py` makes the
  track's data again; `--replace` places them afresh. In a terminal the same with words
  (`zones <track> help`), and `picture` draws the map as an SVG.
- Zones follow a smoothed course edge: where an open corner's run-off made the edge jump out for a
  few points or zigzag, `build_zones.py`, setup and the plugin's paint hold each side within 1 m of
  its neighbours' median and average it over 5 points first (only ever pulling the edge in), so
  automatic zones no longer land on spike tips and the painted boxes run smooth. The painted outer
  line sits half its width inside the edge. Courses without spikes get the same zones as before.
- The plugin paints the track's drift zones on the road for players with Custom Shaders Patch:
  yellow boxes along the outside of the corners, closed at both ends, striped inside (sparser for a
  touch-and-go area), sent as a CSP online script that uses CSP's track paint. On by default
  (`ShowZones`; `ZoneColor`, `ZoneLineWidthMeters`, `ZoneStripeSpacingMeters`). Setup copies the
  track's `<track>_course_zones.csv` to the server with its other files, and copies it again after
  the zone editor changes it.
- Course edges follow the white lines a track paints along its sides. Setup now writes each track's
  painted-line meshes (`<track>_paint_lines.csv.gz`: scenery whose name says paint, line or stripe
  and that is a strip under 0.4 m wide). `build_course.py` keeps the strips that lie on the road
  and run at least 30 m along the racing line at least 4 m from it on average (a line the cars drive
  along, like the one round Klutch Kickers' hairpin, is not the side), then puts each side's edge on the course side of
  the line wherever the line is no more than 3 m past the edge it found before. The edge only
  moves in, so tyre stacks inside the line still set it. Gaps of up to 10 m are bridged, and a
  longer unpainted stretch keeps the old edge, blended in over 10 m. On Spirit Peaks the edges,
  and with them the zones, move in by about 1 m (up to ~2 m) and sit on the white lines, except on
  the unpainted main straight. Tracks without painted side lines are unchanged.
- `drift_rl/zone_line.py` moves a course's AI line into its drift zones (3 m inside each zone's
  edge, eased in and out), keeping the originals beside it and `--undo` to put them back. The bots
  follow the AI line, so this puts them in the zones with no new policy: in the simulator the
  shipped policy's line score goes from 0 to 8 - 14 of 40 on Spirit Peaks, 16 to 35 on Drift
  Playground and 24 to 26 on Klutch Kickers, with a little more running off the course edge (at 2.5 m; on a test server 2.5 m touched walls and 3 m did not).
  Track prep (`setup_track.py`, which setup runs for a new track) now builds this drift line itself
  after the zones (`--drift-line off` keeps the AI line as it is); for track data made before, run
  `zone_line.py` once, and again after editing the zones. Run setup afterwards to put the moved line in the server.
- New `drift_rl/zone_report.py` measures how well policies fill each drift zone of a course
  (share filled, where the rear axle runs against the zone's edge, where the AI line runs).
- Course edges are one continuous curve where paving, ground or a wall sets them (not markers): a
  running minimum over 5 points, then a running mean over 7, so they no longer step in and out
  by up to a metre from point to point. Where the ground changes, the edge is now put on the
  boundary itself rather than at the first sample past it (up to 0.25 m out). On Klutch Kickers
  the zones' outer edges no longer reach the cement and sand beside the tarmac, and their kinks
  of 20 - 30 degrees are gone.
- Course edges also end where the track's model draws the ground as sand, grass or the photo
  ground around the track, even when its surface key counts it as track. Setup now writes each
  track's collision materials (`<track>_collision_material.csv.gz`), and `build_course.py` stops
  an edge at a run of 0.75 m or more of such a material (sand, grass, dirt, gravel, mud, soil,
  cement). Kerbs, rumble strips, other tarmac and whatever the racing line itself drives on stay
  part of the course. On Klutch Kickers the edges now follow the visible tarmac, where before they
  ran up to 2 m onto sand strips, cement aprons and painted grass.
- Course edges (`drift_rl/build_course.py`, which setup runs for each track) no longer spike out
  across open run-off or cross into a neighbouring stretch of the course. Each side is read along
  three rays and the middle one kept, so one crack in the mesh or one gap in a wall no longer
  makes a dip or a spike. An edge may not reach nearer another stretch of the line than its own
  point, which covers the inside of tight corners and stretches side by side. It may also widen
  only gradually along the line. Klutch Kickers' widest edge drops from 30 m to under 11 m, and
  its zones go from 10 scraps on spike tips to 5 real corners. Setup remakes the edges the next
  time it builds a track.
- AI Car Monitor (`tools/ai_car_monitor.py`) records about 15 times less: every car 20 times a
  second (was every frame), only the values the game really sends for another player's car (the
  always-zero columns are gone), written straight to `.csv.gz`. A full server is ~1 MB a minute,
  was ~15 MB. The crowd tools read the new files, and a recording cut short by a crash reads up
  to its last few seconds.
- Bots drift Klutch Kickers at the public drivers' pace: the pace reference (`pace_model 4`) is
  scaled where public drivers have been recorded (`drift_rl/crowd_fit.py` on AI Car Monitor
  recordings). Klutch Kickers goes up 9%; Drift Playground and every other track are unchanged.
  Setup remakes an older pace reference.
- `ChaseTimeGapSeconds` defaults to 0.25 (was 0.2): with `ChaseGapMeters` 2 it is the gap public
  drivers settle at behind each other, 5.5 m at 50 km/h.
- Trainer: `--chase-gap-crowd FILE` asks for chase gaps in the public drivers' proportions.
- Setup puts the AI line where AssettoServer 0.0.54 reads it on a track with layouts:
  `content/tracks/<track>/ai/fast_lane.ai`. It used to go in the layout's own `ai` folder, which
  0.0.54 never reads, so the server would not start ("No ai folder found").
- Setup refuses a track whose model files are protected (every vertex moved, which the game's
  Content Manager / Custom Shaders Patch undo on load): its ground came out as a crumpled sheet with
  layers metres apart, and the bots climbed it into the air.
- `FittedTireCorrection`, a policy-file setting (off by default): each axle's sideways force plus a correction fitted to Assetto Corsa recordings, so the bots' cars stop over-rotating at the end of a transition. A policy has to be trained with it before it changes anything.
- Bots no longer put their nose or tail through guard rails. Walls now touch the car's body, sized
  from its `collider.kn5` and checked every 0.5 m round its edge (`BodyWallContact`, on by default).
  Before, six short pins around the direction of travel, with none straight ahead or behind, let a
  bumper into a rail by up to a metre. Run setup again to write each car's body size into its car file.

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
