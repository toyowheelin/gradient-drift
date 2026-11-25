# Gradient Drift scoring plugin

Players' drift scoring for AssettoServer: each run judged zone by zone (or arcade style), a HUD for
players with Custom Shaders Patch, a chat line per run, each server's board, and the track's drift
zones painted on the road. It is its own plugin, so it needs no bots, no policy and no
GradientDriftPlugin: a competition server can run it on its own. On a server with the bots, setup
installs it beside GradientDriftPlugin.

## What it needs

- **Arcade mode** (`ZoneScoreMode: Arcade`): nothing but the plugin. It works on any track, with AI
  on or off.
- **Zones mode** (the default): the track's drift zones and course edges
  (`<track>_course_zones.csv` and `<track>_course_edges.csv`, which setup writes with the track's
  data; its zone editor changes the zones), and the server's AI line they sit on. AssettoServer loads
  the AI line only with `EnableAi: true` and the track's `ai/fast_lane.ai`, but no AI cars are needed:
  with AI on and no AI entries in the entry list, there are no bots. With AI off the plugin says so in
  the log and scores nothing in this mode. It reads the track files from GradientDriftPlugin's data
  bundle when that plugin is installed beside it with one, else loose from the server's folder.
- A course that is one closed lap. Point-to-point courses are not handled yet.

## Install

```yaml
EnablePlugins: [GradientDriftScoringPlugin]   # loads plugins/GradientDriftScoringPlugin/GradientDriftScoringPlugin.dll
--- !GradientDriftScoringConfiguration
ZoneScoreMode: Zones
```

Put `GradientDriftScoringPlugin.dll` and `GradientDriftScoringPlugin.deps.json` in the server's
`plugins/GradientDriftScoringPlugin/` folder. Every setting has a working default, so the block can
hold only what you want to change, or be left out. Settings are read when the server starts.

Before this plugin existed these settings were in GradientDriftPlugin's block. That plugin still
accepts them so an older `extra_cfg.yml` starts, but ignores them and warns at start; setup moves them
here.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `ShowZones` | `true` | Paints the track's drift zones on the road for players with Custom Shaders Patch: yellow boxes along the outside edge of the corners, closed at both ends, striped inside (sparser for a touch-and-go area). The zones are the ones the bots are trained to fill, from `<track>_course_zones.csv` (setup writes it, and its zone editor changes it) on `<track>_course_edges.csv`'s points. Players without CSP, or with a CSP too old for its track paint, see nothing. |
| `ZoneColor` | `F5C400` | The zones' colour, as RRGGBB. |
| `ZoneLineWidthMeters` | `0.3` | How wide the zones' frame is painted; the stripes are 0.8 of it. The outer line sits half its width in from the course edge, so the paint stays on the course. |
| `ZoneStripeSpacingMeters` | `1.6` | The gap between stripes in an outside zone, along its edge; twice this in a touch-and-go area. |
| `ZoneScoring` | `true` | Scores players' drifting through the zones, from the car status the server already gets (nothing is taken from the player's game, so it cannot be faked there). Each zone gets up to 100: line 40 (the share of the zone driven with the rear axle within 3 m of its edge while sliding 20° or more), angle 30 (full at 45° on average), speed 30 (full at 95% of the track's pace reference speed, or 70 km/h without one); angle and speed count for the filled share only. A tandem bonus of up to +10 for sliding within 8 m behind another car; a spin (slip past 100°) scores the zone 0. A run goes from the first zone after the longest stretch with none to the last zone and scores the zones' mean, less 5 for each wall hit; going the wrong way, jumping (pits, teleport), stopping, or car data that does not add up (faster than 110 m/s, or a reported speed that does not match how far the car moved) ends it unscored. Players' names are kept without control characters, at most 32 long. The board knows players by their Steam ID, which AssettoServer only checks with `UseSteamAuth: true`; without it the plugin warns at start that anyone could keep runs under someone else's ID. Needs the zone files `ShowZones` paints, and the server's AI line (see [What it needs](#what-it-needs)). |
| `ZoneScoreMode` | `Zones` | How drifting is scored. `Zones`: through the zones, as above. `Arcade`: freestyle, no zones needed, so it works on any track (no track files); the zones are not painted. Points build every moment the car slides (15° or more at 30 km/h or more): 100 a second x angle (slip over 45°, at most 1.5) x speed (over 80 km/h, at most 1.5) x combo (starts at 1, +0.25 with each transition to sliding the other way, at most 3) x tandem (up to x2 sliding close behind another moving car: nothing at 10 m, full at 2 m). The chain is banked once the car has not slid for 1.5 s; a spin (slip past 100°), a wall hit, a jump or car data that does not add up loses it. Chains under 100 points are not counted. The board, HUD and chat show chains instead of zone runs, and each mode keeps its own board in `ZoneRecordsFile`. |
| `ZoneHud` | `true` | Players with Custom Shaders Patch get a HUD (a CSP online script the plugin sends): a panel with an angle gauge, their speed against the track's pace, the zone they are in with a live line-fill bar and its points so far, the run so far, their best and place, and the server's top five; its edge glows yellow in a zone. Drag it with the left mouse button, size it with the wheel and switch its speed between km/h and mph with a right click (it starts in the game's units where their CSP says which; all remembered on their PC). Popups as each zone is judged (a grade word, the points, line / angle / speed / tandem), for a spin or a wall hit, and for a run's score (NEW BEST when it beats theirs). It reads the numbers from AssettoServer's HTTP port (`/gradientdrift/zones/<car>`, `/gradientdrift/board`); `EnableClientMessages` is not needed. |
| `ZoneScoreChat` | `true` | A chat line to the driver after each scored run (score, line, angle, speed, best and place), which players without CSP see too; a new server record is announced to everyone. Chat commands `/zones` (how runs are scored, your best) and `/top` (the board) work either way. |
| `ZoneRecordsFile` | `gradientdrift_zone_records.db` | Where this server keeps its scored runs: a SQLite database, relative to the server's folder, so each server has its own records. The board is each driver's best on this track with these zones (editing the zones starts a new board; old runs stay in the file). Uses the system's SQLite (built into Windows 10 / Server 2016 and later; `libsqlite3` on Linux): no other file ships with the plugin. Empty = keep nothing. |
| `ZoneBoardSize` | `10` | How many drivers the board shows: a web page at `http://<server>:<HTTP port>/gradientdrift/leaderboard`, and the HUD's top five (it reads `/gradientdrift/board`, plain text). || `VerboseLogging` | `false` | The plugin's routine lines in the server log as well as its warnings. |

## Building

It builds like GradientDriftPlugin (see the [main README](../README.md#building-the-plugin)), inside
an AssettoServer checkout, next to `GradientDriftPlugin/`: it compiles a few of that folder's files
(the track-file readers) so both plugins read the track's files the same way.
