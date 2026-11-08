# Regression tests

CI runs these on every pull request, so the obvious breakage is caught before anyone builds the
plugin or drives on a server. None of it needs game data: the tracks are the trainer's own
generated courses (`drift_rl/generate_track.py`, the same course every time for a seed and number)
and the car in the setup test is made-up numbers. They do not judge how well a policy drives;
`drift_rl/drive_report.py` and driving on a server do that.

| check | file | catches |
|---|---|---|
| Golden physics traces | `test_sim_golden.py`, `sim_traces.py`, `golden/sim_traces.json` | a change to the simulator's physics nobody meant: four fixed input sequences (a lift-off, full lock, a flick and countersteer, the front brake mid-turn) in three model variants, compared with the saved paths |
| Plugin vs trainer | `plugin/` (run by `build-plugin.yml`), `test_plugin_settings.py`, `test_plugin_cases.py` | the C# plugin giving other actions than `drift_rl/policy.py` for the same inputs (the shipped policy, and made-up ones with two outputs, deeper layers, a residual chase network and a history block); the plugin loading a broken policy or one that needs a setting it cannot apply; the trainer able to write a setting the plugin refuses |
| Short drive checks | `test_drive_smoke.py` | a broken policy, simulator or track tool: the shipped policy drives five generated courses for 30 s each with 8 cars, as the server would (nothing resets a car), and must slide, not spin, and stay off the walls and on the course |
| Setup | `test_setup.py` | setup on a fake game install, server folder and plugin build: it installs the plugin, track files, AI line and car file, edits the server's config, takes out a setting the build lacks, finds nothing to do on a second run, keeps a policy made for the server, refuses the game's own server, and undo puts every file back |
| AI Car Monitor | `test_ai_car_monitor.py` | the in-game logging app (run against a stand-in for the game's API) sampling at another rate than 20 a second, cars sampled at different instants, a recording the crowd readers can't take, or one cut short by a crash that no longer reads |

## Running them

From the repository root, with the trainer's packages (`drift_rl/requirements.txt`, no PyTorch
needed) plus `ruamel.yaml` and `pytest`:

```
python -m pytest tests
```

The first run takes about a minute, most of it compiling the simulator (numba keeps it after that).
`GD_TEST_TRACKS=<folder>` keeps the generated courses between runs.

The plugin checker needs the .NET 8 SDK and an AssettoServer checkout, set up as for building the
plugin (see the main README): copy `tests/plugin/` to `AssettoServer/PluginChecks/`, then

```
python tests/plugin/make_cases.py cases
dotnet run --project AssettoServer/PluginChecks/PluginChecks.csproj -- "$PWD/cases"
```

## When a test fails on purpose

A physics change that is meant moves the golden traces. Write them again and commit them with the
change, saying in the pull request why they moved:

```
python tests/sim_traces.py --write
```

A new plugin setting goes into both `drift_env.PLUGIN_SETTING_PARAMS` and
`DriftPolicy.CanApplySetting` (and `ApplyPolicySettings`), or `test_plugin_settings.py` fails. A new
policy shape (another input count, another kind of network) gets a made-up policy in
`plugin/make_cases.py`.

Your real tracks stay a check on your own machine: `drift_rl/drive_report.py <policy> --track <track>`
with the track's data in `drift_rl/tracks/`.
