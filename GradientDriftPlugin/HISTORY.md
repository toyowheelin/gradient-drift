# GradientDriftPlugin — how it got here

(DriftAiPlugin until 2026-10-12; the entries before then use that name.)

The source in this folder is the cleaned-up plugin: a learned policy drives the car directly and
nothing else does. This file records what was removed to get there and the measurements behind
the mechanisms that are left, so the code does not have to carry that history in comments.

## Timeline

| when | what |
|---|---|
| 2026-09-14 | **Project started.** The owner's first ask: "The other day I found some online drift tracks where they were running AI cars that actually drifted, I assume they are also using assettoserver. Anyway I want to know if that is something that you could help me figure out how to implement on my servers?" The plugin replaces the stock AI's position/velocity/rotation with its own vehicle model, steered by a hand-written controller: spline-following steering law, grip-based speed planner, gap steering from a ray fan, lateral pin steering, drift-intent planning. |
| 2026-10-01/02 | `GlitchCorrector` and `NearestPointCorrector` split out; wall push-out; `UncontrolledSpinRecovery` added after two runaway spins (445.8 km/h, and a car that left the map). |
| 2026-10-03 | Learned policy added as a *residual* on the hand-written controller (steer residual ±0.5 rad, speed ×0.7–1.3). First server test (`long1`): the C# port was right, the car left the track for 45 s. Cause: the simulator placed the car at the spline's height, the plugin did not, and the horizontal sensor rays saw different walls. |
| 2026-10-04 | Simulator given the plugin's height model (`fix2_from_long1`): server run matched. Ground-ray ride height replaced the spline/road-mesh blend (the car had been floating up to 21 cm through a downhill corner). |
| 2026-10-04 | Course layout: the paving is 25–30 m wide, the course between the tire stacks 11–24 m, and no sensor can see a tire stack. The trainer was taught the course (`course1`, `course2`): time outside it 19.7% → 0.0% on the server. |
| 2026-10-04 | Linked drift ("manji") down the straight: a reward for keeping a drift going, a ±1.0 rad steering range, and a scripted demonstration to start from (`manji3`): 71% drift time on the server. |
| 2026-10-04 | **Direct control** (`direct2`): the policy sets the steering angle and target speed itself. Server run: 81.7% drift time against a predicted 81.7%, one-tick replay within 1 cm. Wrong-way recovery replaces `UncontrolledSpinRecovery`. Steering visual added. |
| 2026-10-04 | `RearSlidingGrip` added (the speed dial): the rear axle's sideways force falls to a share of its peak once it is sliding. The bot slid at 74 km/h and 14.4 m/s² sideways; a human-driven car on the same course at 42 km/h and 6.6 m/s². |
| 2026-10-04 | Drivetrain visual: the bot's wheel speeds, engine RPM, gear and throttle now come from its own car. The stock AI sent one speed for all four wheels, 800 – 3,000 RPM, first gear and no throttle, from its own spline follower. |
| 2026-10-05 | High-angle kit: the real cars' 60° steering lock (the model had 28.1°) and a rear thrust limit, both carried by the policy file (`plugin_settings`) so older policies drive as before. |
| 2026-10-06 | The manji looked like the car rotating about its centre (test 13). Measured: flips of 0.37 s at 158°/s with the path turning 2.8°, steering at up to 548°/s. Steering rate limit (`MaxSteerRateDegS`) and the thrust limit's other side (`EnableRearBrakeLimit`) added, both carried by the policy file. |
| 2026-10-09 | Track geometry from two prebuilt files (`tools/track_triangles.py`, the same two the trainer loads) instead of running Python on the track model at server start: `<track>_triangles.csv.gz`, what the sensors see, and `<track>_collision.csv.gz`, what the game itself collides with. The ground ray and the ground attitude use the collision file's ground, the wall push-out its walls (a second set of contact pins on it): on House of RWD the barriers' visible meshes are named neither wall nor tire, so the push-out had let the car through. Spirit Peaks' ground is in `drivingsurface.kn5`, never read before; it lies 2.5 cm below the visible road the car used to ride on. First version (ground only) built and run the same morning (run 27), the collision walls an hour later (run 28): both match the simulator. |
| 2026-10-09 | Several bots, spaced (`HoldBotSpacing`, written, not yet compiled or run): a car taken over goes to the middle of the largest gap between the other bots, and each bot's own `RearSlidingGrip` is eased inside the policy's range (a slightly faster or slower drift) to keep the gaps even. Two telemetry columns: `RearSlidingGrip`, `GapAheadM`. |
| 2026-10-09 | `NearestPointCorrector`: an implausible jump in the nearest point is replaced by the reachable point nearest the car, no longer by the furthest one forward, which ran the car's place on the line ahead of it while it drove beside another part of the track (RiverSide). Same change in the simulator. Written, not yet compiled or run. |
| 2026-10-09 | Tandem trains (`TrainSize`; built and run the same evening with six bots in two trains of three - "look great", the owner): the first bot of each train is driven by the policy, the others replay what it sent a moment earlier, door to door (`TandemGapSeconds` 0.35 s per place, varied ±25% slowly per car, never within `TandemMinGapMeters` 4.8 m of the car in front), each wandering 0.1 - 0.6 m to one side of the lead's line. No policy or physics for the followers, so any car can follow. Leads in a train ignore the stock AI's stops (it measures gaps between the positions the plugin sends and would stop a lead for its own follower). |
| 2026-10-09 | Follower telemetry (written, not yet compiled or run): a row per tandem follower per tick with `FollowLeadId`, `FollowPlace`, `FollowDelayMs` (-1 while the server's AI has it), `FollowGapM` and `FollowOffsetM` appended; -1 in `FollowLeadId` on every other row. `drift_rl/tandem_report.py` summarises them; `check_plugin_policy.py` skips them. |
| 2026-10-10 | Room to the course's edges (written, not yet compiled or run): six new policy inputs, 46 - 51, how far the course reaches either side of the car at its point and 15 m and 30 m on (from `<track>_course_edges.csv`, `CourseEdges.cs`). On a grass-edged road no sensor saw where the road ends (the fan rays are level and pass over grass): `gen2` left RiverSide's 10 m paving 13 - 67% of the time, everywhere round the lap. 46-input policies still load and drive as before; a 52-input one needs the course edges file or the stock AI drives. |
| 2026-10-10 | Pace and left-foot braking (written, not yet compiled or run). A policy trained since today has a third output, the brake with the throttle (front axle only, the front tires' sideways grip capped by what the braking uses; `DriveByPedal`), and reads two more inputs, 52 - 53: the pace it is asked to drive and its last brake. For such a policy `HoldBotSpacing` moves each bot's pace around `BotPaceKmh` (new setting; 0 = the middle of the policy's `pace_kmh_range`) and leaves every bot at the same `RearSlidingGrip`: spacing by grip had parked a bot at 0.40 for a whole run, where it hit the barrels on the sweeper's exit every lap. Older policies load and drive as before (46/52 inputs, 2 outputs; grip spacing). Telemetry: `PolicyBrakeAction`, `PaceKmh`. |
| 2026-10-10 | Timing (written, not yet compiled or run): every `LogTimingIntervalSeconds` (60; 0 = off) a server-log line with the plugin's own time on the update thread - ms a tick split into network, sensors (rays, ground, wall push-out), telemetry and followers, the worst tick, and the share of one CPU core (`PluginTiming.cs`). Asked before rolling the bots out to the owner's eight servers; the estimate without it was ~1% of a core for two leads and four followers. |
| 2026-10-11 | Timing measured (gen4, two leads, four followers, 18 Hz): 26.5 ms a tick, 48% of one core, all but 0.6 ms of it sensors - not the ~1% estimated. Cause and fix under "Sensor cost": `TriangleGrid` with 2 m cells for ordinary triangles and a coarse grid for the large ones (compiled and run the same day: 2.55 ms a tick, 4.6% of a core; same hits as before). |
| 2026-10-11 | Pace error input (run on the server the same day, test 19: all checks pass): a policy with 55 inputs also reads 54, how far over the speed asked of it the car has been driving lately - its progress along the line over pace x the course's reference speed at each point, both averaged over `pace_average_s` from the policy file (1 s), minus 1 (`UpdatePaceOver`, the simulator's counters 8 and 9). Needs `<track>_pace_reference.csv` (`PaceReference.cs`) and a closed AI line, or the stock AI drives. Telemetry: `PaceOver`. |
| 2026-10-11 | Spacing keeps bots out of the slow end (`BotPaceMinPercent`, 85; compiled in CI, not yet run): gen5 slides 61% of the time asked 0.80 against 78% at 0.90 (Spirit Peaks, simulator), so spacing by pace no longer slows a bot below 85% of flat out; `BotPacePercent` is still the middle. Run on server 6 the same day (test 20). A policy trained on generated courses (a `generated` object in its file, as gen5 has) now drives a track that is not in its `tracks` list - the server log says the track is new to it - where before the stock AI drove; setup no longer blocks such a track either. First unseen track, Drift Playground (simulator, gen5, 3 min runs with no resets): asked 0.85 / 1.00, drove 0.83 / 0.88 of its flat out (42.2 / 44.8 km/h), sliding 64 / 71%; but at one hairpin (point ~150, 9 m radius, at the foot of an 8% climb, objects on the grass inside it) it turns in early and cuts across about half the laps: 8.6 - 11.9% of the time off the road, 5 - 7% on walls. gen4 takes that hairpin cleanly (0.3% off the road, 1.7% walls, 46.4 km/h). The server agreed (test 20): 7.8% off the road, 5.0% walls, at the same hairpin, and the followers' only jumps on a player's screen came from replaying their lead's wall push-outs there. |
| 2026-10-11 | What the sensors see (track data only; the plugin is unchanged): `<track>_triangles.csv.gz` is now the walls the game collides with plus a prism for each course marker (`drift_rl/sensor_view.py`), no longer the track's model as drawn. On Drift Playground the drawn model has a 5 m wedge on the grass inside the hairpin of test 20 where the game has a thin barrier; with the solid view gen5's runs cut short there fell from 44 / 16% to 3 / 0% (pace 0.85 / 1.00, simulator). The pace reference is now measured robustly (per point the median over cars, without each car's last 3 s before it was lost): on Drift Playground the old one asked 16 km/h before that hairpin, the new one 45. gen6 trains on both. |
| 2026-10-11 | Foot brake and the front tires (written, CI-compiled, not yet run): braking with the pedal now takes from the front tires' sideways grip as the brake action already did (their friction circle); before, the front could brake and steer at full force at once, up to 8% over what the tires have. Same change in the simulator. Not the cause of the weaving the owner saw on server 6 (gen5 flicks to bleed speed; the trainer's new weave penalty is for that). |
| 2026-10-11 | Security review before release (written, CI-compiled, not yet run). The plugin has no network surface of its own (no ports, endpoints, commands or client messages) and reads only operator files. Two ways it could stop a server, both fixed: AssettoServer skips a tick whose update throws and shuts down after 10 in a row, and nothing caught the plugin's errors - now logged, and after 5 in a row the plugin stops and the stock AI drives; and the telemetry, on by default, grew ~110 MB an hour with no limit, and a full disk would have thrown on every flush - now `TelemetryMaxFileMegabytes` (500) and it stops on the first failed write (also if it cannot be created). Trainer: checkpoints load with `weights_only` (a shared checkpoint can no longer run code); CI workflows get read-only tokens. |
| 2026-10-11 | No files by default (the owner: for release the plugin should write nothing but what goes to the console): `EnableTelemetryLogging` and `LogHumanCarForCalibration` now default to off. A test server turns them on in its `--- !DriftAiPluginConfiguration` block. |
| 2026-10-11 | Tire load sensitivity (`EnableTireLoadSensitivity`, carried by the policy file; written, CI-compiled, not yet run): each axle's peak friction is the tyre file's zero-load grip (DY0) less DY1 per kN on each tire, with cornering's load moved onto the outside tire (CG height, track, roll stiffness share; the Miata's numbers by default, a policy file may carry others) - `AxleFriction`, the simulator's `axle_friction`. Grip at rest 1.24 / 1.22 instead of 1.37 / 1.34. Why: the bots pulled more sideways g than the owner's car and rotated faster (test 21); with it gen6's yaw acceleration came down to the human's in the simulator. Older policies drive as before. |
| 2026-10-12 | Renamed for release (the owner): `GradientDriftPlugin` - the folder `plugins/GradientDriftPlugin/`, the DLL, `EnablePlugins: [GradientDriftPlugin]`, the settings block `--- !GradientDriftConfiguration` and `GradientDriftPlugin.policy.json`, in AssettoServer's own pattern (`RandomWeatherPlugin` / `!RandomWeatherConfiguration`). Log lines start `GradientDrift:`. Setup moves a server set up under the old name: the `EnablePlugins` entry and the settings block renamed in place, the policy the old folder ran copied into the new one, the old folder left for the owner to delete. |
| 2026-10-12 | Setup renamed with it: "Gradient Drift Setup", the program `gradient-drift-setup`, package `gradientdrift_setup`, backups in `gradient-drift-setup-backups` (undo also finds runs in the old `driftai-setup-backups`). |
| 2026-10-04 | This cleanup: everything that only existed for the hand-written controller removed. First build ran on the server the same evening (test 10): every check passed, one-tick replay within 1 cm. |
| 2026-10-18 | `BotPacePercent` unset (0) now means the top of the policy's trained pace range (100% for gen10 and gen11), not the middle (90%): the owner, after the Miatas on Drift Playground felt far too slow with no pace in his config. Spacing then varies a bot's pace between `BotPaceMinPercent` and that top. |
| 2026-10-19 | Quiet log for the release (written, not yet compiled or run; the owner: say something only when the bots have a problem): the routine startup and per-car lines are gone (one `policy loaded from` line stays), `LogTimingIntervalSeconds` defaults to 0. `BotHealth.cs` checks every 10 s that each policy-driven bot has moved 10 m; one that has not for 30 s gets one warning (the policy has it but it is stuck, or the server's own AI holds it stopped) and a line when it moves again, at most one warning a bot every 10 minutes. Spirit Peaks' log for 2026-10-19 had been 605 timing lines and nothing else. The plugin also warns at start when AssettoServer's version is not the one it is built for (`BuiltForAssettoServer`, 0.0.54; build-plugin.yml checks it agrees with setup and the build). |

## Server tests

| # | policy | outside the course | markers / min | drift time (20–50° slip) | simulator predicted |
|---|---|---|---|---|---|
| 1 | long1 (residual) | – | – | 44.1% | 50.2% (old simulator) |
| 2 | fix2_from_long1 | – | – | 46.0% | 46.0% |
| 3 | fix2_from_long1, ground ray | 19.7% | 13.7 | 40.0% | 44.3% |
| 4 | course1 | 0.9% | 0.6 | 31.6% | 32.7% |
| 5 | course2 | 0.0% | 0.0 | 46.2% | 45.1% |
| 6 | manji3 | 0.2% | 1.3 | 71.0% | 70.8% |
| 7 | direct2 (direct control) | 0.2% | 1.1 | 81.7% | 81.7% |
| 8 | direct2 again | 0.2% | 0.9 | 82.0% | 81.5% |
| 9 | direct3, steering visual | 0.0% | 1.0 | 82.4% | 82.6% |
| 10 | direct3, this cleaned source | 0.0% | 0.3 | 83.3% | 82.9% |
| 11 | slide1, RearSlidingGrip 0.6 (59.0 km/h; predicted 58.7) | 0.2% | 0.7 | 69.1% | 69.7% |
| 12 | slide2, RearSlidingGrip 0.6 (59.9 km/h; predicted 59.6) | 0.0% | 0.3 | 84.0% | 84.0% |
| 13 | angle4, RearSlidingGrip 0.5, 60° lock, thrust limit (58.2 km/h; predicted 58.1; mean slip 39.8°; predicted 39.7°) | 0.0% | 0.0 | 84.4% | 84.4% |
| 16 | both_trial: + steering 150°/s and rear brake limit (56.7 km/h; predicted 57.4; flips of 0.44 s at 130°/s; predicted 0.45 s, 129°/s) | 0.0% | 0.8 | 77.5% | 73.4% |
| 17 | angle5 (both_trial + 100M ticks), with ground attitude, lean, skid rule, second-gear engine (57.0 km/h; predicted 57.3; mean slip 41.0°; predicted 41.1°) | 0.4% | 0.9 | 67.2% | 65.2% |
| 18 | gen4 (paces in km/h, brake), two bots spaced by pace: asked 60 / 40, drove 55.4 / 54.0 km/h over the ground | – | – | 83.7% | – |
| 19 | gen5 (paces as a share of flat out, pace error input), two trains of three: spacing asked 0.80 - 0.87 / 0.93 - 1.00, drove 49.9 / 56.3 km/h; gaps 208 / 629 m -> 360 / 476 m in 100 s | 0.0% | 0.6 | 74.6% | 78.2% |
| 20 | gen5 on Drift Playground, a track it never trained on (server 6), two trains of three, `BotPaceMinPercent` 85: leads drove 43.7 / 47.7 km/h at pace 0.85 / 0.98 - 1.00; gap 51 -> 194 m in 200 s (even: ~397 m); wall contact 5.0% (predicted 6.6%), almost all at the hairpin at point ~160; every plugin check passes; 2.8% of one core | 8.2% | 0.0 | 63.9% | 64.2% |
| 21 | gen6 on Drift Playground (server 6), solid-wall sensors, new pace reference, two trains of three, ~2.3 min a lead: every plugin check passes; walls 1.1% (predicted 0.9%), wrong-way recovery 1.7% (up to 1.8% predicted); yaw-rate reversals 13 - 16 a minute (gen5 13 - 19), yaw acceleration p90 286 - 306 deg/s2 (gen5 264 - 283, the human 185) | 2.7% | 0.0 | 54.6% | 57.7% |
| 22 | gen7 on both servers, tire load sensitivity on (after the plugin fix: its settings check had refused gen7 and the stock AI drove; one run with gen5 first): Spirit Peaks, ~7 min of leads at paces 0.85 - 0.95: yaw acceleration p90 201 - 233 deg/s2, direction changes 5.3 - 8.1 a minute, sideways g p90 0.79 - 0.89, sliding 83 - 85% (the human 185, 7.2, 0.83, 86%; gen5 the same evening 259 - 275, 13 - 14); walls 0 - 0.4%, no spins. Drift Playground pace 0.85: 233, 9.1, walls 0.1%, spinning 0.3%; pace 1.0: 287, 17.5, walls 1.8%, spinning 7.1% (predicted 252, 12.9, 0.9%, 2.7%), the spins at points 80 - 120, 240 - 280 (the uphill S the simulator found), 320 - 440 | - | - | - | - |
| 23 | gen9 (any car, skill input) on Spirit Peaks, BotSkill 1, per-bot car files from setup (six gravygarage street cars), two trains of three, ~4 min a lead at pace 0.90: every plugin check passes (all 67 inputs incl. car inputs and skill); 54.7 km/h (predicted 54.8), yaw acceleration p90/p99 197 / 348 deg/s2, direction changes 5.9 a minute, sideways g p90 0.90, sliding 87% (the human 185 / 311, 7.2, 0.83, 86%); walls 0.0%, spinning 0.04%, high angle (40-80 deg) 19.5% vs 23.8% predicted | 0.3% | 1.4 | 83.7% | 83.1% |
| 24 | gen9 on Spirit Peaks at `BotSkill` 0.1, two trains of three, leads at pace 0.85 (4 min) and 1.0 (3 min): every plugin check passes once `check_plugin_policy.py` knows skill (the brake is the network's action one tick late, exactly; steering and pedal noise std 0.088 / 0.184, tick-to-tick 0.89 / 0.90, as HandicapBySkill makes them); the simulator at skill 0.1 matches the run (sliding 62.3% vs 61.3%, high angle 4.3% vs 3.9%, mean slip 22.3 vs 22.5 deg). Pace 0.85 / 1.0: 52.0 / 55.5 km/h, sliding 67 / 71%, yaw acceleration p90 203 / 201, direction changes 8.4 / 6.1 a minute, sideways g p90 0.96 / 1.03, walls 0.1 / 0.0%, spinning 0.1 / 0.3% (simulator 215 / 200, 8.8 / 6.3, 0.0 / 0.5% spinning). Sloppier than skill 1 (sliding 84 - 88%) as designed, and safe | 0.2% | 0.7 | 61.5% | 61.0% |
| 25 | gen10 on Drift Playground (server 6, never trained on), BotSkill 1, merged plugin build, no car files yet (bots on the configured car's numbers), two trains of three, ~10 min of leads: the plugin's actions match gen10 exactly (steering and pedal to 1e-4, front brake = the network's output clipped at 0; all 67 inputs). Pace 0.85 / 0.95 / 1.0: 43.9 / 47.3 / 47.4 km/h, sliding 61 / 67 / 72%, yaw acceleration p90 221 / 236 / 202 deg/s2, direction changes 15.0 / 10.3 / 7.9 a minute, sideways g p90 0.91 / 0.97 / 0.98, walls 0.3 / 0.1 / 0.2%, spinning 0.2 / 0.0 / 0.0% (test 22, gen7 here: pace 1.0 287 deg/s2, 17.5 a minute, walls 1.8%, spinning 7.1%). Calm and clean at full pace; at 0.85 it still flicks twice as often as the human (7.2) | - | - | - | - |
| 26 | gen10 on Spirit Peaks, BotSkill 1, per-bot car files, two trains of three, ~12 min of leads (the actions match gen10; a run just before drove gen9 by mistake - the plugin build folder's policy file was still gen9 and was redeployed - and gives a same-evening gen9 comparison). Pace 0.85 / 0.95 / 1.0: 52.7 / 55.7 / 56.3 km/h, sliding 82 / 89 / 87%, yaw acceleration p90 192 / 190 / 191 deg/s2, direction changes 4.8 / 3.8 / 6.6 a minute, sideways g p90 0.87 / 0.92 / 0.91, walls 0.0%, spinning 0.0% (gen9 the same evening: 199 / 217 / 195, 6.3 / 4.9 / 6.7, spinning 0.4 / 0.6 / 0.0%; the human 185, 7.2). No spins at any pace, rotation at the human's; a little less lively, as car_report predicted | - | - | - | - |
| 27 | Measured pace scales (setup 2026-10-17) on Spirit Peaks: gen10, BotSkill 1, two trains of three at pace 0.9, ~1.7 min of each lead (Miata, S14 Draper). Setup wrote all six car files' PACE_SCALE (S13 0.9936, E36 Compact 1.0112, E36 Touring 1.0028, JZX100 1.006, Miata 0.9926, S14 0.976). The plugin asked each lead exactly its car's scale times the reference speed (pace error vs positions gives 0.9926 and 0.976, steady to 1e-3); check_plugin_policy all pass once it reads the scale back (it had no notion of it: 50% within 0.01 before). 53.9 km/h, sliding 81%, walls 0.0%, no spins; mean slip 29.5 deg just below the simulator's 30.3 - 31.5 (its comparison drives the reference car, not the bots'). Scales this close to 1 change little; the mechanism works | - | - | - | - |
| 28 | gen10 on Drift Playground with setup's car file (Miata, PACE_SCALE 0.927), BotSkill 1, two trains of three at pace 0.9 (asked 0.83 of the reference), ~4 min of each lead: actions and inputs match gen10, both leads asked 0.927 x the reference. 45.0 km/h, sliding 61%, walls 0.1%, no spins. The owner: several corners driven straight. Per 20 m of the 800 m lap: the right hairpin at 200 - 240 m (-170 deg of line turn) 2 - 23% sliding at 33 - 43 km/h, the left at 120 - 140 m 26%, the change of direction at 500 - 540 m 4 - 13%; the long left at 380 - 420 m 100%. As in test 25 (sliding 61 / 67 / 72% at 0.85 / 0.95 / 1.0): the policy never trained here | - | - | - | - |
| 29 | gen12 on Spirit Peaks, plugin build with the 100% pace default, two trains of three, ~3 min of the two leads (pace 0.96 / 1.0): 56 km/h, sliding 87%, no wall push-outs, no spins - but the owner saw the bots run into the tire wall on the right of the straight. They pass within 0.3 - 0.6 m of it at points 505 - 525 on every lap (centre 6.5 m right of the AI line). The wall is 15 - 27 cm tall in the collision file, below the proximity pins (0.15 m above the car's position, ~0.55 m above the road) and the rays, so neither the plugin nor the simulator sees it. The simulator agrees: within 1.5 m of it 36% of the time there at pace 1.0 (gen12 checkpoint at 65M 14%), gen10 0.1%, gen11 0.9% | - | - | - | - |
| 30 | gen12 on Drift Playground, same build, two leads at pace 0.88 - 1.0, ~3.4 min: 45 - 47 km/h, sliding 71 / 80%, walls 0.4 / 0.1%; the owner: the bots drive the hairpin (points 120 - 160) without drifting or spin out. Sliding there 15 - 33% (the simulator at 0.95: 3 - 15%), spin recovery 19% of the time at points 140 - 149. Cause: the server's pace reference asks ~70 km/h at that 9 - 11 m hairpin (gen3 cut it when it was measured); the owner takes it at 35 - 46. gen13 (trained against the target from his laps) put on server 6 with that target the same evening | - | - | - | - |
| 31 | gen14 on both servers (plugin from PR #13 with the load-order fix of PR #17, pace_model speed targets and car pace scales; the first build of PR #13 loaded no policy: 'no track geometry'). Drift Playground, ~3.5 min of two leads at pace 0.89 - 1.0: sliding 81% (gen12 in test 30: 73%), wall push-outs 0.04% (0.25%); the hairpin entry (120 - 130) slides 55% (12 - 32%) and the apex (140) 100% (39%) but at ~20 km/h (the owner 35 - 46), then the exit (150) is driven straight (9%), as in the simulator. The owner: it at least attempts the hairpin, still needs work. Spirit Peaks, two cars ~3.4 min: sliding 79%, through the tire wall stretch (500 - 530) 0 - 2 m from the line (gen12 sat 6.3 m right, on the wall); brief contacts on the right where the tire wall starts (487 - 498, single ticks) and twice 13 m left at point 473 (the outside wall before the straight, 7 ticks each). The owner: pretty normal, a bit of wall hitting coming down the straight | - | - | - | - |
| 32 | gen18 (DisableBrakes) on Drift Playground, 2026-10-20, two leads (Miatas, 2nd gear, no brake), ~4.6 min: HoldBotSpacing held one at pace 1.0 and slowed the other to 0.89 - 0.91. The full-pace bot kept ~30 deg out of the hairpin on 4/4 passes; the slowed one dropped from ~38 deg to under 10 at points 150 - 155 on 3/4 (it reaches the apex over the ask and lifts). The simulator agrees (NOTEBOOK "The slowed bot leaves the hairpin early"). |
| 33 | trialM (FrontBrakeOnly, Miata only; plugin from PR #25) on Drift Playground, 2026-10-20, two leads, ~4.6 min: the first start loaded no policy (no cars block: the resume bug of PR #26; added by hand). The slowed bot (pace 0.91) holds the hairpin on 3/3 passes, 36 - 58 deg through 140 - 160, front brake on at the entry; sliding 84%, walls 0.7%. The full-pace bot (1.0) over-rotates to 60 - 74 deg at 140 - 145, drops to 20 - 30 km/h at 150 and stands up (under 10 deg at 155 - 165) on 3/4 passes; sliding 82%, walls 0.8%. The simulator at 1.0 held 51 - 26 deg. |
| 34 | trialQ (FrontBrakeOnly, all cars, 505.7M ticks) on both servers, 2026-10-21. Drift Playground, two leads (one slowed to 0.88): both hold the hairpin, 33 - 41 deg through 125 - 165 on 3/3 passes. Spirit Peaks (Miata at ~0.91, S14 Draper at 1.0), 10 passes of 460 - 510: wall push-outs at points 485 - 494 on 4, cte +6.2 - 6.4 m as the drift switches sides into the tire stacks before the straight; the simulator never got there at 0.9 - 1.0. Cause: that server's `spiritpeaksraceway_triangles.csv.gz` was still the drawn model (5.5 MB, 2b4172d1), not sensor_view.py's solid walls and markers (1.1 MB) every policy since gen6 trained with; the simulator with the server's file gives trialQ 5.5% off course and walls 0.4% at 0.9 (with the right file 0.9% and 0.0%). Replaced 2026-10-21 (old kept as `.drawn`) |
| 35 | trialQ on Spirit Peaks with the solid sensor file, 2026-10-22, two leads (1.0 and 0.89 - 0.93), 6 passes of 460 - 510: no wall push-outs anywhere on the lap, at most 4.8 m from the line at 485 - 490 (6.3 m in test 34). Fixed. trialQ goes on both servers and into the repo |

## Sensor cost (2026-10-11)

The first timing lines (gen4, two policy cars, four followers, server at 18 Hz): 26.5 ms of
plugin time a tick, 48% of one core; network 0.18 ms, telemetry 0.10, followers 0.14, sensors
25.9. A copy of the grid in Python, fed the run's own car positions, counted ~217,000 triangle
tests per car per tick: fan 180,000, chassis pins 34,000, wall contact 1,500, ground 1,300. The
cell size came from the whole mesh's bounding box (36.9 m for the 434,106 sensor triangles,
which reach ~2.6 km of scenery); one cell held 16,639 triangles.

Smaller cells alone do not do it: a uniform 8 m grid let the ~100 triangles over 64 m fill up to
64^3 cells each (1.5 M cells). Now triangles up to 16 m (all but 659 sensor and 422 collision
ones) go in 2 m cells, the rest in a coarse grid with the old sizing rule, and a ray takes the
nearer hit of the two. On 4,080 rays from the run (fan, pins, ground, contact) the hits matched
the old grid's exactly; tests per ray fell from 12,380 to 852 on the sensor triangles and from
293 to 77 on the collision ones, ~13x fewer in all. Index: 2.1 M + 2.8 M entries (old 0.5 M +
1.9 M). Measured the same day (gen3, two policy cars, four followers, 18 Hz, 3 min): 2.55 ms a tick, 4.6% of one core (was 26.5 ms, 48%); sensors 2.06 ms (was 25.9), about 1 ms per car as expected; worst tick 5 - 8 ms (was 49 - 51).

## What was removed, and why it could be

Under a direct policy none of the following reaches the car. That was checked two ways: by
reading (two independent reviews of the old code path), and by replaying the seventh server run
through the simulator with all of this state left at its defaults — every position, heading and
speed still agreed to the same 1 cm.

* **Hand-written steering law** — heading error with cross-track fade, yaw damping, cross-track
  correction, yaw-damping conflict scaling.
* **Speed planner** — smoothed angular-rate cap, radius cap, braking-aware drift-entry speed,
  cornering look-ahead.
* **Drift-intent planning** — drift severity, planned slip angle, target yaw rate, countersteer
  control.
* **Gap steering** (`GapFinder`, `GapSteeringCorrector`) and **lateral pin steering**
  (`LateralPinCorrector`). The ray fan and the pins themselves stay: the policy reads them.
* **`UncontrolledSpinRecovery`.** Its tire-saturation signal is true 96% of the time in a held
  drift: with `manji3` on the server it fired 9 times in 3 minutes with the slip angle never
  above 42°. Forced on under `direct2` in the simulator it fired 343 times an hour and cost a
  fifth of the drift time.
* **Residual policies.** `manji3`, `course2` and earlier need the hand-written controller under
  them; this plugin refuses them. They still run on the legacy build.
* The legacy slip-angle look, the kinematic (no lateral dynamics) fallback, reverse spline
  direction, and the 45 settings and about 50 telemetry columns that belonged to the above.

## What is left, and the numbers behind it

* **Vehicle model** — single-track car, linear-then-saturating tires, friction ellipse on the
  rear axle. Parameters in `GradientDriftConfiguration.cs`; the trainer simulates the same
  numbers.
* **Rear sliding grip** — with a flat-topped tire curve a held drift balances at one speed per
  corner radius (about 73 km/h through this course's 30 m corners), whatever the speed limit.
  `RearSlidingGrip` scales the rear force past twice the peak slip angle; the policy is told
  the value (input 35) and trained over a range of it, so it is a setting, not a retrain,
  inside that range. Simulator, policy `slide1`: 0.6 → 59 km/h, 0.8 → 65 km/h, 1.0 → 71 km/h.
  On the server at 0.6 (tests 11 and 12): 59.0 and 59.9 km/h, one-tick replay within 1 cm.
  Policy `slide2` (100M more ticks) covers 0.4 to 1.0, 51 to 73 km/h, with about 83% drift
  time at every setting.
  A policy trained at full grip (`direct3`) spins at 0.9, which is why the plugin holds the
  setting inside the range the policy file records.
* **Wrong-way recovery** — the model holds forward speed while lateral velocity is free, so a
  spin never slows on its own. With no net, 13 of 256 simulated 3-minute runs of `direct2` had
  a spin-out lasting up to 65 s, at up to 1,089 km/h. Triggering on where the car *points*
  (more than 67° off the line for 3 ticks) took that to 0 of 256 at 4.6 recoveries an hour,
  with drift time unchanged. Triggering on slip angle did not help (13 of 256): a spinning car
  holds 50–60° of slip in this model. The largest heading-to-line angle in the server runs was
  58° (`manji3`) and 47.7° (`direct2`). It has not yet fired on the server.
* **Ground-ray ride height** — the sensors are cast from the car's height, so it is part of
  what the policy sees.
* **`NearestPointCorrector`** — floor of 10 m after the bare speed × dt budget fired on 21% of
  ordinary ticks; cap of 60 consecutive corrections after a wedged car was dead-reckoned for 924
  ticks.
* **`GlitchCorrector`** — the line's direction jumps by 0.4 rad or more in one tick at a few
  places on this spline.
* **Telemetry flush every 20 ticks** — a flush per row on the update thread caused hitches.
* **Steering visual** — cosmetic. The encoding was measured in the ninth server test from the
  bytes a human-driven car was sending (2,071 ticks): same sign as the model's steering
  command; road wheels 2 byte units per degree (largest human value 123 units = 61.5°, which
  matches that car's steering geometry); steering wheel 4.27 × the road-wheel offset, pinned
  at ±127 beyond about 15° of road wheel. Sent that way, the bot's steering wheel sat on a
  stop 96.4% of test 13 (the human's: 31.5%) and went end to end 12 times a minute, so the
  plugin spreads the byte over the whole lock instead (its limit at full lock).
* **Drivetrain visual** — cosmetic. Rear wheelspin is what the tire model implies (a rear axle
  keeping a share s of its peak sideways force has a tread slipping lengthwise by
  √(1/s² − 1) times its sideways slip); RPM is rear wheel speed through the gearbox in the
  settings. Compared with six minutes of a human drifting this course (client-side recording,
  2026-09-18), using test 12's telemetry for the bot: driven wheels at a median 92 km/h
  (human 93), 1.58 times the car's speed (human 1.56), RPM 4,313 / 6,441 / 8,009 at the 10th /
  50th / 90th percentile (human 4,328 / 6,606 / 7,876), second gear 93% of the time (human:
  always). The tire diameter (0.54 m) is from the recording: 70.94 RPM per km/h of wheel
  speed in second. The human never changes up. Shift points of 7,800 / 7,000 RPM of wheel
  speed changed gear 30 times a minute; 9,400 / 4,000 still went to third whenever wheelspin
  outran second (5 times in the 79 s of test 16). Now the gear goes by the car's forward
  speed (6,000 up, 2,500 down) and the rear wheels are capped at what the limiter allows in
  that gear: second throughout on the telemetry of tests 15 and 16.
  Second is also the only gear the model's thrust is true for. The model has no engine; its
  thrust limit is the rear tires' (5,816 N at RearSlidingGrip 0.5). From the bot car's data the
  engine has 1.3 – 2 times that in second from 41 km/h of wheel speed up, and 0.8 of it in
  third at the bot's usual 45 km/h of forward speed (2,500 RPM, off boost): in third the car
  bogs and falls out of the drift, which is why the human never changes up. Limiter and tire
  diameter are the bot car's now (8,750 RPM, 0.52 m), not those of the human's recorded car.
  The gear byte and throttle scale are still to be confirmed from what a human's client sends
  the server.
  With the thrust limit (test 13, `angle4`) it reads high: rear wheels at a median 109 km/h,
  1.96 times the car's speed, RPM 7,444 at the median and on the limiter over a tenth of the
  time, because the policy has the rear tires at their thrust limit 77% of the time.
  The encoding is confirmed from what a human's game sent in test 13: log-scaled wheel speed
  bytes, gear byte 2 = first, throttle 0 – 255.
  Front wheels are sent the speed of their axle along the way they point (58 km/h at the
  median of test 13); they had been sent the car's forward speed (44 km/h, the car doing 59).
* **Lights** — the stock AI sets brake lights, hazards, indicators and horn from its own
  spline follower. The plugin clears them and shows brake lights while the policy asks for
  less speed than the car has, held 0.3 s (in test 13: 40 spells of a median 0.61 s; without
  the hold, 57, 13 of them one tick long).
* **Ground attitude and body lean** — cosmetic. Pitch and roll were the racing line's slope
  and camber, for a car metres off the line and at 40° to it: 0.6 – 0.7° off the ground under
  the car's own wheels (standard deviation, tests 15 and 16), where a human's car follows its
  own contact points at 1.01 × their slope. Now four ground rays, under each axle and 0.8 m
  to either side: within 0.13 – 0.16°. Sign and size of the lean are from a recorded human car:
  packet roll = slope across the car + 0.91° per g toward the same side,
  packet pitch = slope along the car + 0.62° per g forward. Scaled to the bot's car by weight ×
  centre-of-gravity height over roll and pitch stiffness, both from the cars' data: 0.85°
  and 0.57° per g.
* **Skid marks** — status bits 1, 2, 4 and 8, which the server has no names for, are "this
  tire is skidding" (front left, front right, rear left, rear right), and clients draw rubber
  from them. The stock AI never sets them: the bot made smoke and left no rubber until the
  plugin did (test 16). A human's game sets a tire's bit while its normalised slip is above
  about 4 times the slip of its peak grip (about 1,380 ticks of a human's flags against the same
  car's own tire values: 83 of 86 bit-on ticks and all but 1 of 5,450 bit-off ticks). The
  plugin uses the same rule: sideways and lengthwise slip together over tan of the peak slip
  angle, above 4.
* **Throttle** — not settled. The game's Python API reads the bot's throttle as 0 on the
  client, but the player hears it on throttle, and that API reads zero for nearly everything
  about a car driven elsewhere.
* **The human, for reference** — same recording: 57.8 km/h and 7.5 m/s² of sideways
  acceleration at the median (bot at RearSlidingGrip 0.6: 60.9 and 8.4), rear slip angle 39°
  at the median and over 50° a third of the time (bot: 32°, never over 40°), steering to the
  full 450° at the wheel.
* **Steering lock** — 28.1° was `STEER_LOCK / STEER_RATIO` from the car's data; the
  human-driven car's road wheels reached 61.5°. A policy trained with the 60° lock carries it
  in its file.
* **Rear thrust limit** — the model set the car's forward speed and held it whatever the car
  was doing. A sliding car loses forward speed at lateral velocity × yaw rate unless it is
  pushed, so that was a free, unlimited push, small at 30° of slip and most of the cornering
  force at 50°. With the 60° lock alone the bot slid at a median 84 km/h with 13.8 m/s² of
  sideways acceleration (2.5 g at the 90th percentile); a human on this course does 57.8 km/h
  and 7.5 m/s². The limit allows only the thrust the friction circle leaves the rear tires
  after their sideways force, plus the front tires' pull along the car. The human recording
  shows that push being used: forward acceleration rises with slip angle the way lateral
  velocity × yaw rate does (0.55 g against 0.55 g at 35 – 50° of rear slip, 0.65 against 0.78
  at 50 – 65°). With it, the same policy family drives at 56 – 59 km/h at
  every `RearSlidingGrip` setting and the setting changes the angle instead (0.7: 28° of mean
  slip, 0.4: 42°).
  Confirmed on the server (test 13): 58.2 km/h against a predicted 58.1, 39.8° of mean slip
  against 39.7°, one-tick replay within 1.0 cm.
* **Steering rate limit** (`MaxSteerRateDegS`, 0 = none; set by the policy file) — the steering
  command was an angle reached within one tick. `angle4` used that to throw the car from one
  drift into the opposite one: a flip (20° of slip one way to 20° the other) took 0.37 s at
  158°/s of rotation with the car's path turning 2.8°, the front wheels steered at a median
  peak of 331°/s and up to 548°/s. A human on this course: 0.59 s, 111°/s, 8.5°, and 63°/s at
  the road wheels of a 28° rack (135°/s on a 60° one). With 150°/s and 22M ticks of training:
  0.45 s, 129°/s, 6.3°, at the same speed and angle. On the server (test 16): 0.44 s, 130°/s,
  5.8°, steering never above 150°/s, one-tick replay within 1.0 cm.
* **Rear brake limit** (`EnableRearBrakeLimit`; set by the policy file) — the thrust limit
  capped pushing only. When a car swings back toward its direction of travel its forward speed
  rises by itself (lateral velocity × yaw rate is positive); the model still let the policy
  hold it down, which in test 13 took more pull-back than the rear tires have on 6.9% of the
  ticks, up to 1.9 g. Now the rear tires can hold the car back with what they could push it
  with, no more. Straight-line braking is unchanged (the 5.6 m/s² cap is below the 5.9 m/s²
  the rear axle has). On the server in test 16: it held speed up on 1.2% of the ticks.
* **Pedal and rear friction circle** (`EnableRearFrictionCircle`; set by the policy file;
  written 2026-10-06, not yet compiled or run) — the rear tires' sideways grip depended on slip
  angle alone, so lifting in a flip did nothing; a human's rear tires go from 0.54 × load at
  full wheelspin to 1.26 with none, and that bite after the crossing is what throws the car
  onto its new side. With the switch on, the policy's second output is a pedal (throttle above
  0, brakes below), forward speed comes from the forces along the car, and the rear's sideways
  force is capped at what the thrust or braking leaves of its circle. Takes the place of the
  speed ramp and both limits. In the simulator (the trainer's lab notebook, "How the car changes
  direction"): rear side force after the crossing 0.78 → 1.14 × load when the policy lifts,
  and the path no longer swings before the car has rotated (`angle5` did that by dropping
  9 km/h going into every flip).
* **Model switches belong to the policy** — `EnableRearThrustLimit`, `EnableRearBrakeLimit`,
  `EnableRearFrictionCircle` and `MaxSteerRateDegS` are now off for a policy whose file does
  not turn them on, whatever `extra_cfg.yml` says. Before, a value set there stayed for a
  policy that had never been trained with it.
* **Wrong-way recovery entry** — 67° for the 28.1° lock (largest heading-to-line angle seen
  on the server: 58°); a policy drifting at 50 – 60° carries 90° in its file.
* **Bot spacing** (`HoldBotSpacing`, on by default; written 2026-10-09, not yet compiled or
  run) — AssettoServer spawns AI cars around a joining player, so several bots start bunched,
  and cars driven by the same policy stay bunched. Each tick, before any car moves, every bot's
  distance along the line is taken from its nearest point; a bot with more room ahead than
  behind has its `RearSlidingGrip` pushed toward the top of the policy's range (faster), the
  other way toward the bottom: target = setting + 2 × (ahead − behind) / (ahead + behind) ×
  half the range, clamped to the range, eased in over 4 s. The policy sees the grip it drives
  with (input 35) and was trained at every value in the range, so nothing it sees is new; only
  a change within a run is (training holds it for an episode). With `lock50b` the range is
  57.5 – 60.7 km/h. A car taken over is also moved to the middle of the largest gap between
  the other bots, unless that spot is within `SpawnSafetyDistanceToPlayerMeters` of a player or
  outside 0.8 × `PlayerRadiusMeters` of the point `PlayerPositionOffsetMeters` ahead of every
  active one (the server's own despawn test; it would take the car back). Simulator, Spirit
  Peaks, `lock50b`, setting 0.5, cars started on top of each other: 3 cars reach gaps of
  221 – 279 m (even: 279) by minute 8; 6 cars hold 96 – 139 m (even: 139) over minutes 15 – 20;
  sliding 88.5 – 89.1% against 88.7% with the grip fixed, walls ≤ 0.01%. The bots still do not
  see each other, so one that spins can be driven through by the one behind. On a lap longer
  than about twice `PlayerRadiusMeters` evenly spread bots are out of the server's range of a
  player and get respawned beside one; set the radius to at least half a lap.
  **First server run (2026-10-09, six bots, 3.5 min):** all six taken over at once as the
  server spawned them, sliding 84 – 88% each, no wall contact; gaps went from 44 – 608 m
  (median of each 20 s, then the largest) to 128 m median, 80 – 249 m, by minute 3.5, still
  closing on even (139 m). No car was placed: the largest gap was the one the player sat in,
  and its middle was within 150 m of him every time. Placement now tries the gaps largest
  first and takes the first whose middle suits the players.

## Behaviour that changed in the cleanup itself

These are the only intended differences from the last legacy build when a direct policy is
loaded:

1. A re-teleported car starts completely afresh. The legacy build kept its old forward speed,
   line angle and ride height across a re-teleport; the simulator never did.
2. A re-teleport seen while the stock AI has the car below 1 m/s is no longer lost.
3. A non-finite policy output repeats the previous action (legacy: the hand-written controller
   drove that tick).
4. No usable policy, or a policy for another track: the stock AI drives, untouched (legacy: the
   hand-written controller drove). Since 2026-10-11 a policy trained on generated courses counts
   as one for every track.
5. `EnableLearnedPolicy` defaults to on.
