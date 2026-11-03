---
name: tune-drift-bots
description: Tune Gradient Drift bots on the user's own AssettoServer. Use when they want their drift bots faster, calmer, cleaner, chasing better, more varied, or fixed after a bad session, or ask why the bots aren't driving.
---

# Tune Gradient Drift bots

You are helping an AssettoServer owner fit the Gradient Drift bots to their server. Nothing on the
server learns: tuning means changing the plugin's settings in `cfg/extra_cfg.yml`, restarting, and
reading how the bots drove. Work in small, reversible steps and let the owner judge the driving.

The settings reference is `GradientDriftPlugin/SETTINGS.md` in the gradient-drift repo
(https://github.com/toyowheelin/gradient-drift). Read it before suggesting a value, and use only
the **Operator settings**. Never change a **Developer setting**: the policy only drives the car and
sensors it was trained with, so those make the bots worse, not different.

## 1. Find the server

Ask for, or find, the server folder (the one with `cfg/extra_cfg.yml`, `cfg/server_cfg.ini`,
`cfg/entry_list.ini` and `plugins/GradientDriftPlugin/`). If the owner runs several servers, work on
one at a time and say which. If you can't reach their files (for example in a chat with no folder
access), ask them to paste the files and logs below and give them the edits to make by hand.

## 2. Check the bots are really driving

Read the newest server log (`logs/` in the server folder) for lines starting `GradientDrift:`.

- `policy loaded from ... N training ticks` means the plugin found its policy.
- Any warning ending `- the stock AI drives` means the plugin stood aside. Fix that first: it is
  almost always missing track files or a plugin/policy mismatch. The cure is to run the
  Gradient Drift setup program again for this track and its cars (it writes the track data, the car
  files and the settings, with backups and undo). Don't hand-make track files.
- `built and tested for AssettoServer X, this server runs Y` means they need the plugin release made
  for their AssettoServer version.
- `no usable car file for <model>` means setup hasn't been run for that car.
- `X is A, but this policy was trained for L to H - using U` means a setting is outside the
  policy's range and was clamped. Set it inside the range so the config says what really happens.

## 3. Get a scorecard

Scorecards are the measuring stick. If `EnableScorecard` isn't `true`, offer to turn it on (it writes
a few KB a session into `gradientdrift_scorecards/` in the server folder). Only time with a player on
counts, so the owner needs to drive or host a session; 15 minutes or more gives steadier numbers.
With Python available, `python drift_rl/scorecard_report.py <server>/gradientdrift_scorecards` puts
sessions side by side. Otherwise read the `.txt` files: its header explains every column.

Ask the owner what they want before reading too much into numbers: "more angle", "less crashing
into players", "tandems with me", "they all look the same". Their eye wins over any metric.

## 4. Change one thing

Map what they see to the smallest change (check names and ranges in SETTINGS.md):

| They see / scorecard shows | Try |
|---|---|
| Bots too fast, run wide, many wall hits | `BotPacePercent` down 5 (e.g. 100 to 95) |
| Bots too tame, little slide | `BotPacePercent` up 5, or `BotSkill` toward 1 |
| Want beginner-friendly bots | `BotSkill` down (0.6 to 0.8), maybe `BotPacePercent` 90 |
| Bots slide less with several out | `BotPaceMinPercent` back toward 85 |
| Bots hit players they chase, gap p50 too short | `ChaseGapMeters` up 0.5 to 1, or `ChaseTimeGapSeconds` up 0.1 |
| Chase gap too loose, band share low | `ChaseGapMeters` down, `ChaseResponseSeconds` down 0.1 |
| Bots won't join a tandem | `ChaseOptInMeters` up (10 to 12) |
| Too many bot trains / bots should stay spread | `ChaseBots: false`, or `ChaseTrainMax` down |
| Bots drive off and leave the player | `LeadPlayers: true` |
| All bots look the same | `BotStyleSpread` 0.4 to 0.7, `BotSkillSpread` 0.1 to 0.2, or named `BotStyles` |
| Bots should match the regulars' pace | `AdaptToPlayers: true`; lower `AdaptToPlayersStrength` if they run wide |
| Bots sound quiet | `EnableCustomUpdate: true` in AssettoServer's own settings |
| High `stock` share or `stuck` count | not a tuning problem: check the log (step 2) and the track files |

Then:

1. **Back up first.** Copy `cfg/extra_cfg.yml` to `cfg/extra_cfg.yml.before_<change>` before
   editing. Never overwrite an existing backup.
2. Edit only inside the `--- !GradientDriftConfiguration` block. Keep the YAML indentation and the
   rest of the file exactly as it was. Leave AssettoServer's own settings alone unless the owner
   agrees to a specific change.
3. Show the owner the change (old value, new value, why) before saving.
4. The plugin reads settings only at start: ask the owner to restart the server, then check the new
   log for the `GradientDrift:` lines again.
5. After their next session, compare its scorecard with the one before. Keep the change if it helped,
   otherwise put the backup back.

One change per session keeps cause and effect clear. Keep a short note of what was tried in
`gradientdrift_scorecards/TUNING.md` (date, setting, old -> new, result), so the next session, or
the next Claude, picks up where this one left off.

## Limits to tell the owner about

- Tuning moves the bots within what the policy learned. It can't teach a new trick, such as a deeper
  angle than the policy holds, a front-drive car, or a track it stands aside on. That needs a new
  policy: training is covered in the repo's README ("Training your own policy") and needs a PC with
  a decent GPU and a few hours.
- Rear-drive cars only. Linux servers are untested.
- Don't run the server's restart or stop it yourself unless the owner asks: players may be on.
