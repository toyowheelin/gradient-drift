# Tune your bots with Claude

This folder is a Claude Code plugin with one skill, `tune-drift-bots`. It lets Claude tune the
Gradient Drift bots to your own server: it checks the plugin's log, reads the bots' scorecards, and
changes `extra_cfg.yml` one setting at a time, with a backup each time. You judge the driving.

## Install

In Claude Code, on the machine that runs your server (or one that can see its folder):

```
/plugin marketplace add toyowheelin/gradient-drift
/plugin install gradient-drift-tuning@gradient-drift
```

Then start Claude Code in (or give it access to) your server folder and say something like
"tune my drift bots, they hit the wall too much".

On claude.ai or the Claude desktop app, you can add the skill instead: zip the
`skills/tune-drift-bots` folder and upload it as a skill in your Claude settings. Without access
to your server's files, Claude will ask you to paste the log and scorecard and give you the edits
to make.

Turn on `EnableScorecard` in the plugin settings (Claude offers to) so there is something to measure.
Settings are explained in [SETTINGS.md](../GradientDriftPlugin/SETTINGS.md).

This plugin is part of Gradient Drift and is not made, affiliated with or endorsed by Anthropic.
Claude and Claude Code are Anthropic's products; you use them under your own account and its terms.
