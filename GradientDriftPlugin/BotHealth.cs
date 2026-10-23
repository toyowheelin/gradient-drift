using System.Numerics;
using AssettoServer.Server.Ai;
using Serilog;

namespace GradientDriftPlugin;

// Watches the bots for trouble, so the server log can stay quiet while they drive and say
// something only when one is not: every SampleMs each bot the policy drives (train leads; a
// follower replays its lead) is checked for having moved. A bot that has not moved for
// StillSamples samples in a row gets one warning - whether the policy has it (pinned against a
// wall, spun) or the server's own AI is holding it stopped - and a line when it moves again. At
// most one warning a bot every RepeatMs; a bot still stuck when that is up gets it then, and the
// episodes in between are counted in it.
internal sealed class BotHealth
{
    private const long SampleMs = 10_000;
    private const int StillSamples = 3;
    private const float MinMoveM = 10f;
    private const long RepeatMs = 600_000;

    private sealed class Bot
    {
        public Vector3 Position;
        public byte SpawnCounter;
        public int Still;
        public bool Warned;
        public long SeenMs;
    }

    private readonly Func<AiState, bool> _policyHasIt;
    private readonly Func<Vector3, int> _pointAt;
    private readonly Dictionary<AiState, Bot> _bots = new();
    // kept when a bot is taken away, so one the server keeps respawning into the same trouble is
    // still warned about at most every RepeatMs: when it was last, and the episodes since
    private readonly Dictionary<AiState, (long Ms, int Unreported)> _warnings = new();
    private readonly List<AiState> _gone = new();
    private long _lastSampleMs = -1;

    // Whether the policy has a bot (else the server's AI is holding it), and the AI-line point
    // a position is nearest.
    public BotHealth(Func<AiState, bool> policyHasIt, Func<Vector3, int> pointAt)
    {
        _policyHasIt = policyHasIt;
        _pointAt = pointAt;
    }

    // Once a tick, with the bots driven this tick.
    public void Check(List<(byte SessionId, AiState State)> driven, long nowMs)
    {
        if (_lastSampleMs >= 0 && nowMs - _lastSampleMs < SampleMs)
        {
            return;
        }
        _lastSampleMs = nowMs;

        foreach (var (sessionId, state) in driven)
        {
            Vector3 position = state.Status.Position;
            if (!_bots.TryGetValue(state, out Bot? bot) || bot.SpawnCounter != state.SpawnCounter)
            {
                // new, or placed somewhere else by the server: start watching it afresh
                _bots[state] = new Bot { Position = position, SpawnCounter = state.SpawnCounter, SeenMs = nowMs };
                continue;
            }
            bot.SeenMs = nowMs;
            float moved = new Vector2(position.X - bot.Position.X, position.Z - bot.Position.Z).Length();
            bot.Position = position;
            if (moved >= MinMoveM)
            {
                if (bot.Warned)
                {
                    Log.Information("GradientDrift: bot {SessionId} is moving again", sessionId);
                }
                bot.Still = 0;
                bot.Warned = false;
                continue;
            }

            bot.Still++;
            if (bot.Warned || bot.Still < StillSamples)
            {
                continue;
            }
            (long lastMs, int unreported) = _warnings.TryGetValue(state, out var last) ? last : (long.MinValue / 2, 0);
            if (nowMs - lastMs < RepeatMs)
            {
                if (bot.Still == StillSamples)
                {
                    _warnings[state] = (lastMs, unreported + 1);
                }
                continue;
            }
            string more = unreported > 0 ? $" ({unreported} more times since the last warning)" : "";
            long seconds = bot.Still * SampleMs / 1000;
            if (_policyHasIt(state))
            {
                Log.Warning("GradientDrift: bot {SessionId} has been stuck for {Seconds} s near AI-line point {Point}{More:l}: the policy is driving it but it is not getting anywhere (against a wall? spun?)",
                    sessionId, seconds, _pointAt(position), more);
            }
            else
            {
                Log.Warning("GradientDrift: bot {SessionId} has stood still for {Seconds} s near AI-line point {Point}{More:l}: the server's own AI is holding it stopped, so the policy is not driving it (a car in the way?)",
                    sessionId, seconds, _pointAt(position), more);
            }
            bot.Warned = true;
            _warnings[state] = (nowMs, 0);
        }

        // bots the server has taken away
        _gone.Clear();
        foreach (var (state, bot) in _bots)
        {
            if (bot.SeenMs != nowMs)
            {
                _gone.Add(state);
            }
        }
        foreach (AiState state in _gone)
        {
            _bots.Remove(state);
        }
    }
}
