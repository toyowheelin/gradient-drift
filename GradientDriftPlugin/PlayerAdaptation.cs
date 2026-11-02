using System;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("PluginChecks")]

namespace GradientDriftPlugin;

// AdaptToPlayers: the bots fit in with the players drifting with them, for as long as they are on.
// No learning: the plugin watches the players and moves two numbers the policy already reads,
// each bot's pace and the gap it is asked to hold when chasing, toward what the players do,
// inside the range the policy was trained over. When the server empties they ease back to the
// settings and the averages are dropped, so the next group starts fresh.
//
// Only drifting players on the line count: sliding between MinSlipDeg and MaxSlipDeg (not spinning),
// within MaxOffLineMeters of the AI line, driving its way above MinSpeedMs. A gap counts only while
// a player follows a car closely at about its speed (a tandem, not a pass or a crash).
//
//   pace  the player's speed along the line / the speed a pace of 1.0 asks for there
//   gap   the player's outline gap to the car ahead less ChaseTimeGapSeconds x speed: the
//         ChaseGapMeters that would ask a bot for that gap at that speed
internal sealed class PlayerAdaptation
{
    public const float MinSlipDeg = 12f, MaxSlipDeg = 75f;
    public const float MaxOffLineMeters = 5f;
    public const float MinSpeedMs = 7f;
    public const float GapMaxMeters = 15f;          // further back than this is not a tandem
    public const float GapSpeedMatchMs = 3f;        // closing faster than this: a pass or a catch-up
    public const float PaceLow = 0.5f, PaceHigh = 1.5f;   // a sample outside these is thrown away

    private readonly float _strength, _learnS, _forgetS, _averageS;

    private float _pace = float.NaN, _gap = float.NaN;
    private float _paceConfidence, _gapConfidence;     // 0 - 1
    private float _paceSum, _gapSum;
    private int _paceCount, _gapCount;

    public PlayerAdaptation(float strength, float learnSeconds, float forgetSeconds, float averageSeconds)
    {
        _strength = Math.Clamp(strength, 0f, 1f);
        _learnS = MathF.Max(1f, learnSeconds);
        _forgetS = MathF.Max(1f, forgetSeconds);
        _averageS = MathF.Max(1f, averageSeconds);
    }

    // the players' average pace and chase ask (NaN before any sample), and how far the bots are
    // moved to them now (0 - 1: AdaptToPlayersStrength x how much has been seen)
    public float Pace => _pace;
    public float Gap => _gap;
    public float PaceWeight => _strength * _paceConfidence;
    public float GapWeight => _strength * _gapConfidence;
    public int PaceSamplesThisTick => _paceCount;

    public static bool Drifting(float slipRad)
    {
        float deg = MathF.Abs(slipRad) * 180f / MathF.PI;
        return deg >= MinSlipDeg && deg <= MaxSlipDeg;
    }

    public void PaceSample(float pace)
    {
        if (pace >= PaceLow && pace <= PaceHigh)
        {
            _paceSum += pace;
            _paceCount++;
        }
    }

    public void GapSample(float ask)
    {
        if (float.IsFinite(ask))
        {
            _gapSum += ask;
            _gapCount++;
        }
    }

    // Once a tick, after the samples. playersOn: any player connected. Returns true when the
    // averages were just forgotten (the server emptied and the bots are back on the settings).
    public bool EndTick(float dt, bool playersOn)
    {
        bool forgotten = false;
        float ease = Math.Min(1f, dt / _averageS);
        if (_paceCount > 0)
        {
            float x = _paceSum / _paceCount;
            _pace = float.IsNaN(_pace) ? x : _pace + (x - _pace) * ease;
            _paceConfidence = Math.Min(1f, _paceConfidence + dt / _learnS);
        }
        if (_gapCount > 0)
        {
            float x = _gapSum / _gapCount;
            _gap = float.IsNaN(_gap) ? x : _gap + (x - _gap) * ease;
            _gapConfidence = Math.Min(1f, _gapConfidence + dt / _learnS);
        }
        if (!playersOn)
        {
            bool had = _paceConfidence > 0f || _gapConfidence > 0f;
            _paceConfidence = MathF.Max(0f, _paceConfidence - dt / _forgetS);
            _gapConfidence = MathF.Max(0f, _gapConfidence - dt / _forgetS);
            if (had && _paceConfidence == 0f && _gapConfidence == 0f)
            {
                _pace = _gap = float.NaN;
                forgotten = true;
            }
        }
        _paceSum = _gapSum = 0f;
        _paceCount = _gapCount = 0;
        return forgotten;
    }

    // A bot's style moved toward the players. Each number shifts by the same amount for every bot
    // (the players' value less the setting), so BotStyles' differences between bots stay.
    // basePace / baseGap: the server's settings (BotPacePercent, ChaseGapMeters); the low / high
    // pairs: the policy's trained ranges.
    public BotStyle Apply(in BotStyle style, float basePace, float paceLow, float paceHigh,
        float baseGap, float gapLow, float gapHigh)
    {
        float pace = style.PaceMiddle;
        if (!float.IsNaN(_pace) && basePace > 0f && PaceWeight > 0f)
        {
            pace = Math.Clamp(pace + PaceWeight * (_pace - basePace), paceLow, paceHigh);
        }
        float gap = style.ChaseGapMeters;
        if (!float.IsNaN(_gap) && GapWeight > 0f)
        {
            gap = Math.Clamp(gap + GapWeight * (_gap - baseGap), gapLow, gapHigh);
        }
        return style with { PaceMiddle = pace, ChaseGapMeters = gap };
    }
}
