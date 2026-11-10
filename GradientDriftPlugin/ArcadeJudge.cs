using System.Numerics;

namespace GradientDriftPlugin;

// Freestyle (arcade) drift scoring: no zones, no lines on the road. It works on any track. A chain
// starts when the car starts sliding and builds points every moment it slides:
//   points per second = PointsPerSecond x angle x speed x combo x tandem
//   angle   the slip angle over AngleFullDeg (so 45 deg = 1), at most AngleMaxFactor
//   speed   the speed over SpeedRefKmh, at most SpeedMaxFactor
//   combo   starts at 1 and goes up ComboStep with each transition (sliding the other way within
//           the chain), up to ComboMax
//   tandem  up to x(1 + TandemMax) for sliding close behind another moving car: nothing at
//           TandemGapM, full at TandemFullM or closer
// The chain is banked (scored, kept on the board) once the car has not slid for ChainGraceS. A spin
// (slip past SpinDeg), a wall hit, a jump (pits, teleport) or car data that does not add up loses
// it. Like ZoneJudge, it reads only the car status the server receives.
public sealed class ArcadeJudge
{
    public const float SlidingDeg = 15f, MovingKmh = 30f, SpinDeg = 100f;
    public const float PointsPerSecond = 100f;
    public const float AngleFullDeg = 45f, AngleMaxFactor = 1.5f;
    public const float SpeedRefKmh = 80f, SpeedMaxFactor = 1.5f;
    public const float ComboStep = 0.25f, ComboMax = 3f;
    public const float TandemGapM = 10f, TandemFullM = 2f, TandemMax = 1f;
    public const float ChainGraceS = 1.5f;
    public const float MinChainScore = 100f;   // a shorter chain is a wiggle: not told, not kept
    public const float JumpM = ZoneJudge.JumpM;

    private readonly CarDataCheck _check = new();
    private bool _have;
    private Vector2 _lastPosition;
    private int _lastSign;
    private float _graceS, _slideS, _angleSum, _speedSum, _tandemS;

    public List<ZoneJudge.Event> Events { get; } = new();
    public ZoneJudge.RunResult? LastRun { get; private set; }

    // What the HUD shows: whether a chain is going, its points, combo and transitions, and the car now.
    public bool Chaining { get; private set; }
    public float ChainPoints { get; private set; }
    public float Combo { get; private set; } = 1f;
    public int Transitions { get; private set; }
    public float AngleNowDeg { get; private set; }
    public float SpeedNowKmh { get; private set; }
    public float TandemNow { get; private set; }
    public float Multiplier => Combo * (1f + TandemMax * TandemNow);

    // One look at the car: its middle (x, z), heading (radians, x toward z), velocity, seconds since
    // the last look, how far behind another moving car it is (infinity: none ahead) and wall hits since.
    public void Step(float x, float z, float heading, float vx, float vz, float dt, float gapAheadM = float.PositiveInfinity, int wallHits = 0)
    {
        var position = new Vector2(x, z);
        var velocity = new Vector2(vx, vz);
        float speed = velocity.Length();
        SpeedNowKmh = speed * 3.6f;
        float slip = 0f;
        int sign = 0;
        if (speed > 0.5f)
        {
            var forward = new Vector2(MathF.Cos(heading), MathF.Sin(heading));
            slip = MathF.Acos(Math.Clamp(Vector2.Dot(forward, velocity) / speed, -1f, 1f)) * 180f / MathF.PI;
            sign = forward.X * velocity.Y - forward.Y * velocity.X >= 0f ? 1 : -1;
        }
        AngleNowDeg = slip;
        TandemNow = Math.Clamp((TandemGapM - gapAheadM) / (TandemGapM - TandemFullM), 0f, 1f);

        bool jumped = _have && Vector2.Distance(position, _lastPosition) > JumpM;
        _have = true;
        _lastPosition = position;
        if (jumped)
        {
            Lose(ZoneJudge.Kind.Ended, "left the track");
            _check.Reset();
            return;
        }
        if (!_check.Ok(position, speed, dt))
        {
            Lose(ZoneJudge.Kind.Ended, "car data does not add up");
            return;
        }
        if (wallHits > 0)
        {
            Lose(ZoneJudge.Kind.Wall, "wall");
            return;
        }
        if (Chaining && SpeedNowKmh > 10f && slip > SpinDeg)
        {
            Lose(ZoneJudge.Kind.Spin, "spin");
            return;
        }

        bool sliding = SpeedNowKmh >= MovingKmh && slip >= SlidingDeg && slip <= SpinDeg;
        if (!sliding)
        {
            if (Chaining)
            {
                _graceS += dt;
                if (_graceS > ChainGraceS) Bank();
            }
            return;
        }
        if (!Chaining)
        {
            Chaining = true;
            ChainPoints = 0f;
            Combo = 1f;
            Transitions = 0;
            _slideS = _angleSum = _speedSum = _tandemS = 0f;
        }
        else if (sign != _lastSign)
        {
            Transitions++;
            float before = Combo;
            Combo = MathF.Min(ComboMax, Combo + ComboStep);
            Events.Add(new ZoneJudge.Event(ZoneJudge.Kind.Combo, 0, Combo, 0f, slip, SpeedNowKmh / SpeedRefKmh, TandemNow,
                Combo > before ? "transition" : "transition (combo full)"));
        }
        _lastSign = sign;
        _graceS = 0f;
        float angle = MathF.Min(AngleMaxFactor, slip / AngleFullDeg);
        float pace = MathF.Min(SpeedMaxFactor, SpeedNowKmh / SpeedRefKmh);
        ChainPoints += PointsPerSecond * angle * pace * Multiplier * dt;
        _slideS += dt;
        _angleSum += slip * dt;
        _speedSum += SpeedNowKmh / SpeedRefKmh * dt;
        if (TandemNow > 0f) _tandemS += dt;
    }

    private void Bank()
    {
        float score = ChainPoints;
        Chaining = false;
        if (score < MinChainScore)
        {
            return;
        }
        float slide = MathF.Max(_slideS, 1e-6f);
        // the board's columns: score, mean angle, mean speed over SpeedRefKmh, and (in place of a
        // zone fill) the share of the chain spent close behind another car
        var result = new ZoneJudge.RunResult(score, Array.Empty<float>(), _angleSum / slide, _speedSum / slide, _tandemS / slide, 0);
        LastRun = result;
        Events.Add(new ZoneJudge.Event(ZoneJudge.Kind.Run, 0, score, result.MeanFill, result.MeanAngleDeg,
            result.MeanSpeedOfPace, result.MeanFill, $"x{Combo:0.00}, {Transitions} transitions"));
    }

    private void Lose(ZoneJudge.Kind kind, string why)
    {
        if (Chaining && ChainPoints >= MinChainScore)
        {
            Events.Add(new ZoneJudge.Event(kind, 0, -ChainPoints, 0f, AngleNowDeg, SpeedNowKmh / SpeedRefKmh, 0f, why));
        }
        Chaining = false;
        ChainPoints = 0f;
        Combo = 1f;
        Transitions = 0;
        _graceS = 0f;
    }
}
