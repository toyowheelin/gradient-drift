using System.Numerics;
using AssettoServer.Shared.Model;
using AssettoServer.Shared.Network.Packets.Outgoing;

namespace GradientDriftPlugin;

// Tandem trains (TrainSize > 1): the first bot of a train is driven by the policy; each follower
// is sent what the lead car sent a moment earlier - its position, rotation, velocity, steering,
// wheel speeds, engine, gear, throttle and lights - so it drives the lead's line exactly, behind
// it. No policy, no physics: the followers cannot crash on their own, and any car can follow.
//
// The delay is TandemGapSeconds per place in the train, varied slowly (TandemGapVariation) and
// never closer than TandemMinGapMeters to the car in front (a fixed delay closes up where the
// lead slows, as a real chase does, but not into it). Each follower also sits a little to one
// side of the lead's line (TandemLateralOffsetMeters), drifting in and out, so it reads as a
// chase rather than a ghost in the lead's tyre tracks.

// One tick of what a lead car sent to clients.
internal struct TrainSample
{
    public long TimeMs;
    public Vector3 Position;
    public Vector3 Rotation;
    public Vector3 Velocity;
    public byte SteerAngle;
    public byte WheelAngle;
    public ushort EngineRpm;
    public byte Gear;
    public byte Gas;
    public byte Tyre0, Tyre1, Tyre2, Tyre3;
    public CarStatusFlags StatusFlag;

    public static TrainSample From(CarStatus s, long timeMs) => new()
    {
        TimeMs = timeMs,
        Position = s.Position,
        Rotation = s.Rotation,
        Velocity = s.Velocity,
        SteerAngle = s.SteerAngle,
        WheelAngle = s.WheelAngle,
        EngineRpm = s.EngineRpm,
        Gear = s.Gear,
        Gas = s.Gas,
        Tyre0 = s.TyreAngularSpeed[0],
        Tyre1 = s.TyreAngularSpeed[1],
        Tyre2 = s.TyreAngularSpeed[2],
        Tyre3 = s.TyreAngularSpeed[3],
        StatusFlag = s.StatusFlag,
    };

    public readonly void WriteTo(CarStatus s, Vector3 position)
    {
        s.Position = position;
        s.Rotation = Rotation;
        s.Velocity = Velocity;
        s.SteerAngle = SteerAngle;
        s.WheelAngle = WheelAngle;
        s.EngineRpm = EngineRpm;
        s.Gear = Gear;
        s.Gas = Gas;
        s.TyreAngularSpeed[0] = Tyre0;
        s.TyreAngularSpeed[1] = Tyre1;
        s.TyreAngularSpeed[2] = Tyre2;
        s.TyreAngularSpeed[3] = Tyre3;
        s.StatusFlag = StatusFlag;
    }
}

// The last HistorySeconds of one lead car, oldest first.
internal sealed class TrainHistory
{
    private const int Capacity = 2048;          // 30 s at the server's 18 Hz with room to spare
    private readonly TrainSample[] _samples = new TrainSample[Capacity];
    private int _start;
    private int _count;

    public byte SpawnCounter;
    public DriftCarState? Owner;        // the lead's driving state the samples came from

    public int Count => _count;

    public void Clear() => _count = 0;

    public void Add(in TrainSample sample)
    {
        if (_count == Capacity)
        {
            _start = (_start + 1) % Capacity;
            _count--;
        }
        _samples[(_start + _count) % Capacity] = sample;
        _count++;
    }

    private ref readonly TrainSample At(int i) => ref _samples[(_start + i) % Capacity];

    public long OldestMs => _count > 0 ? At(0).TimeMs : long.MaxValue;

    // The lead's state at timeMs, between the two samples either side of it; false if the history
    // does not reach back that far.
    public bool TryAt(long timeMs, out TrainSample sample, out Vector3 position)
    {
        sample = default;
        position = default;
        if (_count == 0 || timeMs < At(0).TimeMs)
        {
            return false;
        }
        int lo = 0;
        int hi = _count - 1;
        if (timeMs >= At(hi).TimeMs)
        {
            sample = At(hi);
            position = sample.Position;
            return true;
        }
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (At(mid).TimeMs <= timeMs)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        ref readonly TrainSample a = ref At(lo);
        ref readonly TrainSample b = ref At(hi);
        float t = b.TimeMs > a.TimeMs ? (float)(timeMs - a.TimeMs) / (b.TimeMs - a.TimeMs) : 0f;
        sample = t < 0.5f ? a : b;              // the visual bytes and lights from the nearer one
        position = Vector3.Lerp(a.Position, b.Position, t);
        sample.Velocity = Vector3.Lerp(a.Velocity, b.Velocity, t);
        sample.Rotation = new Vector3(LerpAngle(a.Rotation.X, b.Rotation.X, t),
            LerpAngle(a.Rotation.Y, b.Rotation.Y, t), LerpAngle(a.Rotation.Z, b.Rotation.Z, t));
        return true;
    }

    private static float LerpAngle(float a, float b, float t)
    {
        float d = MathF.IEEERemainder(b - a, 2f * MathF.PI);
        return a + d * t;
    }
}

internal static class TandemFollower
{
    // Slow, smooth wandering in [-1, 1] that differs per car: two sines with periods that do not
    // line up, phases from the car's session id.
    public static float Wander(int seed, float timeS, float periodA, float periodB)
    {
        float phaseA = (seed * 2.399963f) % (2f * MathF.PI);
        float phaseB = (seed * 1.618034f + 1.3f) % (2f * MathF.PI);
        return 0.6f * MathF.Sin(2f * MathF.PI * timeS / periodA + phaseA)
            + 0.4f * MathF.Sin(2f * MathF.PI * timeS / periodB + phaseB);
    }

    // Sideways from a sample's direction of travel (its heading when barely moving).
    public static Vector3 Side(in TrainSample s)
    {
        var travel = new Vector2(s.Velocity.X, s.Velocity.Z);
        if (travel.LengthSquared() < 0.25f)
        {
            float heading = s.Rotation.X + MathF.PI / 2f;
            travel = new Vector2(MathF.Cos(heading), MathF.Sin(heading));
        }
        travel = Vector2.Normalize(travel);
        return new Vector3(-travel.Y, 0f, travel.X);
    }
}
