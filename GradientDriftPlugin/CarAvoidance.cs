using System;

namespace GradientDriftPlugin;

// AvoidCars: a bot slows for a car it would otherwise run into. The policy sees no other cars
// unless it chases one, so without this only spacing and the pace dial keep bots apart, and a car
// that is not chased (a spun player, a car alongside, a bot whose chase is taken) is invisible.
//
// A velocity-obstacle test (Fiorini and Shiller 1998; reciprocal form van den Berg, Lin and
// Manocha 2008; ORCA, van den Berg, Guy, Lin and Manocha 2011) worked in the AI line's frame, where
// both cars follow the same curves: s along the line, d across it, each car its outline at its own
// angle to the line. The other car moves relative to this one at constant velocity; the first time
// (every TStep up to the horizon) the two outlines, grown by a margin, overlap is the contact time.
// The only control is the bot's pace, so the only answer is to slow down: the largest share k of
// its own velocity (1 down to a floor) that keeps the other car out for the horizon, or the one
// that puts the contact off longest; 1 when slowing would not help. The car behind along the line
// yields (ORCA's split of responsibility, reduced to who can act by slowing).
//
// The same as drift_rl/car_avoid.py, which the plugin checker holds this to.
public static class CarAvoidance
{
    public const float HalfLengthM = 1.95f;     // (CarFrontM + CarRearM) / 2
    public const float HalfWidthM = ChaseInputs.CarHalfWidthM;
    public const float KStep = 0.02f;
    public const float TStep = 0.1f;

    // One car in the line's frame: distance along the line (another car's relative to this one's,
    // wrapped), offset across it, velocity along and across it, and its heading minus the line's.
    public struct LineCar
    {
        public float S, D, Vs, Vd, Theta;
    }

    public static bool Overlap(float ds, float dd, float th0, float th1, float margin)
    {
        float hl = HalfLengthM + 0.5f * margin, hw = HalfWidthM + 0.5f * margin;
        float c0 = MathF.Cos(th0), s0 = MathF.Sin(th0), c1 = MathF.Cos(th1), s1 = MathF.Sin(th1);
        return !Separates(ds, dd, c0, s0, c0, s0, c1, s1, hl, hw)
            && !Separates(ds, dd, -s0, c0, c0, s0, c1, s1, hl, hw)
            && !Separates(ds, dd, c1, s1, c0, s0, c1, s1, hl, hw)
            && !Separates(ds, dd, -s1, c1, c0, s0, c1, s1, hl, hw);
    }

    private static bool Separates(float ds, float dd, float ax, float az, float c0, float s0, float c1, float s1,
        float hl, float hw)
    {
        float r = hl * MathF.Abs(c0 * ax + s0 * az) + hw * MathF.Abs(-s0 * ax + c0 * az)
                + hl * MathF.Abs(c1 * ax + s1 * az) + hw * MathF.Abs(-s1 * ax + c1 * az);
        return MathF.Abs(ds * ax + dd * az) > r;
    }

    // The first time looked at when the other car overlaps this one; +inf when not within the horizon.
    public static float FirstContact(float ds, float dd, float us, float ud, float th0, float th1, float margin,
        float horizon)
    {
        int n = (int)(horizon / TStep + 1e-6f);
        for (int j = 0; j <= n; j++)
        {
            float t = j * TStep;
            if (Overlap(ds + us * t, dd + ud * t, th0, th1, margin))
                return t;
        }
        return float.PositiveInfinity;
    }

    // The share of its own speed (floor..1) the car own drives at to keep clear of other.
    public static float SpeedShare(in LineCar own, in LineCar other, float horizon, float margin, float floor)
    {
        float ds = other.S - own.S, dd = other.D - own.D;
        if (ds <= 0f)
            return 1f;                  // the other car is behind: it yields
        float baseT = FirstContact(ds, dd, other.Vs - own.Vs, other.Vd - own.Vd, own.Theta, other.Theta, margin, horizon);
        if (float.IsPositiveInfinity(baseT))
            return 1f;
        if (baseT == 0f)
            return floor;               // touching already, the other car ahead: back off
        float bestK = 1f, bestT = baseT;
        int steps = (int)((1f - floor) / KStep + 1e-6f);
        for (int i = 1; i <= steps; i++)
        {
            float k = 1f - i * KStep;
            float t = FirstContact(ds, dd, other.Vs - k * own.Vs, other.Vd - k * own.Vd, own.Theta, other.Theta, margin, horizon);
            if (float.IsPositiveInfinity(t))
                return k;
            if (t > bestT)
            {
                bestK = k;
                bestT = t;
            }
        }
        return bestK;
    }
}
