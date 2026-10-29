// The chase inputs of a tandem policy (drift_rl/drift_env.py CHASE_OBS_SLOT .. the lead's controls):
// what a bot chasing a player sees of the car ahead. drift_env._chase_obs is the reference and Fill()
// below is a line-for-line port of it for a lead that hands over no plan (a player): its path ahead
// is predicted from its late state, as in training. Checked against _chase_obs on 300 recorded-lead
// cases: equal to float rounding, except the lead's path ahead, whose speed is measured over
// LineSpeedSeconds here and over CHASE_LEAD_SPEED_TICKS samples there (about 0.5 s either way).
// Change either only together with a retrain.
//
// The lead in training is a recorded run seen CHASE_DELAY_S late; here it is the player's car status,
// which is late by itself (its client's update rate and ping). Each new status is one LeadSample.

using System;

namespace GradientDriftPlugin;

// One state of the car ahead: server seconds it was first seen, where it is (x, z), heading
// (atan2(Z, X)), body slip (heading less the direction of travel, fast_sim's sign), yaw rate (rad/s),
// speed (m/s), place along the line (m) and offset from it (m, + = left), road-wheel steer (rad),
// throttle and brake (0 - 1).
public struct LeadSample
{
    public float TimeS;
    public float X, Z, Heading, Slip, Yaw, Speed, Arc, Cte, Steer, Pedal, Brake;
}

// A player's recent states, oldest first.
public sealed class LeadTrack
{
    public const int Capacity = 512;
    private readonly LeadSample[] _samples = new LeadSample[Capacity];
    private int _start;

    public int Count { get; private set; }

    public void Clear()
    {
        _start = 0;
        Count = 0;
    }

    public void Add(in LeadSample sample)
    {
        if (Count == Capacity)
        {
            _samples[_start] = sample;
            _start = (_start + 1) % Capacity;
        }
        else
        {
            _samples[(_start + Count) % Capacity] = sample;
            Count++;
        }
    }

    public ref readonly LeadSample this[int i] => ref _samples[(_start + i) % Capacity];
    public ref readonly LeadSample Latest => ref this[Count - 1];

    // the newest sample at least seconds older than sample k (0 if there is none that old)
    public int Back(int k, float seconds)
    {
        float t = this[k].TimeS - seconds + 1e-3f;
        int j = k;
        while (j > 0 && this[j].TimeS > t)
            j--;
        return j;
    }
}

public static class ChaseInputs
{
    public const int Slot = 67;                 // drift_env.CHASE_OBS_SLOT
    public const int Count = 13 + 10 + 4;       // N_CHASE_OBS + N_PLAN_OBS + N_LEAD_CTRL_OBS
    public const int PlanSlot = Slot + 13;
    public const int ControlSlot = PlanSlot + 10;
    public const float CarFrontM = 1.80f;       // CAR_FRONT_M, CAR_REAR_M, CAR_HALF_WIDTH_M
    public const float CarRearM = 2.10f;
    public const float CarHalfWidthM = 0.85f;
    public const float GapLowM = 0.5f;          // CHASE_GAP_M: the asked gap at chase_gap 0 and 1
    public const float GapHighM = 6.0f;
    public const float LineSpeedSeconds = 0.5f; // CHASE_LEAD_SPEED_TICKS x 0.05 s
    public const float AccelSeconds = 0.25f;    // LEAD_ACCEL_TICKS x 0.05 s
    public const int HistorySamples = 400;      // CHASE_HISTORY
    private static readonly float[] PathAheadS = { 0.5f, 1.0f, 2.0f };

    // chase_gap: the asked gap (m) as the 0 - 1 the network reads
    public static float AskInput(float gapM) => Math.Clamp((gapM - GapLowM) / (GapHighM - GapLowM), 0f, 1f);

    public static float WrapArc(float d, float lap)
    {
        if (d > 0.5f * lap) return d - lap;
        if (d < -0.5f * lap) return d + lap;
        return d;
    }

    private static float WrapPi(float angle)
    {
        float shifted = (angle + MathF.PI) % (2f * MathF.PI);
        if (shifted < 0f)
            shifted += 2f * MathF.PI;
        return shifted - MathF.PI;
    }

    // the lead's speed along the line over the last LineSpeedSeconds (m/s)
    public static float LineSpeed(LeadTrack lead, float lap)
    {
        int k = lead.Count - 1;
        int k0 = lead.Back(k, LineSpeedSeconds);
        if (k <= k0)
            return lead[k].Speed;
        return WrapArc(lead[k].Arc - lead[k0].Arc, lap) / MathF.Max(lead[k].TimeS - lead[k0].TimeS, 1e-3f);
    }

    // the lead's acceleration over the last AccelSeconds (m/s^2), drift_env._lead_accel
    public static float Accel(LeadTrack lead)
    {
        int k = lead.Count - 1;
        int k0 = lead.Back(k, AccelSeconds);
        if (k <= k0)
            return 0f;
        return (lead[k].Speed - lead[k0].Speed) / MathF.Max(lead[k].TimeS - lead[k0].TimeS, 1e-3f);
    }

    // The gap between the two car bodies' outlines, negative once this car is the further along the
    // line (drift_env._outline_gap).
    public static float OutlineGap(float x, float z, float h, float arc, float lx, float lz, float lh, float larc, float lap)
    {
        float dist = OutlineDistance(x, z, h, lx, lz, lh);
        return WrapArc(larc - arc, lap) >= 0f ? dist : -dist;
    }

    public static bool CarsOverlap(float x1, float z1, float h1, float x2, float z2, float h2)
    {
        float halfLen = 0.5f * (CarFrontM + CarRearM);
        float shift = 0.5f * (CarFrontM - CarRearM);
        float c1x = x1 + shift * MathF.Cos(h1), c1z = z1 + shift * MathF.Sin(h1);
        float c2x = x2 + shift * MathF.Cos(h2), c2z = z2 + shift * MathF.Sin(h2);
        for (int a = 0; a < 4; a++)
        {
            float h = a < 2 ? h1 : h2;
            float ux = a % 2 == 0 ? MathF.Cos(h) : -MathF.Sin(h);
            float uz = a % 2 == 0 ? MathF.Sin(h) : MathF.Cos(h);
            float r1 = halfLen * MathF.Abs(MathF.Cos(h1) * ux + MathF.Sin(h1) * uz) + CarHalfWidthM * MathF.Abs(MathF.Cos(h1) * uz - MathF.Sin(h1) * ux);
            float r2 = halfLen * MathF.Abs(MathF.Cos(h2) * ux + MathF.Sin(h2) * uz) + CarHalfWidthM * MathF.Abs(MathF.Cos(h2) * uz - MathF.Sin(h2) * ux);
            if (MathF.Abs((c2x - c1x) * ux + (c2z - c1z) * uz) > r1 + r2)
                return false;
        }
        return true;
    }

    private static (float X, float Z) Corner(float x, float z, float h, int k)
    {
        float f = k < 2 ? CarFrontM : -CarRearM;
        float w = k == 0 || k == 3 ? CarHalfWidthM : -CarHalfWidthM;
        float c = MathF.Cos(h), s = MathF.Sin(h);
        return (x + f * c - w * s, z + f * s + w * c);
    }

    public static float OutlineDistance(float x1, float z1, float h1, float x2, float z2, float h2)
    {
        if (CarsOverlap(x1, z1, h1, x2, z2, h2))
            return 0f;
        float best = 1e9f;
        for (int side = 0; side < 2; side++)
        {
            (float xa, float za, float ha, float xb, float zb, float hb) = side == 0
                ? (x1, z1, h1, x2, z2, h2) : (x2, z2, h2, x1, z1, h1);
            for (int k = 0; k < 4; k++)
            {
                var p = Corner(xa, za, ha, k);
                for (int e = 0; e < 4; e++)
                {
                    var a = Corner(xb, zb, hb, e);
                    var b = Corner(xb, zb, hb, (e + 1) % 4);
                    float ex = b.X - a.X, ez = b.Z - a.Z;
                    float t = Math.Clamp(((p.X - a.X) * ex + (p.Z - a.Z) * ez) / MathF.Max(ex * ex + ez * ez, 1e-9f), 0f, 1f);
                    float dx = p.X - a.X - t * ex, dz = p.Z - a.Z - t * ez;
                    best = MathF.Min(best, MathF.Sqrt(dx * dx + dz * dz));
                }
            }
        }
        return best;
    }

    // Fills obs[Slot .. Slot + Count - 1] for a car at x, z, heading h, arc along the line, cte off it,
    // forward speed ownSpeed (last tick's, as the rest of the observation), behind lead (at least one
    // sample). lap: the line's length; dt: this tick's length; delayS: how old the lead's latest
    // sample is; ask: chase_gap (AskInput). lastGap: the gap this returned last tick (NaN = none); set
    // to this tick's. Returns the gap.
    public static float Fill(Span<float> obs, LeadTrack lead, float x, float z, float h, float arc, float cte,
        float ownSpeed, float lap, float dt, float delayS, float ask, ref float lastGap)
    {
        int k = lead.Count - 1;
        ref readonly LeadSample l = ref lead[k];
        float lvLine = LineSpeed(lead, lap);
        float gap = OutlineGap(x, z, h, arc, l.X, l.Z, l.Heading, l.Arc, lap);
        float closing = float.IsNaN(lastGap) ? 0f : (lastGap - gap) / MathF.Max(dt, 1e-3f);
        lastGap = gap;
        float dx = l.X - x, dz = l.Z - z;
        float ch = MathF.Cos(h), sh = MathF.Sin(h);
        int k0 = Slot;
        obs[k0] = 1f;
        obs[k0 + 1] = Math.Clamp(gap / 20f, -1f, 3f);
        obs[k0 + 2] = Math.Clamp(gap / MathF.Max(ownSpeed, 3f), -1f, 3f);
        obs[k0 + 3] = Math.Clamp((dx * ch + dz * sh) / 20f, -3f, 3f);
        obs[k0 + 4] = Math.Clamp((dz * ch - dx * sh) / 20f, -3f, 3f);
        obs[k0 + 5] = Math.Clamp(closing / 10f, -3f, 3f);
        obs[k0 + 6] = WrapPi(l.Heading - h) / MathF.PI;
        obs[k0 + 7] = l.Slip / (MathF.PI / 2f);
        obs[k0 + 8] = l.Yaw / 2f;
        obs[k0 + 9] = Math.Clamp((l.Cte - cte) / 5f, -3f, 3f);
        // where the lead last passed this car's place along the line (drift_env._ghost_here)
        obs[k0 + 10] = 0f;
        obs[k0 + 11] = 0f;
        for (int j = k; j > 0 && k - j < HistorySamples; j--)
        {
            if (WrapArc(lead[j - 1].Arc - arc, lap) <= 0f && 0f < WrapArc(lead[j].Arc - arc, lap))
            {
                obs[k0 + 10] = Math.Clamp((lead[j - 1].Cte - cte) / 5f, -3f, 3f);
                obs[k0 + 11] = lead[j - 1].Slip / (MathF.PI / 2f);
                break;
            }
        }
        obs[k0 + 12] = ask;

        obs[ControlSlot] = l.Steer / (MathF.PI / 4f);
        obs[ControlSlot + 1] = l.Pedal;
        obs[ControlSlot + 2] = l.Brake;
        obs[ControlSlot + 3] = Math.Clamp(Accel(lead) / 5f, -3f, 3f);

        // no plan from a player: it keeps its speed along the line, its offset and its slip
        obs[PlanSlot] = 0f;
        for (int m = 0; m < PathAheadS.Length; m++)
        {
            float farc = l.Arc + lvLine * (PathAheadS[m] + delayS);
            float fgap = WrapArc(farc - arc, lap) - CarFrontM - CarRearM;
            obs[PlanSlot + 1 + 3 * m] = Math.Clamp(fgap / 20f, -1f, 3f);
            obs[PlanSlot + 2 + 3 * m] = Math.Clamp((l.Cte - cte) / 5f, -3f, 3f);
            obs[PlanSlot + 3 + 3 * m] = l.Slip / (MathF.PI / 2f);
        }
        return gap;
    }
}
