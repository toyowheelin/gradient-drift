using System;

namespace GradientDriftPlugin;

// Car-on-car contact (CarContactPush): two overlapping car bodies meet as a collision at the point
// of the first body nearest the second's centre, along the line between their centres, with
// restitution and no friction - the same as the tandem trainer's push (drift_env._touch_push).
// Each car takes its share of the impulse by its own mass and yaw inertia: its forward and
// sideways speed change, and an off-centre hit turns it. Nothing happens while the two are
// already moving apart.
//
// A bot hit by a player takes its share; the player's car takes the other share in the player's
// own game, which already collides with the bots. Two bots both take theirs here.
public static class CarContact
{
    public const float Restitution = 0.2f;          // drift_env.TOUCH_RESTITUTION

    // One car: centre, heading (rad, as DriftCarState.Heading), forward and sideways speed (m/s,
    // the car's own axes) and yaw rate (rad/s); its mass and yaw inertia.
    public struct Body
    {
        public float X, Z, Heading, Forward, Lateral, YawRate, MassKg, InertiaKgM2;
    }

    // Applies the collision to a and b (when they close on each other) and returns the yaw-rate
    // change it gave b, rad/s; 0 when nothing happened.
    public static float Collide(ref Body a, ref Body b)
    {
        float nx = a.X - b.X, nz = a.Z - b.Z;               // from b to a
        float dist = MathF.Sqrt(nx * nx + nz * nz);
        if (dist < 1e-6f)
            return 0f;
        nx /= dist;
        nz /= dist;
        float cs = MathF.Cos(a.Heading), sn = MathF.Sin(a.Heading);
        // the contact point: a's body point nearest b's centre, relative to a's centre
        float f = (b.X - a.X) * cs + (b.Z - a.Z) * sn;
        float l = -(b.X - a.X) * sn + (b.Z - a.Z) * cs;
        f = Math.Clamp(f, -ChaseInputs.CarRearM, ChaseInputs.CarFrontM);
        l = Math.Clamp(l, -ChaseInputs.CarHalfWidthM, ChaseInputs.CarHalfWidthM);
        float rx = f * cs - l * sn, rz = f * sn + l * cs;
        // the same point from b's centre
        float qx = a.X + rx - b.X, qz = a.Z + rz - b.Z;
        var (avx, avz) = PointVelocity(a, rx, rz);
        var (bvx, bvz) = PointVelocity(b, qx, qz);
        float vn = (avx - bvx) * nx + (avz - bvz) * nz;
        if (vn >= 0f)
            return 0f;                                       // already moving apart
        float rn = rx * nz - rz * nx;
        float qn = qx * nz - qz * nx;
        float j = -(1f + Restitution) * vn
            / (1f / a.MassKg + 1f / b.MassKg + rn * rn / a.InertiaKgM2 + qn * qn / b.InertiaKgM2);
        Apply(ref a, j * nx, j * nz, rn * j);
        float bw = -qn * j / b.InertiaKgM2;
        Apply(ref b, -j * nx, -j * nz, -qn * j);
        return bw;
    }

    // A player pressing on a bot: the player's own game treats the bot as a fixed object, so after
    // the first hit the player's car sits against it with little closing speed left, and Collide
    // (closing speed only) gives nothing. While the two still overlap, Press keeps the bot moving
    // away: its speed along the contact normal is raised to at least the pusher's, plus the overlap
    // depth over PressSeconds (at most PressMaxMs), at its centre. A push held on keeps moving it.
    public const float PressSeconds = 0.15f;
    public const float PressMaxMs = 3f;
    // Real cars are longer than the contact box (3.9 m): a player's car touching a bot's nose or
    // tail in its own game can sit just outside it. A player contact counts this much longer at
    // each end.
    public const float PlayerExtraLengthM = 0.3f;

    // How deep two car boxes overlap (separating axes over both cars' sides): the depth in metres
    // and the unit normal (nx, nz) from b towards a along the shallowest axis; 0 when apart. Each
    // box is longer by extraLengthM at both ends.
    public static float Penetration(in Body a, in Body b, float extraLengthM, out float nx, out float nz)
    {
        nx = nz = 0f;
        float halfLen = 0.5f * (ChaseInputs.CarFrontM + ChaseInputs.CarRearM) + extraLengthM;
        float shift = 0.5f * (ChaseInputs.CarFrontM - ChaseInputs.CarRearM);
        float ca = MathF.Cos(a.Heading), sa = MathF.Sin(a.Heading);
        float cb = MathF.Cos(b.Heading), sb = MathF.Sin(b.Heading);
        float ax = a.X + shift * ca, az = a.Z + shift * sa;
        float bx = b.X + shift * cb, bz = b.Z + shift * sb;
        float best = float.MaxValue;
        for (int k = 0; k < 4; k++)
        {
            float h = k < 2 ? a.Heading : b.Heading;
            float ux = k % 2 == 0 ? MathF.Cos(h) : -MathF.Sin(h);
            float uz = k % 2 == 0 ? MathF.Sin(h) : MathF.Cos(h);
            float ra = halfLen * MathF.Abs(ca * ux + sa * uz) + ChaseInputs.CarHalfWidthM * MathF.Abs(ca * uz - sa * ux);
            float rb = halfLen * MathF.Abs(cb * ux + sb * uz) + ChaseInputs.CarHalfWidthM * MathF.Abs(cb * uz - sb * ux);
            float d = (ax - bx) * ux + (az - bz) * uz;
            float overlap = ra + rb - MathF.Abs(d);
            if (overlap <= 0f)
                return 0f;
            if (overlap < best)
            {
                best = overlap;
                float sign = d >= 0f ? 1f : -1f;
                nx = sign * ux;
                nz = sign * uz;
            }
        }
        return best;
    }

    // Raises car's speed along (nx, nz) (from the pusher towards car) to the pusher's speed along it
    // plus depth / PressSeconds, capped at PressMaxMs over the pusher's; true when it changed.
    public static bool Press(ref Body car, in Body pusher, float depth, float nx, float nz)
    {
        float cs = MathF.Cos(car.Heading), sn = MathF.Sin(car.Heading);
        float pc = MathF.Cos(pusher.Heading), ps = MathF.Sin(pusher.Heading);
        float have = (car.Forward * cs - car.Lateral * sn) * nx + (car.Forward * sn + car.Lateral * cs) * nz;
        float pusherAlong = (pusher.Forward * pc - pusher.Lateral * ps) * nx + (pusher.Forward * ps + pusher.Lateral * pc) * nz;
        float want = pusherAlong + MathF.Min(depth / PressSeconds, PressMaxMs);
        if (have >= want)
            return false;
        float dv = want - have;
        car.Forward = MathF.Max(0f, car.Forward + dv * (nx * cs + nz * sn));
        car.Lateral += dv * (-nx * sn + nz * cs);
        return true;
    }

    // a body point's velocity (world axes) at r from its centre
    private static (float X, float Z) PointVelocity(in Body c, float rx, float rz)
    {
        float cs = MathF.Cos(c.Heading), sn = MathF.Sin(c.Heading);
        return (c.Forward * cs - c.Lateral * sn - c.YawRate * rz,
                c.Forward * sn + c.Lateral * cs + c.YawRate * rx);
    }

    // impulse (jx, jz) at a point whose moment arm gives the yaw impulse jw (N m s)
    private static void Apply(ref Body c, float jx, float jz, float jw)
    {
        float cs = MathF.Cos(c.Heading), sn = MathF.Sin(c.Heading);
        float dvx = jx / c.MassKg, dvz = jz / c.MassKg;
        c.Forward = MathF.Max(0f, c.Forward + dvx * cs + dvz * sn);
        c.Lateral += -dvx * sn + dvz * cs;
        c.YawRate += jw / c.InertiaKgM2;
    }
}
