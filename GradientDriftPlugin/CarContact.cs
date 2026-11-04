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
