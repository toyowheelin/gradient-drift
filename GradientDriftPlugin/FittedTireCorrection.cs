namespace GradientDriftPlugin;

// FittedTireCorrection (a policy-file setting): what Assetto Corsa's E36 does that this vehicle
// model does not, per axle, as a share of the axle's static load, added to each axle's sideways
// force in every sub-step. Fitted (ridge regression) to the axle forces worked out from the car's
// recorded motion on two tracks, minus this model's forces in the same state; checked on six other
// recorded sessions. Made for the physics the shipped policy trains with (RoundTireCurve,
// RearSpinGripShare 0.95, tyre load sensitivity, rear friction circle). The same numbers and terms
// as drift_rl/fast_sim.py _fitted_axle_correction; tests/test_plugin_settings.py checks they match.
// Every term is odd: mirrored slips, yaw and body slip give the mirrored force.
internal static class FittedTireCorrection
{
    // per input x in (front slip, rear slip, yaw rate, body slip): x, tanh(x / 0.15), x speed / 20,
    // x throttle, x |front slip|, x |rear slip|, x |yaw rate|, x |body slip|
    private static readonly float[] Front =
    {
        -0.012135f, -0.073053f, 0.006406f, -0.03875f, 0.003466f, -0.137821f, 0.0026f, -0.161797f,
        0.007838f, -0.158396f, -0.035409f, 0.009226f, -0.128156f, 0.270833f, 0.056691f, 0.231038f,
        0.014653f, 0.048599f, 0.076753f, 0.028526f, -0.011456f, 0.026863f, -0.022523f, 0.18313f,
        -0.029164f, -0.154194f, -0.049205f, -0.009549f, -0.141798f, 0.231127f, 0.031054f, 0.200912f,
    };

    private static readonly float[] Rear =
    {
        -0.025664f, 0.040017f, -0.028092f, -0.01111f, -0.026923f, 0.037245f, -0.05085f, 0.036328f,
        0.034147f, 0.381336f, -0.194426f, -0.193532f, 0.044537f, 0.041424f, -0.005817f, 0.050465f,
        0.083099f, 0.02241f, 0.013467f, -0.017624f, 0.057105f, 0.062262f, -0.009957f, 0.099836f,
        0.050813f, 0.078396f, -0.187616f, -0.195762f, 0.051196f, 0.045883f, -0.003435f, 0.060221f,
    };

    private const float Limit = 0.6f;

    // Angles in radians, yaw rate in rad/s, speed in m/s, throttle 0 - 1.
    public static (float Front, float Rear) Correction(float frontSlip, float rearSlip, float yawRate,
        float bodySlip, float speedMs, float throttle)
    {
        System.Span<float> xs = stackalloc float[] { frontSlip, rearSlip, yawRate, bodySlip };
        System.Span<float> mags = stackalloc float[] { System.MathF.Abs(frontSlip), System.MathF.Abs(rearSlip),
            System.MathF.Abs(yawRate), System.MathF.Abs(bodySlip) };
        float vs = speedMs / 20f;
        float cf = 0f, cr = 0f;
        for (int i = 0; i < 4; i++)
        {
            float x = xs[i];
            int k = 8 * i;
            float t = System.MathF.Tanh(x / 0.15f);
            cf += Front[k] * x + Front[k + 1] * t + Front[k + 2] * x * vs + Front[k + 3] * x * throttle;
            cr += Rear[k] * x + Rear[k + 1] * t + Rear[k + 2] * x * vs + Rear[k + 3] * x * throttle;
            for (int j = 0; j < 4; j++)
            {
                cf += Front[k + 4 + j] * x * mags[j];
                cr += Rear[k + 4 + j] * x * mags[j];
            }
        }
        return (System.Math.Clamp(cf, -Limit, Limit), System.Math.Clamp(cr, -Limit, Limit));
    }
}
