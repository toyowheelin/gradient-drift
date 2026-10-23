using System;

namespace GradientDriftPlugin;

// What GlitchCorrector.Correct returns: the angle to use and whether it had to be limited.
public readonly struct GlitchCorrectionResult
{
    public readonly float CorrectedAngle;

    public readonly bool GlitchCorrected;

    public GlitchCorrectionResult(float correctedAngle, bool glitchCorrected)
    {
        CorrectedAngle = correctedAngle;
        GlitchCorrected = glitchCorrected;
    }
}

// Rate limiter on the racing line's direction at the car. The interpolated spline direction
// occasionally jumps by 0.4 rad or more in a single tick; a real road cannot turn faster than
// the car could follow, so a step implying more than twice the lateral acceleration budget at
// the car's current speed is treated as a bad reading and limited to that step.
public static class GlitchCorrector
{
    // previousCorrectedAngle: this car's own output from the previous tick; null on its first
    // tick, when the raw angle is returned as is.
    // ownSpeedForGlitchCheck: the car's speed from the PREVIOUS tick - call this before the
    // speed ramp updates it.
    public static GlitchCorrectionResult Correct(float rawSplineAngle, float? previousCorrectedAngle,
        float ownSpeedForGlitchCheck, float dt, float maxLateralAccelMs2)
    {
        if (!previousCorrectedAngle.HasValue)
        {
            return new GlitchCorrectionResult(rawSplineAngle, false);
        }

        float prevAngle = previousCorrectedAngle.Value;
        float rawStep = rawSplineAngle - prevAngle;
        while (rawStep > MathF.PI) rawStep -= 2 * MathF.PI;
        while (rawStep < -MathF.PI) rawStep += 2 * MathF.PI;

        float maxPlausibleRate = (2f * maxLateralAccelMs2) / MathF.Max(ownSpeedForGlitchCheck, 0.5f);
        float maxPlausibleStep = maxPlausibleRate * dt;

        if (MathF.Abs(rawStep) > maxPlausibleStep)
        {
            float correctedAngle = prevAngle + MathF.Sign(rawStep) * maxPlausibleStep;
            return new GlitchCorrectionResult(correctedAngle, true);
        }

        return new GlitchCorrectionResult(rawSplineAngle, false);
    }
}
