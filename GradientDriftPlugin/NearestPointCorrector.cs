using System;
using System.Numerics;
using AssettoServer.Server.Ai.Splines;

namespace GradientDriftPlugin;

// What NearestPointCorrector.Correct returns: the point id to use, whether it replaced the raw
// one, and for how many ticks in a row it has done so.
public readonly struct NearestPointCorrectionResult
{
    public readonly int CorrectedPointId;

    public readonly bool GlitchCorrected;

    public readonly int ConsecutiveCorrectionStreak;

    public NearestPointCorrectionResult(int correctedPointId, bool glitchCorrected, int consecutiveCorrectionStreak)
    {
        CorrectedPointId = correctedPointId;
        GlitchCorrected = glitchCorrected;
        ConsecutiveCorrectionStreak = consecutiveCorrectionStreak;
    }
}

// Continuity check on the nearest spline point. AiSpline.WorldToSpline is a plain
// closest-point search with no notion of lap order, so where two parts of the track pass close
// to each other it can jump to a point far away along the lap. Everything downstream (line
// direction, cross-track error, radius, the policy's look-ahead) reads from that id, so it is
// corrected once, here: a point that cannot be reached from last tick's point within the
// distance the car could have travelled is replaced by the point nearest the car among those
// that can.
public static class NearestPointCorrector
{
    // Margin on speed x dt: covers chord-vs-arc shortening and last tick's speed being stale.
    // Real jumps are 100+ points, so the exact value does not matter.
    private const float PlausibilityMarginFactor = 2f;

    // Floor under the travel budget. Without it the check fired on 21% of ordinary ticks at low
    // speed, on search noise of a few metres.
    private const float MinimumPlausibilityFloorM = 10f;

    // After this many corrected ticks in a row the raw reading is trusted again. Without it a
    // car wedged against a wall was dead-reckoned along the line for 924 ticks.
    private const int MaxConsecutiveCorrectionTicks = 60;

    // previousCorrectedPointId / previousConsecutiveCorrectionStreak: this car's own outputs from
    // the previous tick (null / 0 on its first tick, when the raw id is returned as is).
    // ownSpeedForGlitchCheck: the car's speed from the PREVIOUS tick.
    // A plausible step is accepted in either direction (a car can slide backwards).
    public static NearestPointCorrectionResult Correct(int rawPointId, int? previousCorrectedPointId,
        int previousConsecutiveCorrectionStreak, float ownSpeedForGlitchCheck, float dt, Vector3 position,
        ReadOnlySpan<SplinePoint> points)
    {
        if (!previousCorrectedPointId.HasValue)
        {
            return new NearestPointCorrectionResult(rawPointId, false, 0);
        }

        int prevId = previousCorrectedPointId.Value;
        if (prevId < 0 || prevId >= points.Length || rawPointId < 0 || rawPointId >= points.Length)
        {
            return new NearestPointCorrectionResult(rawPointId, false, 0);
        }

        if (rawPointId == prevId)
        {
            return new NearestPointCorrectionResult(rawPointId, false, 0);
        }

        if (previousConsecutiveCorrectionStreak >= MaxConsecutiveCorrectionTicks)
        {
            return new NearestPointCorrectionResult(rawPointId, false, 0);
        }

        float maxPlausibleDistanceM = Math.Max(
            ownSpeedForGlitchCheck * dt * PlausibilityMarginFactor,
            MinimumPlausibilityFloorM);

        if (IsReachable(points, prevId, rawPointId, maxPlausibleDistanceM, forward: true) ||
            IsReachable(points, prevId, rawPointId, maxPlausibleDistanceM, forward: false))
        {
            return new NearestPointCorrectionResult(rawPointId, false, 0);
        }

        int correctedId = NearestWithin(points, prevId, maxPlausibleDistanceM, position);
        return new NearestPointCorrectionResult(correctedId, true, previousConsecutiveCorrectionStreak + 1);
    }

    // True if targetId is within maxDistanceM of startId walking the spline's own links.
    private static bool IsReachable(ReadOnlySpan<SplinePoint> points, int startId, int targetId,
        float maxDistanceM, bool forward)
    {
        int currentId = startId;
        float traveled = 0f;
        for (int step = 0; step < points.Length; step++)
        {
            ref readonly var current = ref points[currentId];
            int neighborId = forward ? current.NextId : current.PreviousId;
            if (neighborId < 0 || neighborId >= points.Length)
            {
                return false;
            }

            traveled += forward ? current.Length : points[neighborId].Length;
            if (traveled > maxDistanceM)
            {
                return false;
            }

            if (neighborId == targetId)
            {
                return true;
            }

            currentId = neighborId;
        }

        return false;
    }

    // The point nearest position among those within maxDistanceM of startId along the spline's own
    // links, either way.
    private static int NearestWithin(ReadOnlySpan<SplinePoint> points, int startId, float maxDistanceM,
        Vector3 position)
    {
        int bestId = startId;
        float best = Vector3.DistanceSquared(points[startId].Position, position);
        for (int direction = 0; direction < 2; direction++)
        {
            bool forward = direction == 0;
            int currentId = startId;
            float traveled = 0f;
            for (int step = 0; step < points.Length; step++)
            {
                ref readonly var current = ref points[currentId];
                int neighborId = forward ? current.NextId : current.PreviousId;
                if (neighborId < 0 || neighborId >= points.Length)
                {
                    break;
                }
                traveled += forward ? current.Length : points[neighborId].Length;
                if (traveled > maxDistanceM)
                {
                    break;
                }
                float distance = Vector3.DistanceSquared(points[neighborId].Position, position);
                if (distance < best)
                {
                    best = distance;
                    bestId = neighborId;
                }
                currentId = neighborId;
            }
        }
        return bestId;
    }
}
