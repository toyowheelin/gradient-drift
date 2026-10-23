using System.Numerics;

namespace GradientDriftPlugin;

// Signed sideways distance of the car from the racing line, in metres: positive = left of the
// line. Measured fresh from the car's real position every tick, never integrated.
public static class CrossTrackErrorCalculator
{
    // interpolatedPathPosition: the point on the line abeam the car. splineDirection: unit vector
    // along the line there (the caller passes the one it already built, not a second cos/sin).
    public static float Compute(Vector3 carPosition, Vector3 interpolatedPathPosition, Vector3 splineDirection)
    {
        Vector3 toCarHorizontal = new Vector3(
            carPosition.X - interpolatedPathPosition.X, 0f, carPosition.Z - interpolatedPathPosition.Z);

        Vector3 pathLeftUnit = new Vector3(-splineDirection.Z, 0f, splineDirection.X);
        return Vector3.Dot(toCarHorizontal, pathLeftUnit);
    }
}
