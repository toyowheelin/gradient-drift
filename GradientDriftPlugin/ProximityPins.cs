using System;
using System.Numerics;

namespace GradientDriftPlugin;

// Six short rays fixed to the chassis: straight left and right plus the four diagonals. Unlike
// RaycastFan they follow where the car POINTS, so each keeps watching the same side of the car
// through a slide. A second set on the collision triangles, re-aimed along the direction of
// travel, detects contact with the walls the game collides with.
public class ProximityPins
{
    // Ray directions relative to the base angle, in the order Cast returns them.
    public static readonly float[] OffsetsRadians =
    {
        -135f * MathF.PI / 180f,
        -90f  * MathF.PI / 180f,
        -45f  * MathF.PI / 180f,
         45f  * MathF.PI / 180f,
         90f  * MathF.PI / 180f,
         135f * MathF.PI / 180f,
    };

    private readonly TriangleGrid _grid;
    private readonly float _maxDistance;
    private readonly float _heightOffset;
    private readonly bool _wallsOnly;

    // wallsOnly: only walls stop a ray (the contact pins, on the collision triangles).
    public ProximityPins(TriangleGrid grid, float maxDistance, float heightOffset, bool wallsOnly = false)
    {
        _grid = grid;
        _maxDistance = maxDistance;
        _heightOffset = heightOffset;
        _wallsOnly = wallsOnly;
    }

    // baseAngle: radians, atan2(Z, X). A ray that finds nothing reports (maxDistance, None).
    public FanRay[] Cast(Vector3 carPosition, float baseAngle)
    {
        var origin = new Vector3(carPosition.X, carPosition.Y + _heightOffset, carPosition.Z);
        var results = new FanRay[OffsetsRadians.Length];

        for (int i = 0; i < OffsetsRadians.Length; i++)
        {
            float angle = baseAngle + OffsetsRadians[i];
            var direction = new Vector3(MathF.Cos(angle), 0f, MathF.Sin(angle));
            results[i] = _grid.Raycast(origin, direction, _maxDistance, out var hit, _wallsOnly)
                ? new FanRay(hit.Distance, hit.Surface)
                : new FanRay(_maxDistance, SurfaceType.None);
        }

        return results;
    }
}
