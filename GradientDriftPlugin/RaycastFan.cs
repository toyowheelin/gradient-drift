using System;
using System.Numerics;

namespace GradientDriftPlugin;

// One sensor ray: how far it got and what it hit (SurfaceType.None at full range = nothing).
public readonly struct FanRay
{
    public readonly float Distance;
    public readonly SurfaceType Surface;

    public FanRay(float distance, SurfaceType surface)
    {
        Distance = distance;
        Surface = surface;
    }
}

// A fan of horizontal rays against the track geometry, spread evenly either side of a centre
// angle. The caller centres it on the car's direction of TRAVEL, not where it points: in a
// slide those differ by the slip angle, and what matters is what the car is sliding toward.
public class RaycastFan
{
    private readonly TriangleGrid _grid;
    private readonly int _rayCount;
    private readonly float _fanAngleRadians;
    private readonly float _maxDistance;
    private readonly float _heightOffset;

    public RaycastFan(TriangleGrid grid, int rayCount, float fanAngleDegrees, float maxDistance, float heightOffset)
    {
        _grid = grid;
        _rayCount = Math.Max(rayCount, 2);
        _fanAngleRadians = fanAngleDegrees * (MathF.PI / 180f);
        _maxDistance = maxDistance;
        _heightOffset = heightOffset;
    }

    // carPosition is at the car's ride height; rays start heightOffset above it so they do not
    // graze the road. centerAngle: radians, atan2(Z, X). Rays are ordered from -fan/2 to +fan/2.
    public FanRay[] Cast(Vector3 carPosition, float centerAngle)
    {
        var origin = new Vector3(carPosition.X, carPosition.Y + _heightOffset, carPosition.Z);
        var results = new FanRay[_rayCount];
        float startAngle = centerAngle - _fanAngleRadians / 2f;
        float step = _rayCount > 1 ? _fanAngleRadians / (_rayCount - 1) : 0f;

        for (int i = 0; i < _rayCount; i++)
        {
            float angle = startAngle + step * i;
            var direction = new Vector3(MathF.Cos(angle), 0f, MathF.Sin(angle));
            results[i] = _grid.Raycast(origin, direction, _maxDistance, out var hit)
                ? new FanRay(hit.Distance, hit.Surface)
                : new FanRay(_maxDistance, SurfaceType.None);
        }

        return results;
    }
}
