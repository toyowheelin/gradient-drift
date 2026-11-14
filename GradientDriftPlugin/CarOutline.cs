namespace GradientDriftPlugin;

// A car's outline for wall contact (CarOutlineContact): a rectangle around its centre, reaching
// each axle's distance from the centre plus the overhang forward and back, and half the wider
// track plus the side margin to each side. Its points, in metres (forward, left) of the centre,
// are fast_sim._outline_point's list, in the same order.
public readonly struct CarOutline
{
    public const int PointCount = 8;

    private readonly float _front;
    private readonly float _rear;
    private readonly float _half;

    public CarOutline(CarSpec car, float overhangMeters, float sideMarginMeters)
    {
        _front = car.DistanceCgToFrontAxleM + overhangMeters;
        _rear = car.DistanceCgToRearAxleM + overhangMeters;
        _half = 0.5f * MathF.Max(car.FrontTrackMeters, car.RearTrackMeters) + sideMarginMeters;
    }

    // The front corners and middle, the sides' middles, the rear corners and middle.
    public (float Forward, float Left) Point(int i) => i switch
    {
        0 => (_front, _half),
        1 => (_front, 0f),
        2 => (_front, -_half),
        3 => (0f, _half),
        4 => (0f, -_half),
        5 => (-_rear, _half),
        6 => (-_rear, 0f),
        _ => (-_rear, -_half),
    };
}

// The car's body for wall contact (BodyWallContact): its car file's body when it has one, else the
// outline's rectangle. Points round its edge, no more than SpacingMeters apart, going round from the
// front left corner (across the front, down the right side, across the rear, up the left side),
// each corner once: fast_sim._body_point's list, in the same order.
public readonly struct CarBody
{
    public const float SpacingMeters = 0.5f;

    private readonly float _front;
    private readonly float _rear;
    private readonly float _half;
    private readonly int _across;
    private readonly int _along;

    public CarBody(CarSpec car, float overhangMeters, float sideMarginMeters)
    {
        if (car.BodyFrontM > 0f && car.BodyRearM > 0f && car.BodyHalfWidthM > 0f)
        {
            _front = car.BodyFrontM;
            _rear = car.BodyRearM;
            _half = car.BodyHalfWidthM;
        }
        else
        {
            _front = car.DistanceCgToFrontAxleM + overhangMeters;
            _rear = car.DistanceCgToRearAxleM + overhangMeters;
            _half = 0.5f * MathF.Max(car.FrontTrackMeters, car.RearTrackMeters) + sideMarginMeters;
        }
        _across = Math.Max(1, (int)MathF.Ceiling(2f * _half / SpacingMeters - 1e-6f));
        _along = Math.Max(1, (int)MathF.Ceiling((_front + _rear) / SpacingMeters - 1e-6f));
    }

    public int PointCount => 2 * (_across + _along);
    public float Front => _front;
    public float Rear => _rear;
    public float HalfWidth => _half;

    public (float Forward, float Left) Point(int i)
    {
        float width = 2f * _half;
        float length = _front + _rear;
        if (i < _across)
        {
            return (_front, _half - i * width / _across);
        }
        i -= _across;
        if (i < _along)
        {
            return (_front - i * length / _along, -_half);
        }
        i -= _along;
        if (i < _across)
        {
            return (-_rear, -_half + i * width / _across);
        }
        i -= _across;
        return (-_rear + i * length / _along, _half);
    }
}
