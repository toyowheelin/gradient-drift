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
