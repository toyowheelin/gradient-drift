using System.Diagnostics;
using Serilog;

namespace GradientDriftPlugin;

// How much of the server's time the plugin takes (LogTimingIntervalSeconds): wall-clock time on
// the server's update thread, per tick, split into the parts that cost the most, logged once an
// interval. Stopwatch timestamps only - no allocation, a few tens of nanoseconds per reading.
internal sealed class PluginTiming
{
    public enum Part { Network, Sensors, Telemetry, Followers, Count }

    private static readonly string[] PartNames = { "network", "sensors", "telemetry", "followers" };

    private readonly long _intervalMs;
    private readonly long[] _partTicks = new long[(int)Part.Count];
    private long _totalTicks;
    private long _worstTicks;
    private long _tickCount;
    private long _policyCars;
    private long _followerCars;
    private long _intervalStartMs = -1;

    public PluginTiming(float intervalSeconds)
    {
        _intervalMs = (long)(intervalSeconds * 1000f);
    }

    public bool Enabled => _intervalMs > 0;

    public static long Now() => Stopwatch.GetTimestamp();

    // Adds the time since `start` (a Now() reading) to a part.
    public void Add(Part part, long start)
    {
        if (_intervalMs > 0)
        {
            _partTicks[(int)part] += Stopwatch.GetTimestamp() - start;
        }
    }

    // Once per tick, at its end: the whole tick's time since `start`, the cars it moved, and the
    // log line when the interval is up.
    public void EndTick(long start, int policyCars, int followerCars, long serverTimeMs)
    {
        if (_intervalMs <= 0)
        {
            return;
        }
        long elapsed = Stopwatch.GetTimestamp() - start;
        _totalTicks += elapsed;
        _worstTicks = Math.Max(_worstTicks, elapsed);
        _tickCount++;
        _policyCars += policyCars;
        _followerCars += followerCars;
        if (_intervalStartMs < 0)
        {
            _intervalStartMs = serverTimeMs;
            return;
        }
        long spanMs = serverTimeMs - _intervalStartMs;
        if (spanMs < _intervalMs)
        {
            return;
        }

        double toMs = 1000.0 / Stopwatch.Frequency;
        double ticks = Math.Max(1, _tickCount);
        double perTickMs = _totalTicks * toMs / ticks;
        var parts = new System.Text.StringBuilder();
        long accounted = 0;
        for (int i = 0; i < (int)Part.Count; i++)
        {
            accounted += _partTicks[i];
            parts.Append(PartNames[i]).Append(' ')
                .Append((_partTicks[i] * toMs / ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture))
                .Append(", ");
        }
        parts.Append("the rest ")
            .Append((Math.Max(0, _totalTicks - accounted) * toMs / ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
        Log.Information("GradientDrift timing, last {Seconds:F0} s: {Ticks} ticks with {PolicyCars:F1} policy-driven cars and {Followers:F1} followers on average; " +
                        "{PerTick:F3} ms a tick ({Parts} ms), worst tick {Worst:F2} ms; {Core:F2}% of one CPU core",
            spanMs / 1000.0, _tickCount, _policyCars / ticks, _followerCars / ticks, perTickMs, parts.ToString(),
            _worstTicks * toMs, 100.0 * _totalTicks * toMs / spanMs);

        Array.Clear(_partTicks);
        _totalTicks = 0;
        _worstTicks = 0;
        _tickCount = 0;
        _policyCars = 0;
        _followerCars = 0;
        _intervalStartMs = serverTimeMs;
    }
}
