using System.Globalization;
using Serilog;

namespace GradientDriftPlugin;

// The course's edges per AI-line point: how far the course reaches either side of the line
// (drift_rl/build_course.py: tire stacks, walls or the paving's edge). <track>_course_edges.csv in
// the server's folder (drift_rl/setup_track.py writes it). A policy that reads the room to the
// edges (DriftPolicy.ReadsCourseRoom) needs it; others do not. Beside it, <track>_course_zones.csv
// (drift_rl/build_zones.py, or a server owner's own): which edge has a drift zone at each point, read
// by a policy with the zone inputs (DriftPolicy.ReadsZones).
public sealed class CourseEdges
{
    public const string FileSuffix = "_course_edges.csv";
    public const string ZonesFileSuffix = "_course_zones.csv";

    public float[] PositiveSideM { get; }
    public float[] NegativeSideM { get; }
    public int PointCount => PositiveSideM.Length;
    // +1 a zone on the positive edge at that point, -1 on the negative, 0 none; null = no zones file
    public sbyte[]? ZoneSide { get; private set; }

    private CourseEdges(float[] positive, float[] negative)
    {
        PositiveSideM = positive;
        NegativeSideM = negative;
    }

    // The zones file beside the edges file: false, with the reason logged, if it is missing or unreadable
    // (then no point has a zone).
    public bool LoadZones(string path)
    {
        if (!PluginData.Exists(path))
        {
            return false;
        }
        try
        {
            var side = new sbyte[PointCount];
            using var reader = new StreamReader(PluginData.Open(path));
            string[] header = (reader.ReadLine() ?? "").Split(',');
            int sideColumn = Array.IndexOf(header, "Side");
            int firstColumn = Array.IndexOf(header, "FirstPointId");
            int lastColumn = Array.IndexOf(header, "LastPointId");
            if (sideColumn < 0 || firstColumn < 0 || lastColumn < 0)
            {
                Log.Warning("GradientDrift: {Path} lacks Side / FirstPointId / LastPointId", PluginData.Describe(path));
                return false;
            }
            int n = PointCount;
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split(',');
                if (parts.Length <= Math.Max(sideColumn, Math.Max(firstColumn, lastColumn)))
                {
                    continue;
                }
                int s = Math.Sign(int.Parse(parts[sideColumn], CultureInfo.InvariantCulture));
                int first = int.Parse(parts[firstColumn], CultureInfo.InvariantCulture);
                int last = int.Parse(parts[lastColumn], CultureInfo.InvariantCulture);
                if (first < 0 || first >= n || last < 0 || last >= n)
                {
                    Log.Warning("GradientDrift: {Path} has a zone outside the AI line's {Points} points", PluginData.Describe(path), n);
                    return false;
                }
                int length = ((last - first) % n + n) % n;
                for (int k = 0; k <= length; k++)
                {
                    side[(first + k) % n] = (sbyte)s;
                }
            }
            ZoneSide = side;
            return true;
        }
        catch (Exception e) when (e is IOException or FormatException or OverflowException or UnauthorizedAccessException)
        {
            Log.Warning(e, "GradientDrift: could not read {Path}", PluginData.Describe(path));
            return false;
        }
    }

    // Null, with the reason logged, if the file is missing or unreadable.
    public static CourseEdges? Load(string path)
    {
        if (!PluginData.Exists(path))
        {
            VerboseLog.Information("GradientDrift: no course edges file at {Path}", PluginData.Describe(path));
            return null;
        }
        try
        {
            var positive = new List<float>();
            var negative = new List<float>();
            using var reader = new StreamReader(PluginData.Open(path));
            string[] header = (reader.ReadLine() ?? "").Split(',');
            int idColumn = Array.IndexOf(header, "PointId");
            int posColumn = Array.IndexOf(header, "PositiveSideM");
            int negColumn = Array.IndexOf(header, "NegativeSideM");
            if (idColumn < 0 || posColumn < 0 || negColumn < 0)
            {
                Log.Warning("GradientDrift: {Path} lacks PointId / PositiveSideM / NegativeSideM", PluginData.Describe(path));
                return null;
            }
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split(',');
                if (parts.Length <= Math.Max(idColumn, Math.Max(posColumn, negColumn)))
                {
                    continue;
                }
                int id = int.Parse(parts[idColumn], CultureInfo.InvariantCulture);
                if (id != positive.Count)
                {
                    Log.Warning("GradientDrift: {Path} is not in point order at point {Id}", PluginData.Describe(path), id);
                    return null;
                }
                positive.Add(float.Parse(parts[posColumn], CultureInfo.InvariantCulture));
                negative.Add(float.Parse(parts[negColumn], CultureInfo.InvariantCulture));
            }
            VerboseLog.Information("GradientDrift: loaded course edges ({Count} points) from {Path}", positive.Count, PluginData.Describe(path));
            return new CourseEdges(positive.ToArray(), negative.ToArray());
        }
        catch (Exception e) when (e is IOException or FormatException or OverflowException or UnauthorizedAccessException)
        {
            Log.Warning(e, "GradientDrift: could not read {Path}", PluginData.Describe(path));
            return null;
        }
    }
}
