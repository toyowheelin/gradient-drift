using System.Globalization;
using Serilog;

namespace GradientDriftPlugin;

// The course's edges per AI-line point: how far the course reaches either side of the line
// (drift_rl/build_course.py: tire stacks, walls or the paving's edge). <track>_course_edges.csv in
// the server's folder (drift_rl/setup_track.py writes it). A policy that reads the room to the
// edges (DriftPolicy.ReadsCourseRoom) needs it; others do not.
public sealed class CourseEdges
{
    public const string FileSuffix = "_course_edges.csv";

    public float[] PositiveSideM { get; }
    public float[] NegativeSideM { get; }
    public int PointCount => PositiveSideM.Length;

    private CourseEdges(float[] positive, float[] negative)
    {
        PositiveSideM = positive;
        NegativeSideM = negative;
    }

    // Null, with the reason logged, if the file is missing or unreadable.
    public static CourseEdges? Load(string path)
    {
        if (!File.Exists(path))
        {
            VerboseLog.Information("GradientDrift: no course edges file at {Path}", Path.GetFullPath(path));
            return null;
        }
        try
        {
            var positive = new List<float>();
            var negative = new List<float>();
            using var reader = new StreamReader(path);
            string[] header = (reader.ReadLine() ?? "").Split(',');
            int idColumn = Array.IndexOf(header, "PointId");
            int posColumn = Array.IndexOf(header, "PositiveSideM");
            int negColumn = Array.IndexOf(header, "NegativeSideM");
            if (idColumn < 0 || posColumn < 0 || negColumn < 0)
            {
                Log.Warning("GradientDrift: {Path} lacks PointId / PositiveSideM / NegativeSideM", Path.GetFullPath(path));
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
                    Log.Warning("GradientDrift: {Path} is not in point order at point {Id}", Path.GetFullPath(path), id);
                    return null;
                }
                positive.Add(float.Parse(parts[posColumn], CultureInfo.InvariantCulture));
                negative.Add(float.Parse(parts[negColumn], CultureInfo.InvariantCulture));
            }
            VerboseLog.Information("GradientDrift: loaded course edges ({Count} points) from {Path}", positive.Count, Path.GetFullPath(path));
            return new CourseEdges(positive.ToArray(), negative.ToArray());
        }
        catch (Exception e) when (e is IOException or FormatException or OverflowException or UnauthorizedAccessException)
        {
            Log.Warning(e, "GradientDrift: could not read {Path}", Path.GetFullPath(path));
            return null;
        }
    }
}
