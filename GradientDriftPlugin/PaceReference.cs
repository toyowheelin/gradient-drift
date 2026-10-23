using System.Globalization;
using Serilog;

namespace GradientDriftPlugin;

// The course's reference speed per AI-line point, km/h, which a pace of 1.0 asks for: a quick
// human's speed there, a little faster (drift_rl/pace_model.py; drift_env.py course_pace_reference). A
// policy that reads its pace error (DriftPolicy.ReadsPaceOver) needs <track>_pace_reference.csv in
// the server's folder; the trainer writes it beside the track's other data.
public static class PaceReference
{
    public const string FileSuffix = "_pace_reference.csv";

    // Null, with the reason logged, if the file is missing or unreadable.
    public static float[]? Load(string path)
    {
        if (!File.Exists(path))
        {
            VerboseLog.Information("GradientDrift: no pace reference file at {Path}", Path.GetFullPath(path));
            return null;
        }
        try
        {
            var kmh = new List<float>();
            using var reader = new StreamReader(path);
            string[] header = (reader.ReadLine() ?? "").Split(',');
            int idColumn = Array.IndexOf(header, "PointId");
            int kmhColumn = Array.IndexOf(header, "Kmh");
            if (idColumn < 0 || kmhColumn < 0)
            {
                Log.Warning("GradientDrift: {Path} lacks PointId / Kmh", Path.GetFullPath(path));
                return null;
            }
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split(',');
                if (parts.Length <= Math.Max(idColumn, kmhColumn))
                {
                    continue;
                }
                int id = int.Parse(parts[idColumn], CultureInfo.InvariantCulture);
                float value = float.Parse(parts[kmhColumn], CultureInfo.InvariantCulture);
                if (id != kmh.Count || !(value > 0f))
                {
                    Log.Warning("GradientDrift: {Path} is not in point order or has a speed that is not above 0 at point {Id}",
                        Path.GetFullPath(path), id);
                    return null;
                }
                kmh.Add(value);
            }
            VerboseLog.Information("GradientDrift: loaded the pace reference ({Count} points) from {Path}", kmh.Count, Path.GetFullPath(path));
            return kmh.ToArray();
        }
        catch (Exception e) when (e is IOException or FormatException or OverflowException or UnauthorizedAccessException)
        {
            Log.Warning(e, "GradientDrift: could not read {Path}", Path.GetFullPath(path));
            return null;
        }
    }
}
