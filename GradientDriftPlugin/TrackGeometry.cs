using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using Serilog;

namespace GradientDriftPlugin;

// What a ray hit. Unknown = geometry whose mesh name matched no known surface;
// None = nothing within range (never appears in the triangle file).
public enum SurfaceType { Road, Wall, Grass, Unknown, None }

// One classified triangle of the track model. Grip: its surface's FRICTION over the main road's
// (<track>_collision_grip.csv.gz; 1 without it), what SurfaceGrip scales the tyres by.
public readonly struct TrackTriangle
{
    public readonly Vector3 V1, V2, V3;
    public readonly SurfaceType Surface;
    public readonly float Grip;

    public TrackTriangle(Vector3 v1, Vector3 v2, Vector3 v3, SurfaceType surface, float grip = 1f)
    {
        V1 = v1; V2 = v2; V3 = v3; Surface = surface; Grip = grip;
    }
}

// Track geometry for the ray casts, from the two files tools/track_triangles.py writes from the
// track's own model files (the trainer loads the same two): <track>_triangles.csv.gz, what the
// sensors see, and <track>_collision.csv.gz, what the game itself collides with - the ground the
// car stands on and the walls it is pushed out of. Both sit in the server's working directory.
public static class TrackGeometry
{
    public const string SensorFileSuffix = "_triangles.csv.gz";
    public const string CollisionFileSuffix = "_collision.csv.gz";
    // One row per row of the collision file, in its order: the surface's KEY and its grip
    // (tools/track_triangles.py). Read only with SurfaceGrip.
    public const string CollisionGripFileSuffix = "_collision_grip.csv.gz";

    // Gzip-compressed CSV rows: nine vertex coordinates and a SurfaceType name, after one header
    // line. A missing file gives an empty array (and a warning): the stock AI then drives.
    // gripPath: the collision file's grip rows, read line for line beside it; without the file
    // every triangle's grip is 1, and with one that does not match row for row too (and a warning).
    public static TrackTriangle[] Load(string path, string? gripPath = null)
    {
        float[]? grips = gripPath != null ? LoadGrips(gripPath) : null;
        if (!PluginData.Exists(path))
        {
            Log.Warning("GradientDrift: track triangle file not found at {Path} - geometry unavailable (tools/track_triangles.py writes it)",
                PluginData.Describe(path));
            return Array.Empty<TrackTriangle>();
        }

        var triangles = new List<TrackTriangle>();
        int skipped = 0;
        float[] f = new float[9];

        try
        {
            using var file = PluginData.Open(path);
            using var gzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(gzip);
            reader.ReadLine();
            string? line;
            int row = 0;
            while ((line = reader.ReadLine()) != null)
            {
                float grip = grips != null && row < grips.Length ? grips[row] : 1f;
                row++;
                var parts = line.Split(',');
                if (parts.Length != 10)
                {
                    skipped++;
                    continue;
                }

                bool ok = true;
                for (int i = 0; i < 9; i++)
                {
                    ok &= float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out f[i]);
                }
                if (!ok || !Enum.TryParse<SurfaceType>(parts[9], out var surface))
                {
                    skipped++;
                    continue;
                }

                triangles.Add(new TrackTriangle(
                    new Vector3(f[0], f[1], f[2]),
                    new Vector3(f[3], f[4], f[5]),
                    new Vector3(f[6], f[7], f[8]),
                    surface, grip));
            }
            if (grips != null && row != grips.Length)
            {
                Log.Warning("GradientDrift: {GripPath} has {GripRows} rows and {Path} {Rows}: made from different track files, so every surface grips 1 (run setup again)",
                    gripPath, grips.Length, path, row);
                for (int i = 0; i < triangles.Count; i++)
                {
                    var t = triangles[i];
                    triangles[i] = new TrackTriangle(t.V1, t.V2, t.V3, t.Surface);
                }
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException)
        {
            Log.Warning(ex, "GradientDrift: could not read {Path} - geometry unavailable", PluginData.Describe(path));
            return Array.Empty<TrackTriangle>();
        }

        if (skipped > 0)
        {
            Log.Warning("GradientDrift: {Skipped} track triangle rows in {Path} failed to parse and were dropped", skipped, path);
        }

        if (VerboseLog.On)
        {
            var bySurface = triangles.GroupBy(t => t.Surface).ToDictionary(g => g.Key, g => g.Count());
            VerboseLog.Information("GradientDrift: loaded {Count} track triangles from {Path} - Road={Road}, Wall={Wall}, Grass={Grass}, Unknown={Unknown}",
                triangles.Count, PluginData.Describe(path),
                bySurface.GetValueOrDefault(SurfaceType.Road), bySurface.GetValueOrDefault(SurfaceType.Wall),
                bySurface.GetValueOrDefault(SurfaceType.Grass), bySurface.GetValueOrDefault(SurfaceType.Unknown));
        }
        return triangles.ToArray();
    }

    // The Grip column of a <track>_collision_grip.csv.gz, null if the file is missing or unreadable.
    private static float[]? LoadGrips(string path)
    {
        if (!PluginData.Exists(path))
        {
            return null;
        }
        var grips = new List<float>();
        try
        {
            using var file = PluginData.Open(path);
            using var gzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(gzip);
            reader.ReadLine();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                int comma = line.LastIndexOf(',');
                grips.Add(comma >= 0 && float.TryParse(line.AsSpan(comma + 1), NumberStyles.Float, CultureInfo.InvariantCulture, out float g)
                    && g > 0f ? g : 1f);
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException)
        {
            Log.Warning(ex, "GradientDrift: could not read {Path} - every surface grips 1", PluginData.Describe(path));
            return null;
        }
        if (VerboseLog.On)
        {
            int low = grips.Count(g => g < 0.98f);
            VerboseLog.Information("GradientDrift: surface grip from {Path}: {Low} of {Count} collision triangles grip under 0.98, lowest {Min}",
                PluginData.Describe(path), low, grips.Count, grips.Count > 0 ? grips.Min() : 1f);
        }
        return grips.ToArray();
    }
}
