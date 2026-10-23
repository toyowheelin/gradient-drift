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

// One classified triangle of the track model.
public readonly struct TrackTriangle
{
    public readonly Vector3 V1, V2, V3;
    public readonly SurfaceType Surface;

    public TrackTriangle(Vector3 v1, Vector3 v2, Vector3 v3, SurfaceType surface)
    {
        V1 = v1; V2 = v2; V3 = v3; Surface = surface;
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

    // Gzip-compressed CSV rows: nine vertex coordinates and a SurfaceType name, after one header
    // line. A missing file gives an empty array (and a warning): the stock AI then drives.
    public static TrackTriangle[] Load(string path)
    {
        if (!File.Exists(path))
        {
            Log.Warning("GradientDrift: track triangle file not found at {Path} - geometry unavailable (tools/track_triangles.py writes it)",
                Path.GetFullPath(path));
            return Array.Empty<TrackTriangle>();
        }

        var triangles = new List<TrackTriangle>();
        int skipped = 0;
        float[] f = new float[9];

        try
        {
            using var file = File.OpenRead(path);
            using var gzip = new GZipStream(file, CompressionMode.Decompress);
            using var reader = new StreamReader(gzip);
            reader.ReadLine();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
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
                    surface));
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException)
        {
            Log.Warning(ex, "GradientDrift: could not read {Path} - geometry unavailable", Path.GetFullPath(path));
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
                triangles.Count, Path.GetFullPath(path),
                bySurface.GetValueOrDefault(SurfaceType.Road), bySurface.GetValueOrDefault(SurfaceType.Wall),
                bySurface.GetValueOrDefault(SurfaceType.Grass), bySurface.GetValueOrDefault(SurfaceType.Unknown));
        }
        return triangles.ToArray();
    }
}
