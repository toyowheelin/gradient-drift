using System.Globalization;
using System.Numerics;
using System.Text;
using AssettoServer.Server.Ai.Splines;
using Serilog;

namespace GradientDriftPlugin;

// The track's drift zones painted on the road for players: thick yellow boxes along the outside
// edge of the corners, closed at both ends, with diagonal stripes inside (sparser for a
// touch-and-go area). The zones are the ones the trainer pays the bots to fill: <track>_course_zones.csv
// (drift_rl/build_zones.py, or setup's zone editor) on <track>_course_edges.csv's points, which are
// the AI line's, so each point's height is the AI line's there. Spikes in the edges (open run-off
// where nothing marks the course) are taken out first, as build_zones does.
//
// The plugin sends players a CSP online script (AssettoServer's CSPServerScriptProvider) that
// paints them with CSP's ac.TrackPaint, which lays the lines on the track's surface. Players
// without CSP, or with a CSP too old for ac.TrackPaint, see nothing. setup's zones.paint_shapes is
// the same geometry in Python (tests/plugin checks they agree).
public static class DriftZones
{
    public const string FileSuffix = "_course_zones.csv";
    public const float DepthM = 3f;          // build_zones.ZONE_DEPTH_M: the box, measured in from the edge
    public const float StripeSlantM = 2.1f;  // a stripe ends this far along the inner side from where it starts
    public const float StripeMarginM = 0.6f; // no stripe closer than this to either end bar
    public const string ScriptName = "gradientdrift_zones.lua";
    // build_zones.SPIKE_WINDOW / SPIKE_M / SMOOTH_POINTS / SMOOTH_KEEP_M: where nothing marks the
    // course its edge is the paving up to the first wall, so it jumps out across an open corner's
    // run-off for a few points and zigzags from point to point. Each side's width is held within
    // SpikeM of its running median over SpikeWindow points, then averaged over SmoothPoints, before
    // painting; a width that moves by no more than SmoothKeepM is left as it was.
    public const int SpikeWindow = 15;
    public const float SpikeM = 1f;
    public const int SmoothPoints = 5;
    public const float SmoothKeepM = 0.01f;

    public sealed record Shape(bool OutsideZone, Vector3[] Frame, (Vector3 From, Vector3 To)[] Stripes);

    // A track's course as zones are drawn and judged on it: each AI-line point's two edges (spikes
    // taken out, at the AI line's height there) and the zones on them (side +1 = the positive (left)
    // edge, -1 the negative (right) one; first to last runs round the lap).
    public sealed record Zone(bool OutsideZone, int Side, int First, int Last);
    public sealed record Course(Vector3[] Positive, Vector3[] Negative, List<Zone> Zones);

    // The course and its zones, or null (the reason logged, "what" saying what goes without them)
    // when the files are missing, unreadable or not on this AI line's points.
    public static Course? LoadCourse(string trackName, ReadOnlySpan<SplinePoint> points, string what = "no zones painted")
    {
        string zonesFile = trackName + FileSuffix;
        string edgesFile = trackName + CourseEdges.FileSuffix;
        if (!PluginData.Exists(zonesFile) || !PluginData.Exists(edgesFile))
        {
            VerboseLog.Information("GradientDrift: {What} - {Zones} or {Edges} is not there (setup writes them with the track's other files)",
                what, PluginData.Describe(zonesFile), PluginData.Describe(edgesFile));
            return null;
        }
        try
        {
            List<string[]> edges = ReadTable(edgesFile, "PointId", "PositiveEdgeX", "PositiveEdgeZ", "NegativeEdgeX", "NegativeEdgeZ",
                "PositiveSideM", "NegativeSideM");
            List<string[]> zones = ReadTable(zonesFile, "Kind", "Side", "FirstPointId", "LastPointId");
            if (edges.Count != points.Length)
            {
                Log.Warning("GradientDrift: {What} - {Edges} has {Rows} points, the AI line {Points}",
                    what, PluginData.Describe(edgesFile), edges.Count, points.Length);
                return null;
            }
            int n = edges.Count;
            var positive = new Vector3[n];
            var negative = new Vector3[n];
            var positiveM = new float[n];
            var negativeM = new float[n];
            for (int i = 0; i < n; i++)
            {
                float y = points[i].Position.Y;
                positive[i] = new Vector3(Number(edges[i][1]), y, Number(edges[i][2]));
                negative[i] = new Vector3(Number(edges[i][3]), y, Number(edges[i][4]));
                positiveM[i] = Number(edges[i][5]);
                negativeM[i] = Number(edges[i][6]);
            }
            TakeOutSpikes(positive, negative, positiveM, negativeM);
            var list = new List<Zone>();
            foreach (string[] z in zones)
            {
                bool outside = !z[0].Trim().Equals("touch and go", StringComparison.OrdinalIgnoreCase);
                int side = Number(z[1]) > 0 ? 1 : -1;
                list.Add(new Zone(outside, side, Mod((int)Number(z[2]), n), Mod((int)Number(z[3]), n)));
            }
            return new Course(positive, negative, list);
        }
        catch (Exception e) when (e is IOException or FormatException or OverflowException or UnauthorizedAccessException or InvalidDataException)
        {
            Log.Warning(e, "GradientDrift: {What} - could not read {Path}", what, PluginData.Describe(zonesFile));
            return null;
        }
    }

    // The zones' shapes, or null (the reason logged) when the files are missing, unreadable or not
    // on this AI line's points.
    public static List<Shape>? Load(string trackName, ReadOnlySpan<SplinePoint> points, float stripeSpacingM)
    {
        Course? course = LoadCourse(trackName, points);
        return course == null ? null : Shapes(course, stripeSpacingM);
    }

    public static List<Shape> Shapes(Course course, float stripeSpacingM)
    {
        var shapes = new List<Shape>();
        foreach (Zone z in course.Zones)
        {
            shapes.Add(Build(course.Positive, course.Negative, z.Side, z.First, z.Last, z.OutsideZone, stripeSpacingM));
        }
        VerboseLog.Information("GradientDrift: painting {Count} drift zones for players with CSP", shapes.Count);
        return shapes;
    }

    // One zone's frame (its outer edge forward, then its inner side back) and stripes, from the
    // course's edge points. side +1 = the positive (left) edge, -1 the negative (right) one; first to
    // last runs round the lap (past the start when first > last).
    public static Shape Build(Vector3[] positive, Vector3[] negative, int side, int first, int last, bool outsideZone,
        float stripeSpacingM)
    {
        int n = positive.Length;
        int count = Mod(last - first, n) + 1;
        Vector3[] edge = side > 0 ? positive : negative;
        Vector3[] other = side > 0 ? negative : positive;
        var outer = new Vector3[count];
        var inner = new Vector3[count];
        for (int k = 0; k < count; k++)
        {
            int p = (first + k) % n;
            outer[k] = edge[p];
            Vector3 inward = other[p] - edge[p];
            inward.Y = 0f;
            float width = inward.Length();
            inner[k] = width > 1e-6f ? edge[p] + inward * (MathF.Min(DepthM, width) / width) : edge[p];
        }
        var frame = new Vector3[2 * count];
        for (int k = 0; k < count; k++)
        {
            frame[k] = outer[k];
            frame[2 * count - 1 - k] = inner[k];
        }
        var along = new float[count];
        for (int k = 1; k < count; k++)
        {
            along[k] = along[k - 1] + Vector2.Distance(new Vector2(outer[k].X, outer[k].Z), new Vector2(outer[k - 1].X, outer[k - 1].Z));
        }
        float length = along[count - 1];
        float spacing = MathF.Max(0.3f, stripeSpacingM) * (outsideZone ? 1f : 2f);
        var stripes = new List<(Vector3, Vector3)>();
        for (float t = StripeMarginM; t + StripeSlantM <= length - StripeMarginM + 1e-4f; t += spacing)
        {
            stripes.Add((At(outer, along, t), At(inner, along, t + StripeSlantM)));
        }
        return new Shape(outsideZone, frame, stripes.ToArray());
    }

    // build_zones.smooth_edges: a point whose width either side is clipped (SmoothWidths) has that
    // edge moved along its cross-track line; the other points are left as they are.
    public static void TakeOutSpikes(Vector3[] positive, Vector3[] negative, float[] positiveM, float[] negativeM)
    {
        float[] newPositive = SmoothWidths(positiveM);
        float[] newNegative = SmoothWidths(negativeM);
        for (int i = 0; i < positive.Length; i++)
        {
            if (newPositive[i] == positiveM[i] && newNegative[i] == negativeM[i]) continue;
            var across = new Vector2(positive[i].X - negative[i].X, positive[i].Z - negative[i].Z);
            float width = across.Length();
            float sum = positiveM[i] + negativeM[i];
            if (width <= 1e-6f || sum <= 1e-6f) continue;
            Vector2 unit = across / width;
            Vector2 line = new Vector2(positive[i].X, positive[i].Z) - across * (positiveM[i] / sum);
            Vector2 p = line + unit * newPositive[i];
            Vector2 q = line - unit * newNegative[i];
            positive[i] = new Vector3(p.X, positive[i].Y, p.Y);
            negative[i] = new Vector3(q.X, negative[i].Y, q.Y);
        }
    }

    // build_zones.smooth_widths: one side's widths held within SpikeM of their running median over
    // SpikeWindow points round the lap, then a running mean over SmoothPoints.
    public static float[] SmoothWidths(float[] half)
    {
        int n = half.Length;
        int k = Math.Min(SpikeWindow / 2, (n - 1) / 2);
        var result = (float[])half.Clone();
        if (k < 1) return result;
        var window = new float[2 * k + 1];
        var held = new float[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = -k; j <= k; j++)
            {
                window[j + k] = half[Mod(i + j, n)];
            }
            Array.Sort(window);
            float median = window[k];
            held[i] = Math.Clamp(half[i], median - SpikeM, median + SpikeM);
        }
        int a = Math.Min(SmoothPoints / 2, (n - 1) / 2);
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = -a; j <= a; j++)
            {
                sum += held[Mod(i + j, n)];
            }
            float mean = (float)(sum / (2 * a + 1));
            if (MathF.Abs(mean - half[i]) > SmoothKeepM) result[i] = mean;
        }
        return result;
    }

    // The CSP online script that paints the shapes. colorHex: RRGGBB (a leading # is fine).
    public static string Script(IReadOnlyList<Shape> shapes, string colorHex, float lineWidthM)
    {
        (float r, float g, float b) = Color(colorHex);
        var c = CultureInfo.InvariantCulture;
        var s = new StringBuilder();
        s.AppendLine("-- Gradient Drift: the track's drift zones painted on the road, sent by the GradientDrift server");
        s.AppendLine("-- plugin (https://github.com/toyowheelin/gradient-drift, GradientDriftPlugin/DriftZones.cs).");
        s.AppendLine("if not ac.TrackPaint then return end -- a CSP without track paint: nothing to draw");
        s.AppendLine(string.Format(c, "local solid = rgbm({0:0.###}, {1:0.###}, {2:0.###}, 1)", r, g, b));
        s.AppendLine(string.Format(c, "local light = rgbm({0:0.###}, {1:0.###}, {2:0.###}, 0.75)", r, g, b));
        s.AppendLine(string.Format(c, "local width = {0:0.###}", Math.Clamp(lineWidthM, 0.05f, 1f)));
        s.AppendLine("local zones = {");
        foreach (Shape shape in shapes)
        {
            s.Append("  {o=").Append(shape.OutsideZone ? "true" : "false").Append(", f={");
            foreach (Vector3 p in shape.Frame)
            {
                s.AppendFormat(c, "{0:0.##},{1:0.##},{2:0.##},", p.X, p.Y, p.Z);
            }
            s.Append("}, s={");
            foreach ((Vector3 a, Vector3 b2) in shape.Stripes)
            {
                s.AppendFormat(c, "{0:0.##},{1:0.##},{2:0.##},{3:0.##},{4:0.##},{5:0.##},", a.X, a.Y, a.Z, b2.X, b2.Y, b2.Z);
            }
            s.AppendLine("}},");
        }
        s.AppendLine("}");
        s.AppendLine("""
            local paint = ac.TrackPaint()
            paint.ageFactor = 0.2
            -- the heights sent are the AI line's, which a cambered or crowned road rises above at its
            -- edges: cast every line onto the road's surface rather than trusting them
            paint.forceRecast = true
            for _, z in ipairs(zones) do
              local colour = z.o and solid or light
              local f = z.f
              for j = 1, #f, 3 do paint:to(vec3(f[j], f[j + 1], f[j + 2])) end
              paint:stroke(true, colour, width)
              local st = z.s
              for j = 1, #st, 6 do
                paint:line(vec3(st[j], st[j + 1], st[j + 2]), vec3(st[j + 3], st[j + 4], st[j + 5]), colour, width * 0.8)
              end
            end
            """);
        return s.ToString();
    }

    public static (float R, float G, float B) Color(string? hex)
    {
        string h = (hex ?? "").Trim().TrimStart('#');
        if (h.Length == 6 && int.TryParse(h, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int v))
        {
            return (((v >> 16) & 0xFF) / 255f, ((v >> 8) & 0xFF) / 255f, (v & 0xFF) / 255f);
        }
        Log.Warning("GradientDrift: ZoneColor {Color} is not RRGGBB - the zones are painted yellow", hex);
        return (0xF5 / 255f, 0xC4 / 255f, 0f);
    }

    private static Vector3 At(Vector3[] line, float[] along, float t)
    {
        int last = line.Length - 1;
        if (t <= 0f || last == 0) return line[0];
        if (t >= along[last]) return line[last];
        int k = Array.BinarySearch(along, t);
        if (k >= 0) return line[k];
        k = ~k;                              // along[k - 1] < t < along[k]
        float span = along[k] - along[k - 1];
        float f = span > 1e-6f ? (t - along[k - 1]) / span : 0f;
        return Vector3.Lerp(line[k - 1], line[k], f);
    }

    private static List<string[]> ReadTable(string file, params string[] columns)
    {
        using var reader = new StreamReader(PluginData.Open(file));
        string[] header = (reader.ReadLine() ?? "").Split(',').Select(h => h.Trim()).ToArray();
        int[] index = columns.Select(col => Array.IndexOf(header, col)).ToArray();
        if (index.Any(i => i < 0))
        {
            throw new InvalidDataException($"{file} lacks one of {string.Join(", ", columns)}");
        }
        var rows = new List<string[]>();
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.Trim().Length == 0) continue;
            string[] parts = line.Split(',');
            if (parts.Length <= index.Max())
            {
                throw new InvalidDataException($"{file}: a row with too few columns");
            }
            rows.Add(index.Select(i => parts[i]).ToArray());
        }
        return rows;
    }

    private static float Number(string text) => float.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static int Mod(int a, int n) => ((a % n) + n) % n;
}
