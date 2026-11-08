// The plugin checker: runs the C# plugin's policy loader and network on the cases
// tests/plugin/make_cases.py writes, and fails (exit 1) when
//   - a policy the plugin must run is refused, or gives other actions than the Python policy does
//     (the trainer's and every tool's copy of the network) for the same inputs, or
//   - a policy the plugin must refuse (broken, or needing a setting this build cannot apply) loads, or
//   - a car-on-car push (CarContact) differs from the trainer's, or
//   - the drift zones painted for players (DriftZones) are not the shapes setup's zones.py makes.
//
//     dotnet run --project PluginChecks.csproj -- <cases folder>
using System.Numerics;
using System.Text.Json;
using AssettoServer.Server.Ai.Splines;
using GradientDriftPlugin;

if (args.Length != 1)
{
    Console.Error.WriteLine("usage: PluginChecks <folder with cases.json>");
    return 2;
}
string folder = args[0];
using JsonDocument cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "cases.json")));
float tolerance = (float)cases.RootElement.GetProperty("action_tolerance").GetDouble();

// a round AI line, 50 m across: the loader builds its look-ahead tables from the track's points
const int pointCount = 200;
const float radius = 50f;
var points = new SplinePoint[pointCount];
float step = 2f * MathF.PI * radius / pointCount;
for (int i = 0; i < pointCount; i++)
{
    float angle = 2f * MathF.PI * i / pointCount;
    points[i] = new SplinePoint
    {
        Id = i,
        Position = new Vector3(radius * MathF.Cos(angle), 0f, radius * MathF.Sin(angle)),
        Radius = radius,
        Length = step,
        NextId = (i + 1) % pointCount,
        PreviousId = (i + pointCount - 1) % pointCount,
        JunctionStartId = -1,
        JunctionEndId = -1,
        LeftId = -1,
        RightId = -1,
    };
}

int failures = 0;
foreach (JsonElement c in cases.RootElement.GetProperty("policies").EnumerateArray())
{
    string file = c.GetProperty("file").GetString()!;
    string expect = c.GetProperty("expect").GetString()!;
    string why = c.GetProperty("why").GetString()!;
    DriftPolicy? policy = DriftPolicy.TryLoad(Path.Combine(folder, file), points);
    string? badSetting = policy?.PluginSettings.FirstOrDefault(s => !DriftPolicy.CanApplySetting(s.Key, s.Value)).Key;
    bool runs = policy != null && badSetting == null;

    if (expect == "refused")
    {
        if (runs)
        {
            Console.WriteLine($"FAIL {file}: loaded, but the plugin must refuse it ({why})");
            failures++;
        }
        else
        {
            Console.WriteLine($"ok   {file}: refused ({why})" + (badSetting != null ? $" - setting {badSetting}" : ""));
        }
        continue;
    }

    if (!runs)
    {
        Console.WriteLine($"FAIL {file}: " + (policy == null ? "refused by the loader" : $"setting {badSetting} cannot be applied")
            + $", but the plugin must run it ({why})");
        failures++;
        continue;
    }

    float worst = 0f;
    int worstRow = -1, worstAction = -1;
    float[] expected = new float[3];
    int row = 0;
    using var obsRows = c.GetProperty("obs").EnumerateArray();
    using var actionRows = c.GetProperty("actions").EnumerateArray();
    foreach ((JsonElement obsRow, JsonElement actionRow) in obsRows.Zip(actionRows))
    {
        float[] obs = obsRow.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        int k = 0;
        foreach (JsonElement v in actionRow.EnumerateArray())
            expected[k++] = (float)v.GetDouble();
        DriftPolicyAction action = policy!.Evaluate(obs);
        float[] got = { action.Steer, action.Speed, action.Brake };
        for (int a = 0; a < 3; a++)
        {
            float diff = MathF.Abs(got[a] - expected[a]);
            if (!(diff <= worst))
            {
                worst = float.IsNaN(diff) ? float.PositiveInfinity : diff;
                worstRow = row;
                worstAction = a;
            }
        }
        row++;
    }
    // a case may allow more: what float32 arithmetic itself moves that network's actions (make_cases.py)
    float allowed = c.TryGetProperty("action_tolerance", out JsonElement own) ? (float)own.GetDouble() : tolerance;
    if (worst > allowed)
    {
        Console.WriteLine($"FAIL {file}: actions differ from Python's by up to {worst:G3} (input row {worstRow}, action {worstAction}; allowed {allowed:G3}) ({why})");
        failures++;
    }
    else
    {
        Console.WriteLine($"ok   {file}: {row} inputs, actions within {worst:G3} of Python's ({why})");
    }
}

// AdaptToPlayers (PlayerAdaptation.cs): bots move to the players while they drift, keep their
// differences, stay in the trained range, and are back on the settings once the server is empty
{
    const float dt = 0.05f;
    var adapt = new PlayerAdaptation(1f, 60f, 120f, 60f);
    var fast = new BotStyle(0.5f, 1.0f, 2f, 1f, 7.5f, 1.1f);
    var slow = fast with { PaceMiddle = 0.9f };
    void Expect(bool ok, string what)
    {
        Console.WriteLine((ok ? "ok   " : "FAIL ") + "AdaptToPlayers: " + what);
        if (!ok) failures++;
    }
    Expect(adapt.Apply(fast, 1f, 0.8f, 1.2f, 2f, 0.5f, 6f) == fast, "nothing seen yet: the style is the settings'");
    for (int t = 0; t < 1200; t++)          // 60 s of players drifting at pace 1.1, following at a 4 m ask
    {
        adapt.PaceSample(1.1f);
        adapt.PaceSample(3f);                // not a drifting pace: thrown away
        adapt.GapSample(4f);
        adapt.EndTick(dt, true);
    }
    var a = adapt.Apply(fast, 1f, 0.8f, 1.2f, 2f, 0.5f, 6f);
    var b = adapt.Apply(slow, 1f, 0.8f, 1.2f, 2f, 0.5f, 6f);
    Expect(MathF.Abs(adapt.PaceWeight - 1f) < 1e-3f && MathF.Abs(a.PaceMiddle - 1.1f) < 1e-3f && MathF.Abs(b.PaceMiddle - 1.0f) < 1e-3f,
        $"after 60 s the pace follows the players and the bots keep their difference ({a.PaceMiddle:F3}, {b.PaceMiddle:F3})");
    Expect(MathF.Abs(a.ChaseGapMeters - 4f) < 1e-3f, $"chase gap follows the players ({a.ChaseGapMeters:F2} m)");
    Expect(adapt.Apply(fast, 1f, 0.8f, 1.05f, 2f, 0.5f, 3f) is { PaceMiddle: 1.05f, ChaseGapMeters: 3f }, "held inside the trained range");
    bool forgotten = false;
    for (int t = 0; t < 1200; t++)          // 60 s with players on but none drifting: held
        forgotten |= adapt.EndTick(dt, true);
    Expect(!forgotten && adapt.PaceWeight > 0.99f, "players on but not drifting: kept");
    int forgets = 0;
    for (int t = 0; t < 2400 + 20; t++)     // 120 s with the server empty
        forgets += adapt.EndTick(dt, false) ? 1 : 0;
    Expect(forgets == 1 && adapt.PaceWeight == 0f && float.IsNaN(adapt.Pace) && adapt.Apply(fast, 1f, 0.8f, 1.2f, 2f, 0.5f, 6f) == fast,
        "server empty for AdaptToPlayersForgetSeconds: back on the settings, averages dropped");
}

// CarContactPush (CarContact.cs): the same push as the trainer's (drift_env._touch_push) on
// make_cases.py's overlapping cars
if (cases.RootElement.TryGetProperty("contacts", out JsonElement contacts))
{
    float worst = 0f;
    int count = 0;
    foreach (JsonElement c in contacts.EnumerateArray())
    {
        float[] car = c.GetProperty("car").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        float[] lead = c.GetProperty("lead").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        float[] after = c.GetProperty("after").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        float mass = (float)c.GetProperty("mass").GetDouble();
        float inertia = (float)c.GetProperty("inertia").GetDouble();
        var a = new CarContact.Body
        {
            X = car[0], Z = car[1], Heading = car[2], Forward = car[3], Lateral = car[4], YawRate = car[5],
            MassKg = mass, InertiaKgM2 = inertia,
        };
        float cs = MathF.Cos(lead[2]), sn = MathF.Sin(lead[2]);
        var b = new CarContact.Body
        {
            X = lead[0], Z = lead[1], Heading = lead[2],
            Forward = lead[3] * cs + lead[4] * sn, Lateral = -lead[3] * sn + lead[4] * cs,
            MassKg = mass, InertiaKgM2 = inertia,
        };
        float kick = MathF.Abs(CarContact.Collide(ref a, ref b)) * (180f / MathF.PI);
        float[] got = { a.Forward, a.Lateral, a.YawRate, kick };
        for (int k = 0; k < 4; k++)
        {
            // speeds to 1 mm/s, yaw rate to 0.001 rad/s, the kick to 0.05 deg/s (float32)
            float scale = k == 3 ? 50f : 1f;
            float diff = MathF.Abs(got[k] - after[k]) / scale;
            if (!(diff <= worst))
                worst = float.IsNaN(diff) ? float.PositiveInfinity : diff;
        }
        count++;
    }
    bool ok = count > 0 && worst <= 1e-3f;
    Console.WriteLine((ok ? "ok   " : "FAIL ") + $"CarContact: {count} pushes match the trainer's to within {worst:G3}");
    if (!ok) failures++;
}

// ShowZones (DriftZones.cs): the painted frames and stripes of a generated course's zones, the same
// as setup's zones.paint_shapes, at the AI line's heights; and a script that paints them
if (cases.RootElement.TryGetProperty("zones", out JsonElement zoneCase))
{
    string track = zoneCase.GetProperty("track").GetString()!;
    var line = zoneCase.GetProperty("line").EnumerateArray()
        .Select(p => p.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray()).ToArray();
    var trackPoints = new SplinePoint[line.Length];
    for (int i = 0; i < line.Length; i++)
    {
        trackPoints[i] = new SplinePoint
        {
            Id = i, Position = new Vector3(line[i][0], line[i][1], line[i][2]),
            NextId = (i + 1) % line.Length, PreviousId = (i + line.Length - 1) % line.Length,
            JunctionStartId = -1, JunctionEndId = -1, LeftId = -1, RightId = -1,
        };
    }
    PluginData.Init(Path.Combine(folder, "zones"));
    string before = Directory.GetCurrentDirectory();
    Directory.SetCurrentDirectory(Path.Combine(folder, "zones"));       // the track's files sit in the server folder
    List<DriftZones.Shape>? shapes = DriftZones.Load(track, trackPoints, (float)zoneCase.GetProperty("spacing").GetDouble());
    Directory.SetCurrentDirectory(before);
    var expected = zoneCase.GetProperty("shapes").EnumerateArray().ToArray();
    string problem = "";
    float worst = 0f;
    if (shapes == null || shapes.Count != expected.Length)
    {
        problem = $"{shapes?.Count.ToString() ?? "no"} zones loaded, {expected.Length} expected";
    }
    else
    {
        for (int z = 0; z < shapes.Count && problem == ""; z++)
        {
            var frame = expected[z].GetProperty("frame").EnumerateArray().Select(p => p.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray()).ToArray();
            var stripes = expected[z].GetProperty("stripes").EnumerateArray().ToArray();
            if (shapes[z].OutsideZone != expected[z].GetProperty("outside").GetBoolean()
                || shapes[z].Frame.Length != frame.Length || shapes[z].Stripes.Length != stripes.Length)
            {
                problem = $"zone {z + 1}: kind, frame points ({shapes[z].Frame.Length} vs {frame.Length}) or stripes ({shapes[z].Stripes.Length} vs {stripes.Length}) differ";
                break;
            }
            for (int k = 0; k < frame.Length; k++)
            {
                worst = MathF.Max(worst, MathF.Max(MathF.Abs(shapes[z].Frame[k].X - frame[k][0]), MathF.Abs(shapes[z].Frame[k].Z - frame[k][1])));
            }
            for (int k = 0; k < stripes.Length; k++)
            {
                var ends = stripes[k].EnumerateArray().Select(p => p.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray()).ToArray();
                (Vector3 a, Vector3 b) = shapes[z].Stripes[k];
                worst = MathF.Max(worst, new[] { a.X - ends[0][0], a.Z - ends[0][1], b.X - ends[1][0], b.Z - ends[1][1] }.Max(d => MathF.Abs(d)));
            }
            float lineY = shapes[z].Frame[0].Y;
            if (!trackPoints.Any(p => MathF.Abs(p.Position.Y - lineY) < 1e-4f))
            {
                problem = $"zone {z + 1}: painted at a height that is not the AI line's";
            }
        }
    }
    string script = shapes != null ? DriftZones.Script(shapes, "#F5C400", 0.3f) : "";
    if (problem == "" && !(worst <= 0.02f))
        problem = $"shapes differ from setup's by up to {worst:G3} m";
    if (problem == "" && (!script.Contains("ac.TrackPaint()") || script.Split("{o=").Length - 1 != shapes!.Count))
        problem = "the script does not paint every zone";
    bool ok = problem == "";
    Console.WriteLine((ok ? "ok   " : "FAIL ") + $"DriftZones: " + (ok ? $"{shapes!.Count} zones painted as setup draws them (to within {worst:G3} m), script {script.Length} characters" : problem));
    if (!ok) failures++;
}

Console.WriteLine(failures == 0 ? "plugin checks passed" : $"{failures} plugin check(s) failed");
return failures == 0 ? 0 : 1;
