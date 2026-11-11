// The plugin checker: runs the C# plugin's policy loader and network on the cases
// tests/plugin/make_cases.py writes, and fails (exit 1) when
//   - a policy the plugin must run is refused, or gives other actions than the Python policy does
//     (the trainer's and every tool's copy of the network) for the same inputs, or
//   - a policy the plugin must refuse (broken, or needing a setting this build cannot apply) loads, or
//   - a car-on-car push (CarContact) differs from the trainer's, or
//   - the drift zones painted for players (DriftZones) are not the shapes setup's zones.py makes, or
//   - players' zone scores (ZoneJudge) fill zones otherwise than setup's zones.score, or the run
//     records (ZoneRecords) do not keep and rank runs, or
//   - arcade drift scoring (ArcadeJudge) does not bank a drifted chain, or keeps one after a spin,
//     a wall hit or car data that does not add up.
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
    List<DriftZones.Shape>? shapes = DriftZones.Load(track, trackPoints, (float)zoneCase.GetProperty("spacing").GetDouble(),
        (float)zoneCase.GetProperty("line_width").GetDouble());
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
    string script = shapes != null ? DriftZones.Script(shapes, "#F5C400", (float)zoneCase.GetProperty("line_width").GetDouble()) : "";
    if (problem == "" && !(worst <= 0.02f))
        problem = $"shapes differ from setup's by up to {worst:G3} m";
    if (problem == "" && (!script.Contains("ac.TrackPaint()") || script.Split("{o=").Length - 1 != shapes!.Count))
        problem = "the script does not paint every zone";
    bool ok = problem == "";
    Console.WriteLine((ok ? "ok   " : "FAIL ") + $"DriftZones: " + (ok ? $"{shapes!.Count} zones painted as setup draws them (to within {worst:G3} m), script {script.Length} characters" : problem));
    if (!ok) failures++;
}

// ZoneScoring (ZoneJudge.cs): made-up laps of the same course judged zone by zone, each zone's
// line fill the one setup's zones.score gives it; a sliding lap is one scored run, a straight one
// scores nothing. Then the run records (ZoneRecords.cs, the system's SQLite) keep runs and rank
// each driver's best.
if (cases.RootElement.TryGetProperty("zones", out JsonElement judgeCase) && judgeCase.TryGetProperty("runs", out JsonElement judgeRuns))
{
    string track = judgeCase.GetProperty("track").GetString()!;
    var line = judgeCase.GetProperty("line").EnumerateArray()
        .Select(p => p.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray()).ToArray();
    var trackPoints = line.Select((p, i) => new SplinePoint
    {
        Id = i, Position = new Vector3(p[0], p[1], p[2]), NextId = (i + 1) % line.Length,
        PreviousId = (i + line.Length - 1) % line.Length, JunctionStartId = -1, JunctionEndId = -1, LeftId = -1, RightId = -1,
    }).ToArray();
    PluginData.Init(Path.Combine(folder, "zones"));
    string before = Directory.GetCurrentDirectory();
    Directory.SetCurrentDirectory(Path.Combine(folder, "zones"));
    DriftZones.Course? course = DriftZones.LoadCourse(track, trackPoints);
    Directory.SetCurrentDirectory(before);
    foreach (JsonElement run in judgeRuns.EnumerateArray())
    {
        string problem = "";
        float worst = 0f;
        bool sliding = run.GetProperty("sliding").GetBoolean();
        float[] F(string name) => run.GetProperty(name).EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        float[] x = F("x"), z = F("z"), rx = F("rear_x"), rz = F("rear_z"), slip = F("slip"), vx = F("vx"), vz = F("vz");
        float dt = (float)run.GetProperty("dt").GetDouble();
        var fills = run.GetProperty("fills").EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? float.NaN : (float)v.GetDouble()).ToArray();
        var judged = new Dictionary<int, float>();
        ZoneJudge.Event? result = null;
        if (course == null)
        {
            problem = "the course did not load";
        }
        else
        {
            var judge = new ZoneJudge(new ZoneJudge.Course(course, null));
            for (int i = 0; i < x.Length; i++)
            {
                judge.StepAt(new Vector2(x[i], z[i]), new Vector2(rx[i], rz[i]), slip[i], new Vector2(vx[i], vz[i]), dt, false, 0);
                foreach (ZoneJudge.Event e in judge.Events)
                {
                    if (e.Kind is ZoneJudge.Kind.Zone or ZoneJudge.Kind.Spin) judged[e.Zone] = e.Fill;
                    if (e.Kind == ZoneJudge.Kind.Run) result = e;
                    if (e.Kind == ZoneJudge.Kind.Ended && sliding) problem = $"the run ended: {e.Text}";
                }
                judge.Events.Clear();
            }
            for (int k = 0; k < fills.Length && problem == ""; k++)
            {
                if (float.IsNaN(fills[k])) continue;
                if (!judged.TryGetValue(k, out float fill)) { problem = $"zone {k + 1} was not judged"; break; }
                worst = MathF.Max(worst, MathF.Abs(fill - fills[k]));
            }
            if (problem == "" && worst > 0.051f) problem = $"zone fills differ from setup's zones.score by up to {worst:G3}";
            if (problem == "" && sliding && result == null) problem = "no scored run";
            if (problem == "" && sliding && !(result!.Value.Points > 1f && result.Value.Points <= 110f)) problem = $"a run score of {result!.Value.Points}";
            if (problem == "" && !sliding && result != null) problem = $"a straight lap scored {result.Value.Points}";
        }
        if (problem == "" && sliding && course != null)
        {
            // the same lap with a speed five times what the car moves: made-up numbers, not scored
            var judge = new ZoneJudge(new ZoneJudge.Course(course, null));
            bool scored = false, said = false;
            for (int i = 0; i < x.Length; i++)
            {
                judge.StepAt(new Vector2(x[i], z[i]), new Vector2(rx[i], rz[i]), slip[i], 5f * new Vector2(vx[i], vz[i]), dt, false, 0);
                scored |= judge.Events.Any(e => e.Kind == ZoneJudge.Kind.Run);
                said |= judge.Events.Any(e => e.Kind == ZoneJudge.Kind.Ended && e.Text.Contains("add up"));
                judge.Events.Clear();
            }
            if (scored || !said) problem = "a lap whose speed does not match its movement was scored";
        }
        bool ok = problem == "";
        Console.WriteLine((ok ? "ok   " : "FAIL ") + $"ZoneJudge ({(sliding ? "sliding" : "straight")} lap): "
            + (ok ? $"{judged.Count} zones judged as zones.score fills them (to within {worst:G3}), " + (result is { } res ? $"run {res.Points:0.0} pts; the same lap with a made-up speed not scored" : "not scored as a run") : problem));
        if (!ok) failures++;
    }

    string db = Path.Combine(Path.GetTempPath(), $"zone_records_{Environment.ProcessId}.db");
    string recordsProblem = "";
    var r = new ZoneJudge.RunResult(0f, new[] { 50f }, 35f, 0.9f, 0.5f, 0);
    var records = new ZoneRecords(db, "test", "zones1", 10);
    if (!records.Open)
    {
        recordsProblem = "the SQLite database did not open";
    }
    else
    {
        records.Add("A", "1", "car_a", r with { Score = 50f }, null);
        records.Add("A", "1", "car_a", r with { Score = 70f }, null);
        records.Add("B", "2", "car_b", r with { Score = 60f }, null);
        records.Dispose();
        var again = new ZoneRecords(db, "test", "zones1", 10);
        var other = new ZoneRecords(db, "test", "zones2", 10);
        string board = string.Join(", ", again.Board.Select(e => $"#{e.Rank} {e.Driver} {e.Score:0}"));
        if (board != "#1 A 70, #2 B 60") recordsProblem = $"the board reads {board}";
        else if (again.Personal("2") != (60f, 2)) recordsProblem = "B's best and place are wrong";
        else if (other.Board.Count != 0) recordsProblem = "other zones' board is not empty";
        again.Dispose();
        other.Dispose();
    }
    foreach (string f in new[] { db, db + "-wal", db + "-shm" }) File.Delete(f);
    Console.WriteLine((recordsProblem == "" ? "ok   " : "FAIL ") + "ZoneRecords: " + (recordsProblem == "" ? "runs kept, each driver's best ranked, a board per set of zones" : recordsProblem));
    if (recordsProblem != "") failures++;
}

// ArcadeJudge: a car sliding along at 72 km/h, 30 degrees one way for 3 s, then the other way,
// then straight. Banked once it straightens, with one transition; sliding 2 m behind another car
// scores twice as much; a spin, a wall hit or a position pushed along loses the chain.
{
    const float dt = 0.05f, v = 20f;
    (List<ZoneJudge.Event> events, ArcadeJudge judge) Drive(Func<float, float> heading, float gap = float.PositiveInfinity,
        float wallAt = -1f, float push = 1f)
    {
        var judge = new ArcadeJudge();
        var all = new List<ZoneJudge.Event>();
        for (float t = 0f; t < 10f; t += dt)
        {
            int walls = wallAt >= 0f && t >= wallAt && t < wallAt + dt ? 1 : 0;
            judge.Step(v * t * push, 0f, heading(t) * MathF.PI / 180f, v, 0f, dt, gap, walls);
            all.AddRange(judge.Events);
            judge.Events.Clear();
        }
        return (all, judge);
    }
    static float Slide(float t) => t < 3f ? 30f : t < 6f ? -30f : 0f;
    string arcadeProblem = "";
    var (plain, _) = Drive(Slide);
    var banked = plain.Where(e => e.Kind == ZoneJudge.Kind.Run).ToList();
    float expected = 100f * (30f / 45f) * (72f / 80f) * (3f + 3f * 1.25f);
    if (banked.Count != 1) arcadeProblem = $"{banked.Count} chains banked from one drift";
    else if (MathF.Abs(banked[0].Points - expected) > 0.05f * expected) arcadeProblem = $"chain {banked[0].Points:0} pts, expected about {expected:0}";
    else if (plain.Count(e => e.Kind == ZoneJudge.Kind.Combo) != 1) arcadeProblem = "the transition was not counted once";
    else
    {
        var (close, _) = Drive(Slide, gap: 2f);
        float tandem = close.Where(e => e.Kind == ZoneJudge.Kind.Run).Select(e => e.Points).DefaultIfEmpty(0f).First();
        var (spun, _) = Drive(t => t < 3f ? 30f : 150f);
        var (wall, _) = Drive(Slide, wallAt: 4f);
        var (pushed, _) = Drive(Slide, push: 5f);
        var (straight, _) = Drive(_ => 0f);
        if (MathF.Abs(tandem - 2f * banked[0].Points) > 0.05f * tandem) arcadeProblem = $"2 m behind a car scored {tandem:0}, not twice {banked[0].Points:0}";
        else if (!spun.Any(e => e.Kind == ZoneJudge.Kind.Spin) || spun.Any(e => e.Kind == ZoneJudge.Kind.Run)) arcadeProblem = "a spin did not lose the chain";
        else if (!wall.Any(e => e.Kind == ZoneJudge.Kind.Wall) || wall.Where(e => e.Kind == ZoneJudge.Kind.Run).Any(e => e.Points > 0.6f * banked[0].Points))
            arcadeProblem = "a wall hit did not lose the chain";
        else if (pushed.Any(e => e.Kind == ZoneJudge.Kind.Run)) arcadeProblem = "a chain with a made-up position was banked";
        else if (straight.Count != 0) arcadeProblem = "driving straight was scored";
    }
    Console.WriteLine((arcadeProblem == "" ? "ok   " : "FAIL ") + "ArcadeJudge: " + (arcadeProblem == ""
        ? $"a 6 s drift with one transition banks {banked[0].Points:0} pts, twice that in tandem; spin, wall and made-up data lose the chain"
        : arcadeProblem));
    if (arcadeProblem != "") failures++;
}

Console.WriteLine(failures == 0 ? "plugin checks passed" : $"{failures} plugin check(s) failed");
return failures == 0 ? 0 : 1;
