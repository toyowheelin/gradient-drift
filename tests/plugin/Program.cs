// The plugin checker: runs the C# plugin's policy loader and network on the cases
// tests/plugin/make_cases.py writes, and fails (exit 1) when
//   - a policy the plugin must run is refused, or gives other actions than the Python policy does
//     (the trainer's and every tool's copy of the network) for the same inputs, or
//   - a policy the plugin must refuse (broken, or needing a setting this build cannot apply) loads.
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

Console.WriteLine(failures == 0 ? "plugin checks passed" : $"{failures} plugin check(s) failed");
return failures == 0 ? 0 : 1;
