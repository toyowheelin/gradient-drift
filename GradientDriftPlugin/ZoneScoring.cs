using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using AssettoServer.Network.Tcp;
using AssettoServer.Server;
using AssettoServer.Server.Ai;
using AssettoServer.Server.Ai.Splines;
using AssettoServer.Server.Configuration;
using AssettoServer.Server.Plugin;
using AssettoServer.Shared.Network.Packets.Shared;
using Microsoft.Extensions.Hosting;
using Serilog;

namespace GradientDriftPlugin;

// Scores players' drifting through the track's zones (ZoneJudge) from the car status the server
// already gets from them, keeps each server's run records (ZoneRecords) and shows players how they
// are doing: a HUD drawn by a CSP online script (ZoneHud) that asks the server for its numbers over
// AssettoServer's HTTP port (ZoneScoreController), and a chat line for each scored run, which
// players without CSP see too. Chat commands: /zones, /top (ZoneCommands).
//
// It runs whether or not the bots have a policy. In Zones mode it needs the zones painted for
// players (<track>_course_zones.csv and <track>_course_edges.csv) and does nothing without them;
// in Arcade mode (ZoneScoreMode, ArcadeJudge) it needs no track files at all.
public sealed class ZoneScoring : BackgroundService, IAssettoServerAutostart
{
    private const int MaxEvents = 24;

    private readonly EntryCarManager _entryCarManager;
    private readonly Lazy<ACServer> _acServer;
    private readonly SessionManager _sessionManager;
    private readonly GradientDriftConfiguration _configuration;
    private readonly ZoneJudge.Course? _course;
    private readonly Dictionary<byte, Driver> _drivers = new();
    private readonly List<AiState> _aiStates = new();
    private readonly List<Vector2> _others = new();
    private readonly List<Vector2> _othersVelocity = new();
    private readonly List<byte> _othersId = new();
    private long _lastMs = -1;

    public ZoneRecords? Records { get; }
    public bool Arcade { get; }
    public bool On => _course != null || Arcade;
    public int ZoneCount => _course?.ZoneCount ?? 0;

    public ZoneScoring(EntryCarManager entryCarManager, Lazy<ACServer> acServer, SessionManager sessionManager,
        GradientDriftConfiguration configuration, ACServerConfiguration serverConfiguration, AiSpline spline,
        CSPServerScriptProvider cspServerScriptProvider)
    {
        _entryCarManager = entryCarManager;
        _acServer = acServer;
        _sessionManager = sessionManager;
        _configuration = configuration;
        if (!configuration.ZoneScoring)
        {
            return;
        }
        PluginData.Init(Path.GetDirectoryName(GetType().Assembly.Location) ?? "");
        string trackName = Path.GetFileName(serverConfiguration.Server.Track);
        string track = trackName + (string.IsNullOrWhiteSpace(serverConfiguration.Server.TrackConfig)
            ? "" : "/" + serverConfiguration.Server.TrackConfig.Trim());
        Arcade = configuration.IsArcadeScoring();
        DriftZones.Course? course = null;
        float[]? pace = null;
        if (!Arcade)
        {
            if (!string.Equals(configuration.ZoneScoreMode?.Trim(), "Zones", StringComparison.OrdinalIgnoreCase))
            {
                Log.Warning("GradientDrift: ZoneScoreMode {Mode} is neither Zones nor Arcade - scoring through the zones", configuration.ZoneScoreMode);
            }
            course = DriftZones.LoadCourse(trackName, spline.Points, "no zone scoring");
            if (course == null || course.Zones.Count == 0)
            {
                return;
            }
            pace = PaceReference.Load(trackName + PaceReference.FileSuffix);
            _course = new ZoneJudge.Course(course, pace);
        }
        if (!serverConfiguration.Extra.UseSteamAuth)
        {
            Log.Warning("GradientDrift: UseSteamAuth is off, so a player's Steam ID is not checked - anyone could keep runs under someone else's ID on the drift board (turn it on in extra_cfg.yml to stop that)");
        }
        if (!string.IsNullOrWhiteSpace(configuration.ZoneRecordsFile))
        {
            Records = new ZoneRecords(configuration.ZoneRecordsFile, track, course != null ? Fingerprint(course) : ArcadeBoard,
                configuration.ZoneBoardSize);
        }
        if (configuration.ZoneHud)
        {
            cspServerScriptProvider.AddScript(ZoneHud.Script(course?.Zones.Count ?? 0, Arcade), ZoneHud.ScriptName);
        }
        if (Arcade)
        {
            Log.Information("GradientDrift: scoring players' drifting arcade style (chains of angle x speed x combo, tandem bonus; zones not painted)");
        }
        else
        {
            Log.Information("GradientDrift: scoring players' drifting through {Count} zones ({Pace})", course!.Zones.Count,
                pace != null ? "speed against the track's pace reference" : $"no pace reference: {ZoneJudge.SpeedFallbackKmh:0} km/h is full speed points");
        }
    }

    // The board arcade chains are kept under (in place of a zones fingerprint): its own board.
    public const string ArcadeBoard = "arcade";

    // The zones as a short text: an edited zone (or new course edges) starts a new board.
    public static string Fingerprint(DriftZones.Course course)
    {
        var s = new StringBuilder().Append(course.Positive.Length);
        foreach (DriftZones.Zone z in course.Zones)
        {
            s.Append(';').Append(z.OutsideZone ? 'o' : 't').Append(z.Side).Append(',').Append(z.First).Append(',').Append(z.Last);
        }
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(s.ToString())))[..12].ToLowerInvariant();
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!On)
        {
            return Task.CompletedTask;
        }
        _entryCarManager.ClientConnected += (client, _) =>
        {
            client.Collision += OnCollision;
        };
        _entryCarManager.ClientDisconnected += (client, _) =>
        {
            client.Collision -= OnCollision;
            lock (_drivers) _drivers.Remove(client.SessionId);
        };
        _acServer.Value.Update += OnUpdate;
        return Task.CompletedTask;
    }

    public override void Dispose()
    {
        if (_acServer.IsValueCreated)
        {
            _acServer.Value.Update -= OnUpdate;
        }
        Records?.Dispose();
        base.Dispose();
    }

    private void OnCollision(ACTcpClient client, CollisionEventArgs args)
    {
        if (args.TargetCar != null) return;     // a wall, not a car (tandem contact is not marked down)
        Driver? d;
        lock (_drivers) _drivers.TryGetValue(client.SessionId, out d);
        if (d != null) Interlocked.Increment(ref d.WallHits);
    }

    private void OnUpdate(ACServer sender, EventArgs args)
    {
        try
        {
            Update();
        }
        catch (Exception e)
        {
            Log.Error(e, "GradientDrift: zone scoring stopped");
            _acServer.Value.Update -= OnUpdate;
        }
    }

    private void Update()
    {
        long now = _sessionManager.ServerTimeMilliseconds;
        float dt = _lastMs < 0 ? 0f : Math.Clamp((now - _lastMs) / 1000f, 0f, 0.5f);
        _lastMs = now;
        if (dt <= 0f || !On)
        {
            return;
        }
        // every moving car, for who is close behind whom
        _others.Clear();
        _othersVelocity.Clear();
        _othersId.Clear();
        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            if (car.AiControlled)
            {
                _aiStates.Clear();
                car.GetInitializedStates(_aiStates);
                foreach (AiState s in _aiStates)
                {
                    AddOther(car.SessionId, s.Status.Position, s.Status.Velocity);
                }
            }
            else if (car.Client?.HasSentFirstUpdate == true)
            {
                AddOther(car.SessionId, car.Status.Position, car.Status.Velocity);
            }
        }
        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            ACTcpClient? client = car.Client;
            if (car.AiControlled || client?.HasSentFirstUpdate != true)
            {
                continue;
            }
            Driver? d;
            lock (_drivers)
            {
                if (!_drivers.TryGetValue(car.SessionId, out d) || d.Guid != client.Guid)
                {
                    d = Arcade ? new Driver(client.Guid, null, new ArcadeJudge()) : new Driver(client.Guid, new ZoneJudge(_course!), null);
                    _drivers[car.SessionId] = d;
                }
            }
            var status = car.Status;
            float heading = status.Rotation.X + MathF.PI / 2f;
            var position = new Vector2(status.Position.X, status.Position.Z);
            var velocity = new Vector2(status.Velocity.X, status.Velocity.Z);
            int walls = Interlocked.Exchange(ref d.WallHits, 0);
            float gap = GapAhead(car.SessionId, position, velocity);
            List<ZoneJudge.Event> events;
            if (d.Arcade != null)
            {
                d.Arcade.Step(position.X, position.Y, heading, velocity.X, velocity.Y, dt, gap, walls);
                events = d.Arcade.Events;
            }
            else
            {
                d.Judge!.Step(position.X, position.Y, heading, velocity.X, velocity.Y, dt, gap <= ZoneJudge.TandemGapM, walls);
                events = d.Judge.Events;
            }
            if (events.Count > 0)
            {
                foreach (ZoneJudge.Event e in events)
                {
                    Told(car, client, d, e);
                }
                events.Clear();
            }
            d.Live = Live(d);
        }
    }

    private void AddOther(byte id, Vector3 position, Vector3 velocity)
    {
        var v = new Vector2(velocity.X, velocity.Z);
        if (v.LengthSquared() < 25f) return;
        _others.Add(new Vector2(position.X, position.Z));
        _othersVelocity.Add(v);
        _othersId.Add(id);
    }

    // How far behind the nearest other moving car close ahead of this one, going the same way, it
    // is (centre to centre; infinity when there is none within the tandem range).
    private float GapAhead(byte id, Vector2 position, Vector2 velocity)
    {
        float speed = velocity.Length();
        if (speed < 5f) return float.PositiveInfinity;
        Vector2 direction = velocity / speed;
        float range = MathF.Max(ZoneJudge.TandemGapM, ArcadeJudge.TandemGapM);
        float best = float.PositiveInfinity;
        for (int i = 0; i < _others.Count; i++)
        {
            if (_othersId[i] == id) continue;
            Vector2 rel = _others[i] - position;
            float d2 = rel.LengthSquared();
            if (d2 > range * range) continue;
            if (Vector2.Dot(rel, direction) > 1f && Vector2.Dot(_othersVelocity[i], direction) > 0f) best = MathF.Min(best, MathF.Sqrt(d2));
        }
        return best;
    }

    private void Told(EntryCar car, ACTcpClient client, Driver d, ZoneJudge.Event e)
    {
        var c = CultureInfo.InvariantCulture;
        int number = _course?.Number(e.Zone) ?? 0;
        lock (d)
        {
            d.Events.Add((++d.Seq, e, number));
            if (d.Events.Count > MaxEvents) d.Events.RemoveAt(0);
        }
        if (e.Kind != ZoneJudge.Kind.Run || (d.Arcade?.LastRun ?? d.Judge?.LastRun) is not { } run)
        {
            return;
        }
        string name = Name(client.Name, car.SessionId);
        string line = Arcade
            ? string.Format(c, "Drift chain {0:#,0} pts ({1}, angle {2:0}°, {3:0} km/h{4})", run.Score, e.Text, run.MeanAngleDeg,
                run.MeanSpeedOfPace * ArcadeJudge.SpeedRefKmh, run.MeanFill > 0.005f ? $", tandem {run.MeanFill * 100f:0}%" : "")
            : string.Format(c, "Zone run {0:0.0} pts (line {1:0}%, angle {2:0}°, speed {3:0}% of pace{4})",
                run.Score, run.MeanFill * 100f, run.MeanAngleDeg, run.MeanSpeedOfPace * 100f,
                run.Walls > 0 ? $", {run.Walls} wall hit{(run.Walls > 1 ? "s" : "")}" : "");
        string what = Arcade ? "drift chain" : "zone";
        string pts = Arcade ? "{0:#,0}" : "{0:0.0}";
        if (Records is not { Open: true })
        {
            if (_configuration.ZoneScoreChat) client.SendPacket(new ChatMessage { SessionId = 255, Message = line });
            return;
        }
        Records.Add(name, client.Guid.ToString(c), car.Model, run, (best, rank, record) =>
        {
            if (!client.IsConnected) return;
            if (_configuration.ZoneScoreChat)
            {
                (float top, _) = Records.Personal(client.Guid.ToString(c));
                client.SendPacket(new ChatMessage { SessionId = 255, Message = line + (best
                    ? $" - new best, #{rank} on this server"
                    : string.Format(c, " - best " + pts + ", #{1} on this server", top, rank)) });
            }
            if (record)
            {
                _entryCarManager.BroadcastPacket(new ChatMessage { SessionId = 255,
                    Message = string.Format(c, "{0} set this server's {1} record: " + pts.Replace("{0", "{2") + " pts", name, what, run.Score) });
            }
        });
    }

    private LiveView Live(Driver d)
    {
        (float best, int rank) = Records?.Personal(d.Guid.ToString(CultureInfo.InvariantCulture)) ?? (0f, 0);
        if (d.Arcade is { } a)
        {
            // arcade: fill = how close behind another car (0 - 1), points = the multiplier now,
            // run = the chain's points, run zones = its transitions, combo = the combo alone
            return new LiveView(a.Chaining, 0, a.TandemNow, a.Multiplier, a.AngleNowDeg, a.SpeedNowKmh,
                a.SpeedNowKmh / ArcadeJudge.SpeedRefKmh, a.Chaining ? a.ChainPoints : 0f, a.Transitions, best, rank, a.Combo);
        }
        ZoneJudge j = d.Judge!;
        return new LiveView(j.Running, j.CurrentZone >= 0 ? _course!.Number(j.CurrentZone) : 0, j.CurrentFill, j.CurrentPoints,
            j.AngleNowDeg, j.SpeedNowKmh, j.SpeedNowOfPace, j.RunPointsSoFar, j.RunZonesDone, best, rank, 1f);
    }

    // What the HUD asks for: the car's live numbers and what it was told after event since.
    public string LiveText(byte sessionId, int since)
    {
        Driver? d;
        lock (_drivers) _drivers.TryGetValue(sessionId, out d);
        var c = CultureInfo.InvariantCulture;
        var s = new StringBuilder();
        LiveView v = d?.Live ?? new LiveView(false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1f);
        s.AppendFormat(c, "L\t{0}\t{1}\t{2}\t{3:0.000}\t{4:0.00}\t{5:0.0}\t{6:0.0}\t{7:0.000}\t{8:0.0}\t{9}\t{10:0.0}\t{11}\t{12}\t{13}\t{14:0.00}\n",
            v.Running ? 1 : 0, v.Zone, ZoneCount, v.Fill, v.Points, v.AngleDeg, v.SpeedKmh, v.SpeedOfPace, v.RunPoints,
            v.RunZones, v.Best, v.Rank, Records?.Version ?? 0, Arcade ? "A" : "Z", v.Combo);
        if (d != null)
        {
            lock (d)
            {
                foreach ((int seq, ZoneJudge.Event e, int number) in d.Events)
                {
                    if (seq <= since) continue;
                    s.AppendFormat(c, "E\t{0}\t{1}\t{2}\t{3:0.0}\t{4:0.000}\t{5:0.0}\t{6:0.000}\t{7:0.000}\t{8}\n", seq,
                        e.Kind.ToString().ToLowerInvariant(), number, e.Points, e.Fill, e.AngleDeg, e.SpeedOfPace, e.Tandem,
                        Clean(e.Text));
                }
            }
        }
        return s.ToString();
    }

    public string BoardText()
    {
        var c = CultureInfo.InvariantCulture;
        var s = new StringBuilder();
        foreach (ZoneRecords.Entry e in Records?.Board ?? Array.Empty<ZoneRecords.Entry>())
        {
            s.AppendFormat(c, "B\t{0}\t{1}\t{2}\t{3:0.0}\t{4:0.0}\t{5:0.000}\n", e.Rank, Clean(e.Driver), Clean(e.Car), e.Score,
                e.AngleDeg, e.SpeedOfPace);
        }
        return s.ToString();
    }

    // The board as a web page for people (/gradientdrift/leaderboard): the HUD reads BoardText.
    public string BoardHtml()
    {
        var c = CultureInfo.InvariantCulture;
        static string H(string t) => System.Net.WebUtility.HtmlEncode(t);
        var s = new StringBuilder();
        s.Append("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">")
         .Append("<meta http-equiv=\"refresh\" content=\"30\"><title>").Append(Arcade ? "Drift board" : "Drift zone board").Append("</title><style>")
         .Append("body{font-family:system-ui,sans-serif;background:#111318;color:#eee;margin:24px}h1{color:#f5c400;font-size:22px}")
         .Append("table{border-collapse:collapse;min-width:min(640px,100%)}th,td{padding:6px 12px;text-align:left;border-bottom:1px solid #333}")
         .Append("th{color:#aaa;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}tr:first-child td{color:#ffd633}")
         .Append(".note{color:#999}</style></head><body><h1>").Append(Arcade ? "Drift board" : "Drift zone board").Append("</h1>");
        IReadOnlyList<ZoneRecords.Entry> board = Records?.Board ?? Array.Empty<ZoneRecords.Entry>();
        if (Records is not { Open: true })
        {
            s.Append("<p class=\"note\">Runs are scored on this server but not kept (no SQLite database: see the server log).</p>");
        }
        else if (board.Count == 0)
        {
            s.Append(Arcade
                ? "<p class=\"note\">No drift chains yet. A chain is banked once the car stops sliding; it shows here once someone banks one.</p>"
                : string.Format(c, "<p class=\"note\">No scored runs yet. A run goes from the first of the {0} zones to the last; it shows here once someone finishes one.</p>", ZoneCount));
        }
        else
        {
            s.Append("<table><tr><th>#</th><th>Driver</th><th>Car</th><th>Score</th><th>Angle</th><th>")
             .Append(Arcade ? "Speed" : "Speed of pace").Append("</th><th>When (UTC)</th></tr>");
            foreach (ZoneRecords.Entry e in board)
            {
                if (Arcade)
                {
                    s.AppendFormat(c, "<tr><td>{0}</td><td>{1}</td><td>{2}</td><td class=\"n\">{3:#,0}</td><td class=\"n\">{4:0}&deg;</td><td class=\"n\">{5:0} km/h</td><td>{6}</td></tr>",
                        e.Rank, H(e.Driver), H(e.Car), e.Score, e.AngleDeg, e.SpeedOfPace * ArcadeJudge.SpeedRefKmh, H(e.When));
                    continue;
                }
                s.AppendFormat(c, "<tr><td>{0}</td><td>{1}</td><td>{2}</td><td class=\"n\">{3:0.0}</td><td class=\"n\">{4:0}&deg;</td><td class=\"n\">{5:0}%</td><td>{6}</td></tr>",
                    e.Rank, H(e.Driver), H(e.Car), e.Score, e.AngleDeg, e.SpeedOfPace * 100f, H(e.When));
            }
            s.Append("</table>");
        }
        s.Append(Arcade
            ? "<p class=\"note\">Each driver's best drift chain on this track (arcade scoring). Refreshes every 30 s.</p></body></html>"
            : string.Format(c, "<p class=\"note\">Each driver's best run on this track with these {0} zones. Refreshes every 30 s.</p></body></html>", ZoneCount));
        return s.ToString();
    }

    // A player's name as kept and shown: what their game sent, without control characters, at most
    // NameLength characters.
    public const int NameLength = 32;
    public static string Name(string? sent, byte sessionId)
    {
        string name = new string((sent ?? "").Where(ch => !char.IsControl(ch)).ToArray()).Trim();
        if (name.Length > NameLength) name = name[..NameLength];
        return name.Length > 0 ? name : $"car {sessionId}";
    }

    private static string Clean(string text) => text.Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ');

    private sealed record LiveView(bool Running, int Zone, float Fill, float Points, float AngleDeg, float SpeedKmh,
        float SpeedOfPace, float RunPoints, int RunZones, float Best, int Rank, float Combo);

    private sealed class Driver
    {
        public readonly ulong Guid;
        public readonly ZoneJudge? Judge;
        public readonly ArcadeJudge? Arcade;
        public int WallHits;
        public int Seq;
        public readonly List<(int Seq, ZoneJudge.Event Event, int Number)> Events = new();
        public volatile LiveView? Live;

        public Driver(ulong guid, ZoneJudge? judge, ArcadeJudge? arcade)
        {
            Guid = guid;
            Judge = judge;
            Arcade = arcade;
        }
    }
}
