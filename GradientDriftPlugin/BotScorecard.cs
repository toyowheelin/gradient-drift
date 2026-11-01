using System.Globalization;
using System.Text;
using System.Text.Json;
using Serilog;

namespace GradientDriftPlugin;

// EnableScorecard: a short report card of how the bots drove, one per server session, so a server
// run reads as a test result without anyone watching it. Written as <start>_<track>_<session>.txt
// (a table to read) and .json (the same numbers for drift_rl/scorecard_report.py) in
// ScorecardFolder, rewritten every ScorecardWriteMinutes while the session runs and once more when
// it ends or the server stops, so a crash loses at most that much.
//
// Every number is in the terms drift_rl/drive_report.py and chase_report.py use in the simulator:
//   moving    over 25 km/h
//   slide     share of moving time with the slip angle 20 - 80 deg; spin: 80 deg or more
//   walls     share of moving time pushed out of a wall; wall hits: separate touches (0.5 s apart)
//   car hits  separate touches of the bot's outline with another car's (0.5 s apart), bots or players
//   chase     time chasing a car ahead (ChasePlayers), the gap to it (the outline gap for a tandem
//             policy, along the line otherwise) and the share of chase time within 2 m of ChaseGapMeters
//   stock     share of the bot's time the server's own AI drove it: before the plugin takes a car
//             over, while the server holds it stopped, or ticks the policy's output was unusable
//   stuck     BotHealth's stuck episodes (30 s without getting anywhere)
//
// Everything here runs on the server's update thread except Write, which the server's shutdown
// can call from another; one lock covers both.
internal sealed class BotScorecard
{
    private const float MovingMs = 25f / 3.6f;
    private const float SlideLowRad = 20f * MathF.PI / 180f;
    private const float SlideHighRad = 80f * MathF.PI / 180f;
    private const long TouchGapMs = 500;
    private const float GapBinM = 0.25f;
    private const int GapBins = 200;                 // 0 - 50 m; further is the last bin
    private const float InBandM = 2f;

    private sealed class Bot
    {
        public double Seconds, WaitingSeconds, HeldSeconds, InvalidSeconds, FollowingSeconds;
        public double MovingSeconds, SlideSeconds, SpinSeconds, SlideSlipSum, WallSeconds, DistanceM;
        public int WallHits, BotHits, PlayerHits, WrongWays, Takeovers, Stuck;
        public long LastWallMs = long.MinValue / 2;
        public bool WrongWay;
        public double ChaseSeconds, ChaseBotSeconds, InBandSeconds;
        public readonly double[] Gaps = new double[GapBins];
    }

    private readonly object _lock = new();
    private readonly object _fileLock = new();
    private readonly string _folder;
    private readonly string _track;
    private readonly string _policy;
    private readonly float _askedGapM;
    private readonly SortedDictionary<byte, Bot> _bots = new();
    private readonly Dictionary<(byte, byte), long> _touches = new();
    private string _session = "";
    private DateTime _startedLocal;
    private long _startedMs;
    private long _nowMs;
    private string _fileStem = "";
    private int _playersMax;
    private string _note = "";

    public BotScorecard(string folder, string track, string policy, float askedGapM)
    {
        _folder = folder;
        _track = track;
        _policy = policy;
        _askedGapM = askedGapM;
    }

    public string Folder => Path.GetFullPath(_folder);

    // A new session: the last one's card is final, this one starts empty.
    public void Start(string session, long nowMs)
    {
        lock (_lock)
        {
            _session = session;
            _startedLocal = DateTime.Now;
            _startedMs = nowMs;
            _nowMs = nowMs;
            _bots.Clear();
            _touches.Clear();
            _playersMax = 0;
            string safe = new string($"{_track}_{session}".Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_').ToArray());
            _fileStem = $"{_startedLocal:yyyyMMdd_HHmmss}_{safe}";
        }
    }

    // Something that stops the whole card meaning what it says (the plugin stopped after errors).
    public void Note(string note)
    {
        lock (_lock)
        {
            _note = note;
        }
    }

    private Bot For(byte sessionId)
    {
        if (!_bots.TryGetValue(sessionId, out Bot? bot))
        {
            bot = new Bot();
            _bots[sessionId] = bot;
        }
        return bot;
    }

    public void Tick(long nowMs, int players)
    {
        lock (_lock)
        {
            _nowMs = nowMs;
            _playersMax = Math.Max(_playersMax, players);
        }
    }

    // A car the server has out but the plugin has not taken over yet (the stock AI is moving it off).
    public void Waiting(byte sessionId, float dt)
    {
        lock (_lock)
        {
            Bot bot = For(sessionId);
            bot.Seconds += dt;
            bot.WaitingSeconds += dt;
        }
    }

    public void TakenOver(byte sessionId)
    {
        lock (_lock)
        {
            For(sessionId).Takeovers++;
        }
    }

    // A train follower replaying its lead (TrainSize): counted, not graded.
    public void Following(byte sessionId, float dt)
    {
        lock (_lock)
        {
            Bot bot = For(sessionId);
            bot.Seconds += dt;
            bot.FollowingSeconds += dt;
        }
    }

    public void Stuck(byte sessionId)
    {
        lock (_lock)
        {
            For(sessionId).Stuck++;
        }
    }

    // One tick of a car the plugin has. chaseGapM: NaN unless it is chasing a car ahead.
    public void Driven(byte sessionId, float dt, bool heldByStock, bool policyValid, float speedMs, float slipRad,
        float wallPushOutM, bool wrongWay, float chaseGapM, bool chaseLeadIsBot)
    {
        lock (_lock)
        {
            Bot bot = For(sessionId);
            bot.Seconds += dt;
            if (heldByStock)
            {
                bot.HeldSeconds += dt;
                return;
            }
            if (!policyValid)
            {
                bot.InvalidSeconds += dt;
            }
            bot.DistanceM += speedMs * dt;
            if (wallPushOutM > 0f)
            {
                if (_nowMs - bot.LastWallMs > TouchGapMs)
                {
                    bot.WallHits++;
                }
                bot.LastWallMs = _nowMs;
            }
            if (wrongWay && !bot.WrongWay)
            {
                bot.WrongWays++;
            }
            bot.WrongWay = wrongWay;
            if (speedMs > MovingMs)
            {
                bot.MovingSeconds += dt;
                float slip = MathF.Abs(slipRad);
                if (slip >= SlideHighRad)
                {
                    bot.SpinSeconds += dt;
                }
                else if (slip >= SlideLowRad)
                {
                    bot.SlideSeconds += dt;
                    bot.SlideSlipSum += slip * 180f / MathF.PI * dt;
                }
                if (wallPushOutM > 0f)
                {
                    bot.WallSeconds += dt;
                }
            }
            if (!float.IsNaN(chaseGapM))
            {
                bot.ChaseSeconds += dt;
                if (chaseLeadIsBot)
                {
                    bot.ChaseBotSeconds += dt;
                }
                if (MathF.Abs(chaseGapM - _askedGapM) <= InBandM)
                {
                    bot.InBandSeconds += dt;
                }
                bot.Gaps[Math.Clamp((int)(MathF.Max(chaseGapM, 0f) / GapBinM), 0, GapBins - 1)] += dt;
            }
        }
    }

    // Two cars' outlines overlap this tick (other: a bot's or a player's session id).
    public void Touching(byte sessionId, byte other, bool otherIsBot)
    {
        lock (_lock)
        {
            var key = otherIsBot && other < sessionId ? (other, sessionId) : (sessionId, other);
            bool fresh = !_touches.TryGetValue(key, out long last) || _nowMs - last > TouchGapMs;
            _touches[key] = _nowMs;
            if (!fresh)
            {
                return;
            }
            if (otherIsBot)
            {
                For(sessionId).BotHits++;
                For(other).BotHits++;
            }
            else
            {
                For(sessionId).PlayerHits++;
            }
        }
    }

    private static double Share(double part, double whole) => whole > 0 ? part / whole : 0;

    private static double? Percentile(double[] bins, double total, double p)
    {
        if (total <= 0)
        {
            return null;
        }
        double want = p * total, sum = 0;
        for (int i = 0; i < bins.Length; i++)
        {
            sum += bins[i];
            if (sum >= want)
            {
                return (i + 0.5) * GapBinM;
            }
        }
        return (bins.Length - 0.5) * GapBinM;
    }

    private Dictionary<string, object?> Numbers(string name, Bot b)
    {
        double hours = (b.Seconds - b.WaitingSeconds - b.HeldSeconds - b.FollowingSeconds) / 3600.0;
        double stock = b.WaitingSeconds + b.HeldSeconds + b.InvalidSeconds;
        return new Dictionary<string, object?>
        {
            ["bot"] = name,
            ["minutes"] = Math.Round(b.Seconds / 60, 2),
            ["driven_minutes"] = Math.Round(hours * 60, 2),
            ["following_minutes"] = Math.Round(b.FollowingSeconds / 60, 2),
            ["km"] = Math.Round(b.DistanceM / 1000, 2),
            ["kmh"] = Math.Round(b.MovingSeconds > 0 ? b.DistanceM / b.MovingSeconds * 3.6 : 0, 1),
            ["slide"] = Math.Round(Share(b.SlideSeconds, b.MovingSeconds), 3),
            ["slide_slip_deg"] = Math.Round(b.SlideSeconds > 0 ? b.SlideSlipSum / b.SlideSeconds : 0, 1),
            ["spin"] = Math.Round(Share(b.SpinSeconds, b.MovingSeconds), 3),
            ["walls"] = Math.Round(Share(b.WallSeconds, b.MovingSeconds), 3),
            ["wall_hits"] = b.WallHits,
            ["wall_hits_per_hour"] = Math.Round(hours > 0 ? b.WallHits / hours : 0, 1),
            ["bot_hits"] = b.BotHits,
            ["player_hits"] = b.PlayerHits,
            ["car_hits_per_hour"] = Math.Round(hours > 0 ? (b.BotHits + b.PlayerHits) / hours : 0, 1),
            ["chase_minutes"] = Math.Round(b.ChaseSeconds / 60, 2),
            ["chase_bot_minutes"] = Math.Round(b.ChaseBotSeconds / 60, 2),
            ["gap_p10"] = Percentile(b.Gaps, b.ChaseSeconds, 0.1) is double p10 ? (object?)Math.Round(p10, 2) : null,
            ["gap_p50"] = Percentile(b.Gaps, b.ChaseSeconds, 0.5) is double p50 ? (object?)Math.Round(p50, 2) : null,
            ["gap_p90"] = Percentile(b.Gaps, b.ChaseSeconds, 0.9) is double p90 ? (object?)Math.Round(p90, 2) : null,
            ["in_band"] = Math.Round(Share(b.InBandSeconds, b.ChaseSeconds), 3),
            ["stock"] = Math.Round(Share(stock, b.Seconds - b.FollowingSeconds), 3),
            ["stock_waiting_s"] = Math.Round(b.WaitingSeconds, 1),
            ["stock_held_s"] = Math.Round(b.HeldSeconds, 1),
            ["bad_output_s"] = Math.Round(b.InvalidSeconds, 1),
            ["stuck"] = b.Stuck,
            ["wrong_way"] = b.WrongWays,
            ["takeovers"] = b.Takeovers,
        };
    }

    private Bot Pooled()
    {
        var all = new Bot();
        foreach (Bot b in _bots.Values)
        {
            all.Seconds += b.Seconds; all.WaitingSeconds += b.WaitingSeconds; all.HeldSeconds += b.HeldSeconds;
            all.InvalidSeconds += b.InvalidSeconds; all.FollowingSeconds += b.FollowingSeconds;
            all.MovingSeconds += b.MovingSeconds; all.SlideSeconds += b.SlideSeconds; all.SpinSeconds += b.SpinSeconds;
            all.SlideSlipSum += b.SlideSlipSum; all.WallSeconds += b.WallSeconds; all.DistanceM += b.DistanceM;
            all.WallHits += b.WallHits; all.PlayerHits += b.PlayerHits; all.WrongWays += b.WrongWays;
            all.Takeovers += b.Takeovers; all.Stuck += b.Stuck;
            all.ChaseSeconds += b.ChaseSeconds; all.ChaseBotSeconds += b.ChaseBotSeconds; all.InBandSeconds += b.InBandSeconds;
            for (int i = 0; i < GapBins; i++)
            {
                all.Gaps[i] += b.Gaps[i];
            }
        }
        // a touch between two bots is one hit for the pool, not one for each
        all.BotHits = _bots.Values.Sum(b => b.BotHits) / 2;
        return all;
    }

    // Writes both files now (final: the session is over). Errors are logged, never thrown.
    public void Write(bool final)
    {
        string text, json, stem;
        lock (_lock)
        {
            if (_fileStem.Length == 0)
            {
                return;
            }
            stem = _fileStem;
            double minutes = Math.Max(0, _nowMs - _startedMs) / 60000.0;
            var all = Numbers("all", Pooled());
            var bots = _bots.Select(kv => Numbers(kv.Key.ToString(CultureInfo.InvariantCulture), kv.Value)).ToList();
            var card = new Dictionary<string, object?>
            {
                ["track"] = _track,
                ["session"] = _session,
                ["started"] = _startedLocal.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                ["minutes"] = Math.Round(minutes, 1),
                ["final"] = final,
                ["policy"] = _policy,
                ["asked_gap_m"] = _askedGapM,
                ["players_max"] = _playersMax,
                ["note"] = _note,
                ["all"] = all,
                ["bots"] = bots,
            };
            json = JsonSerializer.Serialize(card, new JsonSerializerOptions { WriteIndented = true });
            text = Format(card, all, bots);
        }
        lock (_fileLock)
        try
        {
            Directory.CreateDirectory(_folder);
            string path = Path.Combine(_folder, stem);
            File.WriteAllText(path + ".json.tmp", json);
            File.Move(path + ".json.tmp", path + ".json", overwrite: true);
            File.WriteAllText(path + ".txt.tmp", text);
            File.Move(path + ".txt.tmp", path + ".txt", overwrite: true);
            if (final)
            {
                VerboseLog.Information("GradientDrift: scorecard written to {Path}.txt", Path.GetFullPath(path));
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Log.Warning("GradientDrift: could not write the scorecard ({Message})", e.Message);
        }
    }

    private static string Format(Dictionary<string, object?> card, Dictionary<string, object?> all, List<Dictionary<string, object?>> bots)
    {
        static string Pct(object? v) => Convert.ToDouble(v, CultureInfo.InvariantCulture).ToString("P0", CultureInfo.InvariantCulture).Replace(" ", "");
        static string F(object? v, string f) => v == null ? "-" : Convert.ToDouble(v, CultureInfo.InvariantCulture).ToString(f, CultureInfo.InvariantCulture);
        var sb = new StringBuilder();
        sb.AppendLine("Gradient Drift scorecard");
        sb.AppendLine($"track    {card["track"]}");
        sb.AppendLine($"session  {card["session"]}, started {card["started"]}, {F(card["minutes"], "F0")} min{((bool)card["final"]! ? "" : " so far")}");
        sb.AppendLine($"policy   {card["policy"]}");
        sb.AppendLine($"players  up to {card["players_max"]} on at once; asked chase gap {F(card["asked_gap_m"], "F1")} m");
        if ((string?)card["note"] != "")
        {
            sb.AppendLine($"NOTE     {card["note"]}");
        }
        sb.AppendLine();
        sb.AppendLine("Moving = over 25 km/h. slide: slip 20-80 deg, spin: 80+ (shares of moving time). walls: share of moving");
        sb.AppendLine("time pushed out of a wall. hits: separate touches; car hits = bots/players. gap: chase gap p10/p50/p90 m,");
        sb.AppendLine("band: chase time within 2 m of the asked gap. stock: share of time the server's AI drove the bot.");
        sb.AppendLine();
        string head = $"{"bot",-4} {"driven",7} {"km",6} {"km/h",5} {"slide",6} {"angle",5} {"spin",5} {"walls",6} {"wall hits",9} {"car hits",9} " +
                      $"{"chase",6} {"gap",15} {"band",5} {"stock",6} {"stuck",5} {"wrong way",9}";
        sb.AppendLine(head);
        foreach (var r in bots.Append(all))
        {
            sb.AppendLine($"{r["bot"],-4} {F(r["driven_minutes"], "F0") + "m",7} {F(r["km"], "F1"),6} {F(r["kmh"], "F0"),5} {Pct(r["slide"]),6} " +
                          $"{F(r["slide_slip_deg"], "F0"),5} {Pct(r["spin"]),5} {Pct(r["walls"]),6} {r["wall_hits"] + " " + F(r["wall_hits_per_hour"], "F0") + "/h",9} " +
                          $"{r["bot_hits"] + "/" + r["player_hits"],9} {F(r["chase_minutes"], "F0") + "m",6} " +
                          $"{F(r["gap_p10"], "F1") + "/" + F(r["gap_p50"], "F1") + "/" + F(r["gap_p90"], "F1"),15} {Pct(r["in_band"]),5} " +
                          $"{Pct(r["stock"]),6} {r["stuck"],5} {r["wrong_way"],9}");
        }
        return sb.ToString();
    }
}
