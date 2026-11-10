using System.Globalization;
using AssettoServer.Commands;
using Qmmands;

namespace GradientDriftPlugin;

// Chat commands for the zone scores (ZoneScoring): /zones says how runs are scored and the
// player's best, /top the server's board.
public class ZoneCommands : ACModuleBase
{
    private readonly ZoneScoring _scoring;

    public ZoneCommands(ZoneScoring scoring)
    {
        _scoring = scoring;
    }

    [Command("zones")]
    public void Zones()
    {
        if (!_scoring.On)
        {
            Reply("No drift zones are scored on this server.");
            return;
        }
        if (_scoring.Arcade)
        {
            Reply(string.Format(CultureInfo.InvariantCulture, "Arcade drift scoring: points every moment you slide ({0:0}°+ at {1:0}+ km/h) from angle x speed, "
                + "times a combo that goes up x{2:0.00} with each transition (to x{3:0}) and up to x{4:0} sliding close behind another car. "
                + "Stop sliding for {5:0.0} s to bank the chain; a spin or wall hit loses it. /top for the board.",
                ArcadeJudge.SlidingDeg, ArcadeJudge.MovingKmh, ArcadeJudge.ComboStep, ArcadeJudge.ComboMax, 1f + ArcadeJudge.TandemMax, ArcadeJudge.ChainGraceS));
        }
        else
        {
            Reply($"Drift zones: {_scoring.ZoneCount} per run, from the first zone to the last. Each zone up to 100: line 40 (rear within 3 m of the edge while sliding 20°+), "
              + $"angle 30 (full at {ZoneJudge.AngleFullDeg:0}°), speed 30, tandem up to +{ZoneJudge.TandemBonus:0}; a spin scores the zone 0, each wall hit -{ZoneJudge.WallPenalty:0}. /top for the board.");
        }
        if (Client != null && _scoring.Records is { Open: true } records)
        {
            (float best, int rank) = records.Personal(Client.Guid.ToString(CultureInfo.InvariantCulture));
            Reply(rank > 0 ? string.Format(CultureInfo.InvariantCulture, "Your best: " + (_scoring.Arcade ? "{0:#,0}" : "{0:0.0}") + " pts, #{1} on this server.", best, rank)
                : _scoring.Arcade ? "No banked drift chain of yours on this server yet." : "No scored run of yours on this server yet.");
        }
    }

    [Command("top")]
    public void Top()
    {
        if (_scoring.Records is not { Open: true } records)
        {
            Reply(_scoring.On ? "Drifting is scored but not kept on this server." : "No drift zones are scored on this server.");
            return;
        }
        if (records.Board.Count == 0)
        {
            Reply(_scoring.Arcade ? "No banked drift chains on this server yet." : "No scored zone runs on this server yet.");
            return;
        }
        foreach (ZoneRecords.Entry e in records.Board.Take(5))
        {
            Reply(string.Format(CultureInfo.InvariantCulture, "#{0} {1} " + (_scoring.Arcade ? "{2:#,0}" : "{2:0.0}") + " pts ({3:0}°, {4})", e.Rank, e.Driver, e.Score, e.AngleDeg, e.Car));
        }
    }
}
