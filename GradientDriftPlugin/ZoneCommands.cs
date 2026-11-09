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
        Reply($"Drift zones: {_scoring.ZoneCount} per run, from the first zone to the last. Each zone up to 100: line 40 (rear within 3 m of the edge while sliding 20°+), "
              + $"angle 30 (full at {ZoneJudge.AngleFullDeg:0}°), speed 30, tandem up to +{ZoneJudge.TandemBonus:0}; a spin scores the zone 0, each wall hit -{ZoneJudge.WallPenalty:0}. /top for the board.");
        if (Client != null && _scoring.Records is { Open: true } records)
        {
            (float best, int rank) = records.Personal(Client.Guid.ToString(CultureInfo.InvariantCulture));
            Reply(rank > 0 ? string.Format(CultureInfo.InvariantCulture, "Your best: {0:0.0} pts, #{1} on this server.", best, rank)
                : "No scored run of yours on this server yet.");
        }
    }

    [Command("top")]
    public void Top()
    {
        if (_scoring.Records is not { Open: true } records)
        {
            Reply(_scoring.On ? "Zone runs are scored but not kept on this server." : "No drift zones are scored on this server.");
            return;
        }
        if (records.Board.Count == 0)
        {
            Reply("No scored zone runs on this server yet.");
            return;
        }
        foreach (ZoneRecords.Entry e in records.Board.Take(5))
        {
            Reply(string.Format(CultureInfo.InvariantCulture, "#{0} {1} {2:0.0} pts ({3:0}°, {4})", e.Rank, e.Driver, e.Score, e.AngleDeg, e.Car));
        }
    }
}
