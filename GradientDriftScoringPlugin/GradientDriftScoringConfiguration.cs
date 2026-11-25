namespace GradientDriftScoring;

// Settings, read from the "--- !GradientDriftScoringConfiguration" block of extra_cfg.yml. Every value
// has a working default, and the block can be left out altogether: the plugin then runs with these
// defaults. The plugin needs no bots, no policy and no GradientDriftPlugin beside it.
public class GradientDriftScoringConfiguration
{
    // Score players' drifting (ZoneScoring.cs). Players with CSP see it on a HUD (ZoneHud); everyone
    // gets a chat line per run. Each server keeps its own runs in ZoneRecordsFile (SQLite, in the
    // server's folder; empty = keep nothing) and its board shows each driver's best, ZoneBoardSize of
    // them.
    //
    // ZoneScoreMode picks how drifting is scored:
    // - Zones (ZoneJudge.cs, like a drift series' judges): line, angle and speed in each of the
    //   track's drift zones, a tandem bonus, spins and wall hits marked down, one score for a run from
    //   the first zone to the last. Needs the track's <track>_course_zones.csv and
    //   <track>_course_edges.csv (setup writes them) and the server's AI line (EnableAi: true and the
    //   track's ai/fast_lane.ai; no AI cars are needed).
    // - Arcade (ArcadeJudge.cs: freestyle, no zones and no track files needed, so it works on any
    //   track, with AI off too): points for every moment sliding, from the angle and speed, times a
    //   combo that grows with each transition and a bonus for sliding close behind another car; a spin
    //   or wall hit loses the chain.
    // Each mode keeps its own board.
    public bool ZoneScoring { get; set; } = true;
    public string ZoneScoreMode { get; set; } = "Zones";
    public bool IsArcadeScoring() => string.Equals(ZoneScoreMode?.Trim(), "Arcade", StringComparison.OrdinalIgnoreCase);
    public bool ZoneHud { get; set; } = true;
    public bool ZoneScoreChat { get; set; } = true;
    public string ZoneRecordsFile { get; set; } = "gradientdrift_zone_records.db";
    public int ZoneBoardSize { get; set; } = 10;

    // Paint the track's drift zones on the road for players with CSP (DriftZones.cs): the
    // <track>_course_zones.csv setup writes (and its zone editor changes), as boxes of this colour
    // (RRGGBB) with lines this wide, closed at both ends, striped every ZoneStripeSpacingMeters
    // (twice that for a touch-and-go area). Needs what Zones mode needs. Not painted when
    // ZoneScoreMode is Arcade (the zones are not scored then). Turn it off on a track that has its
    // zones painted already.
    public bool ShowZones { get; set; } = true;
    public string ZoneColor { get; set; } = "F5C400";
    public float ZoneLineWidthMeters { get; set; } = 0.3f;
    public float ZoneStripeSpacingMeters { get; set; } = 1.6f;

    // The plugin's routine lines in the server log (what it loaded) as well as its warnings.
    public bool VerboseLogging { get; set; } = false;
}
