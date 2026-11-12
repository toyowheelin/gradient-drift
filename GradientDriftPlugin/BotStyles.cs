using System;
using System.Collections.Generic;

namespace GradientDriftPlugin;

// One bot's driving style: how pushy it is, how close it follows, how good it is. Every number is
// one the policy already reads (its pace, the asked chase gap, its skill) or a plugin-side chase
// rule, so a style needs no retraining; each is held inside the range the policy was trained over.
//
//   PaceMiddle      the pace HoldBotSpacing eases this bot around (the server's is BotPacePercent)
//   ChaseGapMeters  the gap it is asked to hold when chasing (the chase_gap input and the IDM's s0)
//   Skill           BotSkill for this bot
//   OptInMeters     how far ahead a car may be for this bot to take it up (ChaseOptInMeters)
//   ChaseTopShare   ChasePaceMaxPercent / 100 for this bot
//   TimeGapShare    its share of ChaseTimeGapSeconds (the speed-dependent part of the gap it holds)
//   Aggression      what the pace, the opt-in, the chase top and the time gap came from (0 - 1;
//                   0.5 = the settings)
//   Dials           the style dials, for a policy trained to follow them (StyleDials)
internal readonly record struct BotStyle(float Aggression, float PaceMiddle, float ChaseGapMeters, float Skill,
    float OptInMeters, float ChaseTopShare, float TimeGapShare = 1f, StyleDials Dials = default);

// The style dials (drift_env STYLE_*), each -1 .. 1, 0 = the policy's own style: inputs a policy
// trained with "style_range" reads, picked from the habits that tell players apart.
//   Angle   + a deeper drift angle overall, - a shallower one
//   Swing   + the angle runs deeper at the apex and shallower between corners (as most players
//           drive), - one angle throughout
//   Commit  + carries more speed into corners, - settles the car early
//   Smooth  + calmer, smoother steering, - busier hands
public readonly record struct StyleDials(float Angle, float Swing, float Commit, float Smooth);

// One entry of the BotStyles setting. A missing value is drawn (BotStyleSpread) or the server's.
public class BotStyleEntry
{
    // The bot's entry-list DRIVERNAME. Empty: the entry applies to the AI entries in order
    // (the first entry without a name to the first AI entry, and so on).
    public string? Name { get; set; }
    // 0 - 1: 0 calm (slowest pace spacing allows, takes up a car only close ahead, chases no
    // faster than its own top pace, twice the time gap), 0.5 the server's settings, 1 pushy (top
    // pace, takes up a car from far back, chases at up to 10% over ChasePaceMaxPercent, no time gap:
    // it holds just its chase gap at any speed, door to door).
    public float? Aggression { get; set; }
    // The gap this bot is asked to hold when chasing, metres bumper to bumper (0.5 - 6 for the
    // policies so far; outside that the nearest end is used).
    public float? ChaseGapMeters { get; set; }
    // BotSkill for this bot, 0 - 1.
    public float? Skill { get; set; }
    // The style dials for this bot, -1 .. 1 each (StyleDials; BotStyleAngle ... for the defaults).
    public float? Angle { get; set; }
    public float? Swing { get; set; }
    public float? Commit { get; set; }
    public float? Smooth { get; set; }
}

internal static class BotStyles
{
    // Aggression from 0 to 1 moves the opt-in distance from OptInLow to OptInHigh times the setting
    // and the chase top pace by ChaseTopSwing either way.
    private const float OptInLow = 0.4f, OptInHigh = 1.6f, ChaseTopSwing = 0.10f;
    // ... and the time gap (ChaseTimeGapSeconds) from CalmTimeGapShare times the setting at 0 to none at 1:
    // the time gap is what holds a chaser ~7 m back at drifting speed, so the pushiest bots drop it.
    private const float CalmTimeGapShare = 2f;
    // BotStyleSpread 1: the chase gap is drawn from the setting / GapSpreadFactor to x GapSpreadFactor.
    private const float GapSpreadFactor = 2f;

    // The style of the AI entry at aiEntry (its place among the AI entries) with session id
    // sessionId and entry-list name name. paceLow / paceBase / paceHigh: the pace spacing may ask
    // for (BotPaceMinPercent), the server's (BotPacePercent) and the policy's top, in the policy's
    // units (0 for all three: a policy without a pace). gapLow / gapHigh, skillLow / skillHigh:
    // the ranges the policy was trained over.
    public static BotStyle For(GradientDriftConfiguration cfg, int aiEntry, byte sessionId, string? name,
        float paceLow, float paceBase, float paceHigh, float gapLow, float gapHigh, float skillLow, float skillHigh,
        float styleLimit = 0f)
    {
        BotStyleEntry? entry = Find(cfg.BotStyles, aiEntry, name);

        // a draw of its own for every bot, the same at every spawn and every server start
        // (BotStyleSeed changes them all)
        var random = new Random(unchecked(cfg.BotStyleSeed * 7919 + sessionId * 104729 + 17));
        float drawAggression = (float)random.NextDouble() - 0.5f;
        float drawGap = 2f * (float)random.NextDouble() - 1f;
        float drawSkill = (float)random.NextDouble();
        // drawn after the others, so the older draws stay the same
        float drawAngle = 2f * (float)random.NextDouble() - 1f;
        float drawSwing = 2f * (float)random.NextDouble() - 1f;
        float drawCommit = 2f * (float)random.NextDouble() - 1f;
        float drawSmooth = 2f * (float)random.NextDouble() - 1f;

        float spread = Math.Clamp(cfg.BotStyleSpread, 0f, 1f);
        float aggression = Math.Clamp(entry?.Aggression ?? 0.5f + spread * drawAggression, 0f, 1f);
        float gap = entry?.ChaseGapMeters
            ?? cfg.ChaseGapMeters * MathF.Pow(GapSpreadFactor, spread * drawGap);
        gap = Math.Clamp(gap, gapLow, gapHigh);
        float skill = entry?.Skill ?? cfg.BotSkill - Math.Clamp(cfg.BotSkillSpread, 0f, 1f) * drawSkill;
        skill = Math.Clamp(skill, skillLow, skillHigh);

        // the pace: the server's at 0.5, the floor at 0, the policy's top at 1
        float k = 2f * (aggression - 0.5f);
        float pace = k >= 0f ? paceBase + k * (paceHigh - paceBase) : paceBase + k * (paceBase - paceLow);
        float optIn = cfg.ChaseOptInMeters * (OptInLow + aggression * (OptInHigh - OptInLow));
        float top = cfg.ChasePaceMaxPercent / 100f * (1f + k * ChaseTopSwing);
        float timeGap = k >= 0f ? 1f - k : 1f - k * (CalmTimeGapShare - 1f);
        float dialSpread = Math.Clamp(cfg.BotStyleDialSpread, 0f, 1f);
        float Dial(float? set, float server, float draw) => Math.Clamp(set ?? server + dialSpread * draw, -styleLimit, styleLimit);
        var dials = new StyleDials(Dial(entry?.Angle, cfg.BotStyleAngle, drawAngle), Dial(entry?.Swing, cfg.BotStyleSwing, drawSwing),
            Dial(entry?.Commit, cfg.BotStyleCommit, drawCommit), Dial(entry?.Smooth, cfg.BotStyleSmooth, drawSmooth));
        return new BotStyle(aggression, pace, gap, skill, optIn, top, timeGap, dials);
    }

    private static BotStyleEntry? Find(List<BotStyleEntry>? entries, int aiEntry, string? name)
    {
        if (entries == null || entries.Count == 0)
        {
            return null;
        }
        if (!string.IsNullOrWhiteSpace(name))
        {
            foreach (BotStyleEntry e in entries)
            {
                if (!string.IsNullOrWhiteSpace(e.Name)
                    && string.Equals(e.Name.Trim(), name.Trim(), StringComparison.OrdinalIgnoreCase))
                {
                    return e;
                }
            }
        }
        int k = -1;
        foreach (BotStyleEntry e in entries)
        {
            if (string.IsNullOrWhiteSpace(e.Name) && ++k == aiEntry)
            {
                return e;
            }
        }
        return null;
    }
}
