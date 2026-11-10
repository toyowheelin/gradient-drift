using System.Numerics;

namespace GradientDriftPlugin;

// Judges one car's drifting through the track's zones, the way a drift judge would and the way
// setup's zones.score measures a recorded run, from what the server knows of the car: where it is,
// which way it points and how it moves. Nothing comes from the player's game but its car's status,
// so a modified client cannot send itself a score.
//
// A zone (DriftZones.Course) is judged while the car is on its stretch of the course (its nearest
// point on the course's middle is between the zone's first and last point), moving at least
// MovingKmh, and gets up to 100 points:
//   line  40 x the zone's fill: the share of that time with the rear axle within DepthM of the
//         zone's edge while sliding at least SlidingDeg (zones.score / build_zones.score)
//   angle 30 at AngleFullDeg or more of slip angle on average while sliding, none at SlidingDeg
//   speed 30 at SpeedFullOfPace of the track's pace reference speed (a quick driver's) on average
//         while sliding, none at SpeedNoneOfPace (no pace reference: SpeedFallbackKmh is full)
//   tandem up to +TandemBonus for the share of the time sliding close behind another moving car
//   spin  a slip angle past SpinDeg in the zone: the zone scores 0
// A run starts on entering the first zone after the longest stretch with none and ends on leaving
// the last; it scores the zones' mean, less WallPenalty for each wall hit, if every zone was
// judged on the way. Going the wrong way, jumping (a teleport, back to pits) or stopping ends it
// without a score.
public sealed class ZoneJudge
{
    public const float DepthM = DriftZones.DepthM;
    public const float SlidingDeg = 20f;       // zones.SLIDING_DEG
    public const float MovingKmh = 25f;        // zones.MOVING_KMH: slower samples are not judged at all
    public const float RearAxleBehindM = 1.3f; // build_zones.REAR_AXLE_BEHIND_CENTRE_M
    public const float LinePoints = 40f, AnglePoints = 30f, SpeedPoints = 30f;
    public const float AngleFullDeg = 45f;
    public const float SpeedFullOfPace = 0.95f, SpeedNoneOfPace = 0.6f, SpeedFallbackKmh = 70f;
    public const float TandemBonus = 10f, TandemGapM = 8f;
    public const float SpinDeg = 100f;
    public const float WallPenalty = 5f;
    public const float MinZoneSeconds = 0.25f; // judged time a zone needs to count as driven through
    public const float MinRunScore = 1f;       // a run scoring less is not a drift run (not told, not kept)
    public const float WrongWaySeconds = 1.5f, StoppedSeconds = 8f, JumpM = 40f;
    // The car's status comes from the player's game, so a run whose numbers do not add up is not
    // scored: faster than any car (MaxSpeedMs), or, over each CheckSeconds, moved much further than
    // its reported speed says (a position pushed along) or much less (a speed made up for points).
    // A curve's chord is never under 0.7 of its arc in a second of driving, so honest cars pass.
    public const float MaxSpeedMs = 110f, CheckSeconds = 1f, CheckMinPathM = 12f;
    public const float CheckMovedOverMax = 1.6f, CheckMovedUnderMin = 0.35f, CheckSlackM = 4f;

    public enum Kind { Zone, Run, Ended, Spin, Wall, Combo }     // Combo: arcade scoring's transitions (ArcadeJudge)

    // Something the driver is told: a zone judged (its parts), a run's score, why a run ended
    // without one, a spin, a wall hit.
    public readonly record struct Event(Kind Kind, int Zone, float Points, float Fill, float AngleDeg, float SpeedOfPace,
        float Tandem, string Text);

    public readonly record struct RunResult(float Score, float[] ZonePoints, float MeanAngleDeg, float MeanSpeedOfPace,
        float MeanFill, int Walls);

    private readonly Course _course;
    private readonly ZoneTally[] _tally;
    private readonly bool[] _visiting;
    private readonly float[] _runPoints;
    private readonly ZoneTally[] _runTally;
    private readonly bool[] _runDone;
    private int _point = -1;
    private Vector2 _lastPosition;
    private bool _running;
    private int _runWalls;
    private float _wrongWayS, _stoppedS;
    private readonly CarDataCheck _check = new();

    public List<Event> Events { get; } = new();
    public RunResult? LastRun { get; private set; }

    // What the HUD shows while driving: the zone the car is in (-1 none) and how it is going there,
    // the slip angle and speed right now, and the run so far.
    public int CurrentZone { get; private set; } = -1;
    public float CurrentFill => CurrentZone >= 0 ? _tally[CurrentZone].Fill : 0f;
    public float CurrentPoints => CurrentZone >= 0 ? _tally[CurrentZone].Points(_course.ZoneCount) : 0f;
    public float AngleNowDeg { get; private set; }
    public float SpeedNowKmh { get; private set; }
    public float SpeedNowOfPace { get; private set; }
    public bool Running => _running;
    public int RunZonesDone => _runDone.Count(d => d);
    public float RunPointsSoFar
    {
        get
        {
            int done = 0;
            float sum = 0f;
            for (int z = 0; z < _runDone.Length; z++)
            {
                if (_runDone[z]) { done++; sum += _runPoints[z]; }
            }
            return done == 0 ? 0f : MathF.Max(0f, sum / done - WallPenalty * _runWalls);
        }
    }

    public ZoneJudge(Course course)
    {
        _course = course;
        int n = course.ZoneCount;
        _tally = new ZoneTally[n];
        _runTally = new ZoneTally[n];
        for (int z = 0; z < n; z++) { _tally[z] = new ZoneTally(); _runTally[z] = new ZoneTally(); }
        _visiting = new bool[n];
        _runPoints = new float[n];
        _runDone = new bool[n];
    }

    // One look at the car: its middle (x, z), heading (radians, x toward z), velocity, the seconds
    // since the last look, whether it is close behind another moving car, and wall hits since.
    public void Step(float x, float z, float heading, float vx, float vz, float dt, bool behindCar = false, int wallHits = 0)
    {
        var position = new Vector2(x, z);
        float speed = MathF.Sqrt(vx * vx + vz * vz);
        SpeedNowKmh = speed * 3.6f;
        float slip = 0f;
        if (speed > 0.5f)
        {
            float cos = (MathF.Cos(heading) * vx + MathF.Sin(heading) * vz) / speed;
            slip = MathF.Acos(Math.Clamp(cos, -1f, 1f)) * 180f / MathF.PI;
        }
        AngleNowDeg = slip;
        Vector2 rear = position - RearAxleBehindM * new Vector2(MathF.Cos(heading), MathF.Sin(heading));
        StepAt(position, rear, slip, new Vector2(vx, vz), dt, behindCar, wallHits);
    }

    // Step, with the rear axle and slip angle (degrees, 0 - 180) given: what setup's zones.score
    // reads from a recording, for tests/plugin to compare the two.
    public void StepAt(Vector2 position, Vector2 rear, float slipDeg, Vector2 velocity, float dt, bool behindCar, int wallHits)
    {
        float speedMs = velocity.Length();
        AngleNowDeg = slipDeg;
        SpeedNowKmh = speedMs * 3.6f;
        bool jumped = _point >= 0 && Vector2.Distance(position, _lastPosition) > JumpM;
        _lastPosition = position;
        _point = _course.Nearest(position, jumped ? -1 : _point);
        int p = _point;
        SpeedNowOfPace = SpeedNowKmh / _course.PaceKmh(p);
        if (jumped)
        {
            End("left the course");
            _check.Reset();
        }
        if (!jumped && !_check.Ok(position, speedMs, dt))
        {
            End("car data does not add up");
        }
        if (wallHits > 0 && _running)
        {
            _runWalls += wallHits;
            Events.Add(new Event(Kind.Wall, CurrentZone, -WallPenalty * wallHits, 0, 0, 0, 0, "wall"));
        }

        // going the wrong way round (the way it moves, not where it points: a drift or a spin
        // points it all over), or stopped
        _wrongWayS = speedMs > 3f && Vector2.Dot(_course.Forward(p), velocity) < 0f ? _wrongWayS + dt : 0f;
        _stoppedS = SpeedNowKmh < MovingKmh ? _stoppedS + dt : 0f;
        if (_wrongWayS > WrongWaySeconds) { End("wrong way"); _wrongWayS = 0f; }
        if (_stoppedS > StoppedSeconds) { End("stopped"); _stoppedS = 0f; }

        CurrentZone = -1;
        for (int z = 0; z < _course.ZoneCount; z++)
        {
            bool inside = _course.InZone(z, p);
            if (inside && !_visiting[z])
            {
                _visiting[z] = true;
                _tally[z].Reset();
                // a car that turns up part way along a zone (back from the pits, a jump) is not judged there
                _tally[z].Late = !_course.NearZoneStart(z, p);
                if (z == _course.FirstZone && !_tally[z].Late)
                {
                    StartRun();
                }
            }
            else if (!inside && _visiting[z])
            {
                _visiting[z] = false;
                FinishZone(z);
            }
            if (!inside || _tally[z].Late)
            {
                continue;
            }
            CurrentZone = CurrentZone < 0 ? z : CurrentZone;
            if (SpeedNowKmh < MovingKmh)
            {
                continue;       // zones.score leaves slow rows out
            }
            ZoneTally t = _tally[z];
            t.OnS += dt;
            if (slipDeg > SpinDeg) t.Spun = true;
            if (slipDeg < SlidingDeg)
            {
                continue;
            }
            float depth = _course.Depth(z, p, rear);
            if (depth >= 0f && depth <= DepthM) t.FilledS += dt;
            t.SlideS += dt;
            t.AngleSum += slipDeg * dt;
            t.SpeedSum += SpeedNowOfPace * dt;
            if (behindCar) t.TandemS += dt;
        }
    }

    private void StartRun()
    {
        _running = true;
        _runWalls = 0;
        Array.Clear(_runDone);
        Array.Clear(_runPoints);
    }

    private void FinishZone(int z)
    {
        ZoneTally t = _tally[z];
        if (t.Late || t.OnS < MinZoneSeconds)
        {
            return;
        }
        float points = t.Points(_course.ZoneCount);
        Events.Add(t.Spun
            ? new Event(Kind.Spin, z, 0f, t.Fill, t.MeanAngle, t.MeanSpeed, t.TandemShare, "spin")
            : new Event(Kind.Zone, z, points, t.Fill, t.MeanAngle, t.MeanSpeed, t.TandemShare, ""));
        if (!_running)
        {
            return;
        }
        _runDone[z] = true;
        _runPoints[z] = points;
        _runTally[z].CopyFrom(t);
        if (z != _course.LastZone)
        {
            return;
        }
        int missed = Array.IndexOf(_runDone, false);
        if (missed >= 0)
        {
            End($"missed zone {_course.Number(missed)}");
            return;
        }
        float score = MathF.Max(0f, _runPoints.Average() - WallPenalty * _runWalls);
        if (score < MinRunScore)
        {
            End("too little drifting to score");     // a lap driven, not drifted: not kept
            return;
        }
        float slide = _runTally.Sum(r => r.SlideS);
        var result = new RunResult(score, (float[])_runPoints.Clone(),
            slide > 0 ? _runTally.Sum(r => r.AngleSum) / slide : 0f,
            slide > 0 ? _runTally.Sum(r => r.SpeedSum) / slide : 0f,
            _runTally.Average(r => r.Fill), _runWalls);
        LastRun = result;
        Events.Add(new Event(Kind.Run, z, score, result.MeanFill, result.MeanAngleDeg, result.MeanSpeedOfPace, 0f, ""));
        _running = false;
    }

    // The run so far ends without a score (said why, if one was going), and so do the zones the car is in.
    public void End(string why)
    {
        if (_running && RunZonesDone > 0)
        {
            Events.Add(new Event(Kind.Ended, CurrentZone, 0f, 0f, 0f, 0f, 0f, why));
        }
        _running = false;
        Array.Clear(_visiting);
        CurrentZone = -1;
    }

    // One zone's judged time and what was done in it.
    private sealed class ZoneTally
    {
        public float OnS, FilledS, SlideS, AngleSum, SpeedSum, TandemS;
        public bool Spun, Late;

        public void Reset()
        {
            OnS = FilledS = SlideS = AngleSum = SpeedSum = TandemS = 0f;
            Spun = Late = false;
        }

        public void CopyFrom(ZoneTally t)
        {
            (OnS, FilledS, SlideS, AngleSum, SpeedSum, TandemS, Spun) = (t.OnS, t.FilledS, t.SlideS, t.AngleSum, t.SpeedSum, t.TandemS, t.Spun);
        }

        public float Fill => OnS > 0f ? FilledS / OnS : 0f;
        public float MeanAngle => SlideS > 0f ? AngleSum / SlideS : 0f;
        public float MeanSpeed => SlideS > 0f ? SpeedSum / SlideS : 0f;
        public float TandemShare => OnS > 0f ? TandemS / OnS : 0f;

        public float Points(int zoneCount)
        {
            if (Spun) return 0f;
            float angle = Math.Clamp((MeanAngle - SlidingDeg) / (AngleFullDeg - SlidingDeg), 0f, 1f);
            float speed = Math.Clamp((MeanSpeed - SpeedNoneOfPace) / (SpeedFullOfPace - SpeedNoneOfPace), 0f, 1f);
            // angle and speed only count for the share of the zone that was filled: a big angle
            // in the middle of the road is not what a zone asks for
            return LinePoints * Fill + (AnglePoints * angle + SpeedPoints * speed) * Fill + TandemBonus * TandemShare;
        }
    }

    // The course as the judge reads it: its middle line, each zone's edge, and the order a run
    // meets the zones in.
    public sealed class Course
    {
        private readonly Vector2[] _mid, _positive, _negative;
        private readonly DriftZones.Zone[] _zones;
        private readonly float[]? _paceKmh;
        private readonly int[] _order;     // zone indices in the order a run meets them

        public int PointCount => _mid.Length;
        public int ZoneCount => _zones.Length;
        public int FirstZone => _order[0];
        public int LastZone => _order[^1];

        public Course(DriftZones.Course course, float[]? paceKmh)
        {
            int n = course.Positive.Length;
            _positive = course.Positive.Select(v => new Vector2(v.X, v.Z)).ToArray();
            _negative = course.Negative.Select(v => new Vector2(v.X, v.Z)).ToArray();
            _mid = new Vector2[n];
            for (int i = 0; i < n; i++) _mid[i] = 0.5f * (_positive[i] + _negative[i]);
            _zones = course.Zones.ToArray();
            _paceKmh = paceKmh != null && paceKmh.Length == n ? paceKmh : null;
            _order = RunOrder(_zones, n);
        }

        // The zone number a driver sees (1 = the first a run meets).
        public int Number(int zone) => Array.IndexOf(_order, zone) + 1;

        public float PaceKmh(int point) => _paceKmh?[point] ?? SpeedFallbackKmh / SpeedFullOfPace;

        public bool InZone(int zone, int point)
        {
            DriftZones.Zone z = _zones[zone];
            int n = _mid.Length;
            return Mod(point - z.First, n) <= Mod(z.Last - z.First, n);
        }

        // Whether a point is in the first part of a zone (a third of it, at least 10 points).
        public bool NearZoneStart(int zone, int point)
        {
            DriftZones.Zone z = _zones[zone];
            int n = _mid.Length;
            int length = Mod(z.Last - z.First, n) + 1;
            return Mod(point - z.First, n) <= Math.Max(10, length / 3);
        }

        // How far the rear axle is in from the zone's edge at this point (negative = past the edge).
        public float Depth(int zone, int point, Vector2 rear)
        {
            DriftZones.Zone z = _zones[zone];
            Vector2 edge = z.Side > 0 ? _positive[point] : _negative[point];
            Vector2 other = z.Side > 0 ? _negative[point] : _positive[point];
            Vector2 inward = other - edge;
            float width = inward.Length();
            if (width < 1e-6f) return float.NaN;
            return Vector2.Dot(rear - edge, inward / width);
        }

        public Vector2 Forward(int point)
        {
            Vector2 d = _mid[(point + 1) % _mid.Length] - _mid[point];
            float l = d.Length();
            return l > 1e-6f ? d / l : Vector2.UnitX;
        }

        // The middle-line point nearest to p (zones.score: cKDTree on course.mid), looked for near
        // the last one first; a full search when there is none or it is far off.
        public int Nearest(Vector2 p, int near)
        {
            int n = _mid.Length;
            if (near >= 0)
            {
                int best = -1;
                float bestD = float.MaxValue;
                for (int k = -60; k <= 60; k++)
                {
                    int i = Mod(near + k, n);
                    float d = Vector2.DistanceSquared(p, _mid[i]);
                    if (d < bestD) { bestD = d; best = i; }
                }
                // a local best that is no closer than its neighbours further out, or far off: look everywhere
                if (bestD < 15f * 15f && best != Mod(near - 60, n) && best != Mod(near + 60, n)) return best;
            }
            int all = 0;
            float allD = float.MaxValue;
            for (int i = 0; i < n; i++)
            {
                float d = Vector2.DistanceSquared(p, _mid[i]);
                if (d < allD) { allD = d; all = i; }
            }
            return all;
        }

        // Zones by where they start, beginning after the longest stretch of course with no zone.
        private static int[] RunOrder(DriftZones.Zone[] zones, int n)
        {
            if (zones.Length == 0) return Array.Empty<int>();
            int[] byStart = Enumerable.Range(0, zones.Length).OrderBy(z => zones[z].First).ToArray();
            int start = 0;
            int longest = -1;
            for (int k = 0; k < byStart.Length; k++)
            {
                DriftZones.Zone prev = zones[byStart[(k + byStart.Length - 1) % byStart.Length]];
                int gap = Mod(zones[byStart[k]].First - prev.Last, n);
                if (byStart.Length == 1) gap = n - Mod(prev.Last - prev.First, n);
                if (gap > longest) { longest = gap; start = k; }
            }
            return Enumerable.Range(0, byStart.Length).Select(k => byStart[(start + k) % byStart.Length]).ToArray();
        }
    }

    private static int Mod(int a, int n) => ((a % n) + n) % n;
}

// Whether what a player's game says of its car hangs together (see ZoneJudge.MaxSpeedMs): not faster
// than any car, and over each CheckSeconds moved about as far as its reported speed says.
public sealed class CarDataCheck
{
    private Vector2 _from;
    private float _s = -1f, _pathM;

    public void Reset() => _s = -1f;

    public bool Ok(Vector2 position, float speedMs, float dt)
    {
        if (speedMs > ZoneJudge.MaxSpeedMs)
        {
            _s = -1f;
            return false;
        }
        if (_s < 0f)
        {
            (_from, _s, _pathM) = (position, 0f, 0f);
            return true;
        }
        _s += dt;
        _pathM += speedMs * dt;
        if (_s < ZoneJudge.CheckSeconds)
        {
            return true;
        }
        float moved = Vector2.Distance(position, _from);
        float path = _pathM;
        (_from, _s, _pathM) = (position, 0f, 0f);
        if (path < ZoneJudge.CheckMinPathM && moved < ZoneJudge.CheckMinPathM)
        {
            return true;
        }
        return moved <= ZoneJudge.CheckMovedOverMax * path + ZoneJudge.CheckSlackM
               && moved >= ZoneJudge.CheckMovedUnderMin * path - ZoneJudge.CheckSlackM;
    }
}
