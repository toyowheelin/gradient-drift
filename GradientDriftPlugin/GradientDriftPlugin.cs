using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Text.RegularExpressions;
using AssettoServer.Server;
using AssettoServer.Server.Ai;
using AssettoServer.Server.Ai.Splines;
using AssettoServer.Server.Configuration;
using AssettoServer.Server.Plugin;
using AssettoServer.Shared.Network.Packets.Outgoing;
using Microsoft.Extensions.Hosting;
using Serilog;

namespace GradientDriftPlugin;

// Makes AssettoServer's AI cars drift.
//
// The server's own AI moves each car along the racing line on rails. This plugin lets that run
// (it still decides when a car exists and where it spawns) and then, every server tick,
// replaces the car's position, velocity and rotation with those of its own vehicle model: a
// single-track car with real yaw and sideways motion, steered and throttled by a small neural
// network (DriftPolicy) that was trained offline in a simulator of this same model.
//
// One tick for one car, in the order UpdateCar does it:
//   1. find the car on the racing line (nearest point, the line's direction, distance from it)
//   2. build the policy's 46 inputs from last tick's state and evaluate it
//   3. forward speed: ramp toward the policy's target speed, or (EnableRearFrictionCircle)
//      drive it from the policy's pedal
//   4. integrate yaw rate, lateral velocity and heading from the tire forces
//   5. wrong-way recovery, if the car has spun
//   6. move the car; push it back out of any wall
//   7. put it on the ground, set pitch/roll, write position/velocity/rotation for clients
//   8. cast the sensors the policy will read next tick
//
// drift_rl/fast_sim.py is the same tick in Python and is what the policy is trained in; the two
// must be changed together. drift_rl/replay_live.py replays a telemetry file through it one
// tick at a time, which is how a change here is checked.
//
// Without a usable policy the plugin does nothing at all and the stock AI drives.
public class GradientDriftPlugin : BackgroundService, IAssettoServerAutostart
{
    // Wall triangles the ground ray will pass through (barriers over the road) before giving up.
    private const int GroundRayMaxWallLayers = 12;
    private const float BrakeLightHoldSeconds = 0.3f;
    // The brake action (a policy with three outputs) lights the brake lights from this much of it.
    private const float BrakeLightMinBrake = 0.05f;
    // Ground attitude: how far to either side of the car the ground is sampled, and the
    // height difference between two samples beyond which they are not the same surface.
    private const float AttitudeHalfTrackM = 0.8f;
    private const float AttitudeMaxStepM = 0.4f;
    private const int RideHeightSamples = 9;

    // Status bits 1, 2, 4 and 8, which the server has no names for: "this tire is skidding" for
    // the front left, front right, rear left and rear right wheel. Clients draw rubber on the
    // road from them. A human's game sets one while that tire's slip is more than about 4 times
    // the slip of its peak grip (HISTORY.md).
    private const CarStatusFlags FrontTiresSkidding = (CarStatusFlags)0x3;
    private const CarStatusFlags RearTiresSkidding = (CarStatusFlags)0xC;
    private const float SkidSlipOverPeak = 4f;

    // Optional point cloud of the road surface, used for ride height only where the ground ray
    // finds nothing: <track>_road_mesh.csv in the server's working directory
    // (drift_rl/build_road_mesh.py writes one from the track's Road triangles).
    private const string RoadMeshFileSuffix = "_road_mesh.csv";

    // HoldBotSpacing: how hard a bot's pace (a policy that reads it) or, for an older policy, its
    // RearSlidingGrip is pushed toward the faster or slower end of the policy's range for a given
    // imbalance between the gap ahead and the gap behind, and how quickly it gets there (time
    // constant). Measured in the simulator (README.md).
    private const float SpacingGain = 2f;
    private const float SpacingEaseSeconds = 4f;

    // Telemetry SessionId of a row that carries only the human car (no AI car was out that tick).
    private const byte HumanOnlySessionId = 255;

    // The AssettoServer version this plugin is built and tested against: build-plugin.yml builds it
    // against this tag and checks that the two agree, and setup's TESTED_VERSION says the same.
    public const string BuiltForAssettoServer = "0.0.54";

    private readonly EntryCarManager _entryCarManager;
    private readonly Lazy<ACServer> _acServer;
    private readonly GradientDriftConfiguration _configuration;
    private readonly AiSpline _spline;
    private readonly SessionManager _sessionManager;
    private readonly float _tickRateHz;

    private readonly Vector3[] _roadMeshPoints;
    private readonly MeshKdTree? _roadMeshTree;
    private readonly TriangleGrid? _collisionGrid;
    private readonly RaycastFan? _raycastFan;
    private readonly ProximityPins? _proximityPins;
    private readonly ProximityPins? _contactPins;
    private readonly DriftPolicy? _policy;
    private readonly CourseEdges? _courseEdges;
    private readonly float[]? _paceReferenceKmh;
    // RearSlidingGrip as actually used: the setting, held inside what the policy was trained over.
    private readonly float _rearSlidingGrip = 1f;
    // The pace bots are asked to drive (a policy that reads it), in the policy's units:
    // BotPacePercent (or BotPaceKmh for gen4), held inside what the policy was trained over;
    // HoldBotSpacing eases each car's own pace around it. 0 = no pace.
    private readonly float _basePace;
    private readonly float _paceFloor;      // the slowest pace spacing asks for (BotPaceMinPercent)

    private readonly Dictionary<AiState, DriftCarState> _cars = new();
    private readonly PluginTiming _timing;
    private readonly BotHealth _health;
    private readonly List<AiState> _aiStates = new();

    // Distance along the line from point 0 to each point, the points in that order, and the lap
    // length; null when the line is not one closed loop through every point (no spacing then).
    private readonly float[]? _lapArc;
    private readonly int[]? _lapOrder;
    private readonly float _lapLength;
    private readonly AiParams _aiParams;
    private readonly List<(byte SessionId, AiState State)> _driven = new();
    // TrainSize > 1: each follower, the lead it replays and its place behind it (1 = right behind)
    private readonly List<(byte SessionId, AiState State, AiState Lead, int Place)> _followers = new();
    private readonly Dictionary<AiState, TrainHistory> _trainHistories = new();
    // this tick: each AI entry's first state and the entry's place among the AI entries
    private readonly Dictionary<AiState, int> _trainSlots = new();
    // this tick, per train: the delay the follower one place ahead used, or -1 if it is not following
    private long _aheadDelayMs;
    private AiState? _aheadLead;
    private Vector3 _aheadPosition;
    private Vector3 _aheadPositionBefore;     // the car in front's position the follower was kept clear of
    private readonly Dictionary<AiState, (Vector3 Position, long TimeMs)> _followerLast = new();
    private readonly List<(DriftCarState State, float Arc)> _spaced = new();
    private int[] _gapOrder = new int[8];
    private readonly float[] _observation = new float[DriftPolicy.ObservationSize];

    private DriftTelemetry? _telemetry;
    private long _tickCounter;
    private float _muFront, _muRear;     // each axle's peak friction this tick (AxleFriction)
    // The car being driven this tick, and every car seen so far by model: its own numbers for an
    // any-car policy (cars/<model>.json), else the plugin's configured car (_defaultCar).
    private CarSpec _car;
    private CarSpec _defaultCar;
    private readonly Dictionary<string, CarSpec> _carSpecs = new();
    private readonly float _skill = 1f;           // how good a driver the bots are (BotSkill), for a policy that reads it
    private readonly Random _skillRandom = new();
    private int _failedUpdates;
    private readonly BotNames? _botNames;
    private bool _updatesStopped;
    private const int MaxFailedUpdates = 5;

    // Server time of the previous update; -1 until there has been one. The Update event carries
    // no elapsed time, so the tick length is measured here.
    private long _lastUpdateTimeMs = -1;

    public GradientDriftPlugin(EntryCarManager entryCarManager, Lazy<ACServer> acServer,
        GradientDriftConfiguration configuration, ACServerConfiguration serverConfiguration, AiSpline spline,
        SessionManager sessionManager)
    {
        _entryCarManager = entryCarManager;
        _acServer = acServer;
        _configuration = configuration;
        VerboseLog.On = configuration.VerboseLogging;
        WarnIfOtherAssettoServer();
        _timing = new PluginTiming(configuration.LogTimingIntervalSeconds > 0f || !configuration.VerboseLogging
            ? configuration.LogTimingIntervalSeconds : 60f);
        _tickRateHz = serverConfiguration.Server.RefreshRateHz;
        _spline = spline;
        _botNames = configuration.ShowEntryListNames
            ? new BotNames(entryCarManager, serverConfiguration, configuration.BotNation)
            : null;
        _health = new BotHealth(s => _cars.TryGetValue(s, out DriftCarState? c) && !c.HeldByStock,
            p => _spline.WorldToSpline(p).PointId);
        _sessionManager = sessionManager;
        _aiParams = serverConfiguration.Extra.AiParams;
        (_lapArc, _lapOrder, _lapLength) = LapDistances(spline.Points);
        if (_configuration.HoldBotSpacing && _lapArc == null)
        {
            Log.Warning("GradientDrift: the AI line is not one closed loop - HoldBotSpacing does nothing");
        }

        string trackName = Path.GetFileName(serverConfiguration.Server.Track);
        _roadMeshPoints = LoadRoadMesh(trackName + RoadMeshFileSuffix);
        _roadMeshTree = _roadMeshPoints.Length > 0 ? new MeshKdTree(_roadMeshPoints) : null;

        TrackTriangle[] sensorTriangles = TrackGeometry.Load(trackName + TrackGeometry.SensorFileSuffix);
        TrackTriangle[] collisionTriangles = TrackGeometry.Load(trackName + TrackGeometry.CollisionFileSuffix);
        TriangleGrid? sensorGrid = null;
        if (sensorTriangles.Length > 0 && collisionTriangles.Length > 0)
        {
            sensorGrid = new TriangleGrid(sensorTriangles, _configuration.RaycastGridTargetTrianglesPerCell);
            _collisionGrid = new TriangleGrid(collisionTriangles, _configuration.RaycastGridTargetTrianglesPerCell);
        }

        DriftPolicy? policy = LoadPolicy(serverConfiguration.Server.Track);
        if (policy != null && !ApplyPolicySettings(policy))
        {
            policy = null;
        }
        // the sensors after the policy file's settings (it may set the pins' height)
        if (sensorGrid != null && _collisionGrid != null)
        {
            TriangleGrid collisionGrid = _collisionGrid;
            _raycastFan = new RaycastFan(sensorGrid, _configuration.RaycastFanRayCount,
                _configuration.RaycastFanAngleDegrees, _configuration.RaycastFanMaxDistanceMeters,
                _configuration.RaycastHeightOffsetMeters);
            _proximityPins = new ProximityPins(sensorGrid, _configuration.ProximityPinMaxDistanceMeters,
                _configuration.ProximityPinHeightOffsetMeters);
            _contactPins = new ProximityPins(collisionGrid, _configuration.CollisionContactDistanceMeters,
                _configuration.ProximityPinHeightOffsetMeters, wallsOnly: true);
        }

        if (policy != null && policy.ReadsCourseRoom)
        {
            // the policy reads how much room it has to the course's edges: no edges, no policy
            _courseEdges = CourseEdges.Load(trackName + CourseEdges.FileSuffix);
            if (_courseEdges == null || _courseEdges.PointCount != spline.Points.Length)
            {
                Log.Warning("GradientDrift: this policy needs {File} with one row per AI-line point ({Points}; drift_rl/setup_track.py writes it) - the stock AI drives",
                    trackName + CourseEdges.FileSuffix, spline.Points.Length);
                policy = null;
            }
        }
        if (policy != null && policy.ReadsPaceOver)
        {
            // the policy reads how far over its asked speed it is: no reference speeds, no policy
            _paceReferenceKmh = PaceReference.Load(trackName + PaceReference.FileSuffix);
            if (_paceReferenceKmh == null || _paceReferenceKmh.Length != spline.Points.Length || _lapArc == null)
            {
                Log.Warning("GradientDrift: this policy needs {File} with one row per AI-line point ({Points}; the trainer writes it) and an AI line that is one closed loop - the stock AI drives",
                    trackName + PaceReference.FileSuffix, spline.Points.Length);
                _paceReferenceKmh = null;
                policy = null;
            }
        }
        _policy = policy;
        // the configured car, after the policy file's settings; any-car policies give each bot its own
        _defaultCar = CarSpec.FromConfiguration(_configuration);
        _car = _defaultCar;
        if (_policy != null && _policy.ReadsSkill)
        {
            _skill = Math.Clamp(_configuration.BotSkill, _policy.SkillLow, _policy.SkillHigh);
            VerboseLog.Information("GradientDrift: the bots drive at skill {Skill} (BotSkill; this policy was trained for {Low} to {High})",
                _skill, _policy.SkillLow, _policy.SkillHigh);
        }
        if (_policy != null && _policy.ReadsCar)
        {
            VerboseLog.Information("GradientDrift: this policy drives any car - each bot uses its car's numbers from cars/<model>.json beside the plugin, with an angle kit of at least {Lock} degrees",
                _policy.DriftSteerLockDeg);
        }
        if (_policy != null)
        {
            _rearSlidingGrip = RearSlidingGripFor(_policy, _configuration.RearSlidingGrip);
            if (_policy.ReadsPace)
            {
                _paceFloor = _policy.PaceIsFraction && _configuration.BotPaceMinPercent > 0f
                    ? Math.Clamp(_configuration.BotPaceMinPercent / 100f, _policy.PaceLow, _policy.PaceHigh)
                    : _policy.PaceLow;
                _basePace = Math.Max(_paceFloor,
                    PaceFor(_policy, _policy.PaceIsFraction ? _configuration.BotPacePercent / 100f : _configuration.BotPaceKmh));
                if (_policy.PaceIsFraction && _paceFloor > _policy.PaceLow)
                {
                    VerboseLog.Information("GradientDrift: spacing keeps the bots at {Floor}% of flat-out speed or faster (BotPaceMinPercent)",
                        _paceFloor * 100f);
                }
            }
        }
    }

    // AssettoServer's own version, "0.0.54" of its "0.0.54+51737e2c2e", or null if it has none.
    private static string? RunningAssettoServer()
    {
        string? version = typeof(ACServer).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        Match match = Regex.Match(version ?? "", @"^\d+\.\d+\.\d+");
        return match.Success ? match.Value : null;
    }

    private static void WarnIfOtherAssettoServer()
    {
        string? running = RunningAssettoServer();
        if (running != BuiltForAssettoServer)
        {
            Log.Warning("GradientDrift: this plugin is built and tested for AssettoServer {Built}, this server runs {Running} - the bots may misbehave; use the plugin release made for this version, or build the plugin against it",
                BuiltForAssettoServer, running ?? "a version that could not be read");
        }
    }

    // A policy file carries the plugin settings it was trained with where they are not this
    // plugin's defaults (its steering lock, for one). They replace the configured values: the
    // policy has never driven a car with any others. False, with the reason logged and nothing
    // changed, if the file asks for something this build cannot give it.
    private bool ApplyPolicySettings(DriftPolicy policy)
    {
        foreach ((string name, float value) in policy.PluginSettings)
        {
            bool usable = float.IsFinite(value) && name switch
            {
                "MaxSteerAngleDeg" => value > 0f && value <= 90f,
                "WrongWayEntryThresholdDeg" => value > 0f && value <= 180f,
                "WrongWayEntryTicks" => value >= 1f && value <= 1000f && value == MathF.Round(value),
                "WrongWayExitSlipDeg" => value > 0f && value <= 180f,
                "WrongWayExitYawRateDegS" => value > 0f,
                "EnableRearThrustLimit" => value == 0f || value == 1f,
                "EnableRearBrakeLimit" => value == 0f || value == 1f,
                "EnableRearFrictionCircle" => value == 0f || value == 1f,
                "MaxSteerRateDegS" => value >= 0f,
                "EnableTireLoadSensitivity" => value == 0f || value == 1f,
                "FrontTireLoadSensitivityPerKn" => value <= 0f && value > -1f,
                "RearTireLoadSensitivityPerKn" => value <= 0f && value > -1f,
                "FrontTrackMeters" => value > 0.5f && value < 3f,
                "RearTrackMeters" => value > 0.5f && value < 3f,
                "RollStiffnessFrontShare" => value > 0f && value < 1f,
                "CgHeightMeters" => value > 0.1f && value < 2f,
                "ProximityPinHeightOffsetMeters" => value > -0.4f && value < 1f,
                "DisableBrakes" => value == 0f || value == 1f,
                "FrontBrakeOnly" => value == 0f || value == 1f,
                "CarOutlineContact" => value == 0f || value == 1f,
                _ => false,
            };
            if (!usable)
            {
                Log.Warning("GradientDrift: the policy was trained with a setting this build cannot apply ({Name} = {Value}) - the stock AI drives",
                    name, value);
                return false;
            }
        }

        // The parts of the vehicle model a policy has to have been trained with: off unless
        // its file turns them on, whatever extra_cfg.yml says.
        _configuration.EnableRearThrustLimit = false;
        _configuration.EnableRearBrakeLimit = false;
        _configuration.EnableRearFrictionCircle = false;
        _configuration.MaxSteerRateDegS = 0f;
        _configuration.EnableTireLoadSensitivity = false;
        _configuration.DisableBrakes = false;
        _configuration.FrontBrakeOnly = false;
        _configuration.CarOutlineContact = false;

        foreach ((string name, float value) in policy.PluginSettings)
        {
            switch (name)
            {
                case "MaxSteerAngleDeg":
                    _configuration.MaxSteerAngleRadians = value * MathF.PI / 180f;
                    break;
                case "WrongWayEntryThresholdDeg":
                    _configuration.WrongWayEntryThresholdDeg = value;
                    break;
                case "WrongWayEntryTicks":
                    _configuration.WrongWayEntryTicks = (int)value;
                    break;
                case "WrongWayExitSlipDeg":
                    _configuration.WrongWayExitSlipDeg = value;
                    break;
                case "WrongWayExitYawRateDegS":
                    _configuration.WrongWayExitYawRateDegS = value;
                    break;
                case "EnableRearThrustLimit":
                    _configuration.EnableRearThrustLimit = value == 1f;
                    break;
                case "EnableRearBrakeLimit":
                    _configuration.EnableRearBrakeLimit = value == 1f;
                    break;
                case "EnableRearFrictionCircle":
                    _configuration.EnableRearFrictionCircle = value == 1f;
                    break;
                case "MaxSteerRateDegS":
                    _configuration.MaxSteerRateDegS = value;
                    break;
                case "EnableTireLoadSensitivity":
                    _configuration.EnableTireLoadSensitivity = value == 1f;
                    break;
                case "FrontTireLoadSensitivityPerKn":
                    _configuration.FrontTireLoadSensitivityPerKn = value;
                    break;
                case "RearTireLoadSensitivityPerKn":
                    _configuration.RearTireLoadSensitivityPerKn = value;
                    break;
                case "FrontTrackMeters":
                    _configuration.FrontTrackMeters = value;
                    break;
                case "RearTrackMeters":
                    _configuration.RearTrackMeters = value;
                    break;
                case "RollStiffnessFrontShare":
                    _configuration.RollStiffnessFrontShare = value;
                    break;
                case "CgHeightMeters":
                    _configuration.CgHeightMeters = value;
                    break;
                case "ProximityPinHeightOffsetMeters":
                    _configuration.ProximityPinHeightOffsetMeters = value;
                    break;
                case "DisableBrakes":
                    _configuration.DisableBrakes = value == 1f;
                    break;
                case "FrontBrakeOnly":
                    _configuration.FrontBrakeOnly = value == 1f;
                    break;
                case "CarOutlineContact":
                    _configuration.CarOutlineContact = value == 1f;
                    break;
            }
            VerboseLog.Information("GradientDrift: {Name} = {Value}, from the policy file", name, value);
        }
        return true;
    }

    // The policy has only driven cars whose rear sliding grip was inside its training range. A
    // setting outside it would be a car it has never seen, so the nearest end is used instead
    // (the top of the range for a setting that is not a number).
    // The numbers this AI car drives with: its model's car file for an any-car policy (read once,
    // then kept), else - or if the file is missing - the configured car.
    private CarSpec CarFor(AiState aiState, DriftPolicy policy)
    {
        if (!policy.ReadsCar)
        {
            return _defaultCar;
        }
        string model = aiState.EntryCar.Model;
        if (_carSpecs.TryGetValue(model, out CarSpec? known))
        {
            return known;
        }
        string pluginDirectory = Path.GetDirectoryName(GetType().Assembly.Location) ?? "";
        string path = Path.Combine(pluginDirectory, "cars", model + ".json");
        CarSpec? spec = CarSpec.Load(path, policy.DriftSteerLockDeg * MathF.PI / 180f, out string? problem);
        if (spec == null)
        {
            Log.Warning("GradientDrift: no usable car file for {Model} ({Path}: {Problem}) - its bots drive with the configured car's numbers; run setup to write it",
                model, path, problem);
            spec = _defaultCar;
        }
        else
        {
            VerboseLog.Information("GradientDrift: {Model}: {Mass:F0} kg, wheelbase {Wheelbase:F2} m, lock {Lock:F0} deg, engine {Engine}, from {Path}",
                model, spec.MassKg, spec.DistanceCgToFrontAxleM + spec.DistanceCgToRearAxleM,
                spec.MaxSteerAngleRadians * 180f / MathF.PI, spec.EngineForceN != null ? "limited" : "unlimited", path);
        }
        _carSpecs[model] = spec;
        return spec;
    }

    private static float RearSlidingGripFor(DriftPolicy policy, float configured)
    {
        float used = float.IsFinite(configured)
            ? Math.Clamp(configured, policy.RearSlidingGripLow, policy.RearSlidingGripHigh)
            : policy.RearSlidingGripHigh;
        if (used != configured)
        {
            Log.Warning("GradientDrift: RearSlidingGrip is {Configured}, but this policy was trained for {Low} to {High} - using {Used}",
                configured, policy.RearSlidingGripLow, policy.RearSlidingGripHigh, used);
        }
        else
        {
            VerboseLog.Information("GradientDrift: RearSlidingGrip {Used} (this policy was trained for {Low} to {High})",
                used, policy.RearSlidingGripLow, policy.RearSlidingGripHigh);
        }
        return used;
    }

    // The policy has only been asked for paces inside its training range: the setting is held
    // there, and 0 (or anything not a number) means the top of it, the fastest it was trained
    // for. In the policy's units: a fraction of the course's flat-out speed (BotPacePercent / 100),
    // or km/h for gen4.
    private static float PaceFor(DriftPolicy policy, float configured)
    {
        float used = float.IsFinite(configured) && configured > 0f
            ? Math.Clamp(configured, policy.PaceLow, policy.PaceHigh)
            : policy.PaceHigh;
        float scale = policy.PaceIsFraction ? 100f : 1f;
        string unit = policy.PaceIsFraction ? "% of the course's flat-out speed" : " km/h";
        if (float.IsFinite(configured) && configured > 0f && used != configured)
        {
            Log.Warning("GradientDrift: {Setting} is {Configured}, but this policy was trained for {Low} to {High}{Unit} - using {Used}",
                policy.PaceIsFraction ? "BotPacePercent" : "BotPaceKmh", configured * scale, policy.PaceLow * scale,
                policy.PaceHigh * scale, unit, used * scale);
        }
        else
        {
            VerboseLog.Information("GradientDrift: bots drive at {Used}{Unit} (this policy was trained for {Low} to {High}); spacing them varies their pace, not their grip",
                used * scale, unit, policy.PaceLow * scale, policy.PaceHigh * scale);
        }
        return used;
    }

    // The policy, or null with the reason logged. Everything it was trained with has to exist on
    // this server, or it would be reading inputs it has never seen.
    private DriftPolicy? LoadPolicy(string serverTrack)
    {
        if (!_configuration.EnableLearnedPolicy)
        {
            Log.Information("GradientDrift: EnableLearnedPolicy is off - the stock AI drives");
            return null;
        }

        if (_collisionGrid == null)
        {
            Log.Warning("GradientDrift: no track geometry, so no sensors for the policy - the stock AI drives");
            return null;
        }

        if (_configuration.RaycastFanRayCount != DriftPolicy.FanRayCount)
        {
            Log.Warning("GradientDrift: the policy reads {Trained} fan rays, RaycastFanRayCount is {Configured} - the stock AI drives",
                DriftPolicy.FanRayCount, _configuration.RaycastFanRayCount);
            return null;
        }

        string policyPath = _configuration.LearnedPolicyPath;
        if (string.IsNullOrWhiteSpace(policyPath))
        {
            string pluginDirectory = Path.GetDirectoryName(GetType().Assembly.Location) ?? "";
            policyPath = Path.Combine(pluginDirectory, "GradientDriftPlugin.policy.json");
        }

        DriftPolicy? policy = DriftPolicy.TryLoad(policyPath, _spline.Points);
        if (policy == null)
        {
            Log.Warning("GradientDrift: no usable policy - the stock AI drives");
            return null;
        }

        // Contains, not Equals: Server.Track can carry a path prefix in front of the folder name. A
        // policy trained on generated courses as well ("generated" in its file) has learned courses
        // in general and drives any track; one trained only on real tracks drives only those.
        bool trainedHere = policy.Track.Length == 0
            || policy.Tracks.Append(policy.Track).Any(t => serverTrack.Contains(t, StringComparison.OrdinalIgnoreCase));
        if (!trainedHere && !policy.TrainedOnGeneratedCourses)
        {
            Log.Warning("GradientDrift: the policy was trained on '{PolicyTrack}', this server runs '{ServerTrack}' - the stock AI drives",
                policy.Track, serverTrack);
            return null;
        }
        if (!trainedHere)
        {
            Log.Information("GradientDrift: '{ServerTrack}' is new to this policy (trained on {Tracks} and generated courses)",
                serverTrack, string.Join(", ", policy.Tracks.Append(policy.Track).Distinct()));
        }

        return policy;
    }

    // "x,y,z" rows after one header line. A missing file is normal: the ground ray does not
    // need it.
    private static Vector3[] LoadRoadMesh(string path)
    {
        if (!File.Exists(path))
        {
            VerboseLog.Information("GradientDrift: no road mesh file at {Path} (only used where the ground ray finds nothing)",
                Path.GetFullPath(path));
            return Array.Empty<Vector3>();
        }

        var points = new List<Vector3>();
        foreach (string line in File.ReadLines(path).Skip(1))
        {
            string[] parts = line.Split(',');
            if (parts.Length == 3
                && float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out float x)
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float y)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
            {
                points.Add(new Vector3(x, y, z));
            }
        }

        VerboseLog.Information("GradientDrift: loaded road mesh ({Count} points) from {Path}", points.Count, Path.GetFullPath(path));
        return points.ToArray();
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (_policy == null)
        {
            return Task.CompletedTask;
        }

        // The stock AI only spawns a car once a player is near. Spawn them all now so they
        // drive (and can be watched or logged) on an empty server too. Spread evenly round the
        // lap: started at the same point, cars driven by the same policy stay on top of each other.
        var uninitialized = new List<AiState>();
        var toSpawn = new List<(EntryCar Car, AiState State)>();
        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            if (!car.AiControlled)
            {
                continue;
            }

            _aiStates.Clear();
            uninitialized.Clear();
            car.GetInitializedStates(_aiStates, uninitialized);
            foreach (AiState aiState in uninitialized)
            {
                toSpawn.Add((car, aiState));
            }
        }

        for (int k = 0; k < toSpawn.Count; k++)
        {
            (EntryCar car, AiState aiState) = toSpawn[k];
            int pointId = (int)((long)k * _spline.Points.Length / toSpawn.Count);
            try
            {
                aiState.Teleport(pointId);
                VerboseLog.Information("GradientDrift: spawned AI car {SessionId} at server start, spline point {PointId}",
                    car.SessionId, pointId);
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "GradientDrift: could not spawn AI car {SessionId}", car.SessionId);
            }
        }

        if (_configuration.EnableTelemetryLogging)
        {
            try
            {
                _telemetry = new DriftTelemetry((long)(Math.Max(0f, _configuration.TelemetryMaxFileMegabytes) * 1024 * 1024));
                Log.Information("GradientDrift: telemetry logging to {Path}", Path.GetFullPath(_telemetry.FilePath));
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                Log.Warning("GradientDrift: no telemetry ({Message})", e.Message);
            }
        }

        _acServer.Value.Update += OnUpdate;
        return Task.CompletedTask;
    }

    public override void Dispose()
    {
        if (_acServer.IsValueCreated)
        {
            _acServer.Value.Update -= OnUpdate;
        }

        // Cleared before it is closed, so a tick still running writes nothing rather than to a
        // closed file.
        DriftTelemetry? telemetry = _telemetry;
        _telemetry = null;
        telemetry?.Dispose();
        base.Dispose();
    }

    // Runs on the server's update thread, after the stock AI has moved every car for this tick.
    // AssettoServer skips the rest of a tick whose update throws (no car is sent to anyone) and
    // shuts down after 10 in a row, so nothing from here may reach it: an error is logged, and
    // if it keeps happening the plugin stops and the stock AI drives.
    private void OnUpdate(ACServer sender, EventArgs args)
    {
        if (_updatesStopped)
        {
            return;
        }
        try
        {
            UpdateCars();
            _botNames?.Update();
            _failedUpdates = 0;
        }
        catch (Exception ex)
        {
            _failedUpdates++;
            if (_failedUpdates <= 3)
            {
                Log.Error(ex, "GradientDrift: error in the tick update ({Count} in a row)", _failedUpdates);
            }
            if (_failedUpdates >= MaxFailedUpdates)
            {
                _updatesStopped = true;
                Log.Error("GradientDrift: {Count} tick updates in a row failed - stopped; the stock AI drives", _failedUpdates);
            }
        }
    }

    private void UpdateCars()
    {
        DriftPolicy? policy = _policy;
        TriangleGrid? collision = _collisionGrid;
        RaycastFan? fan = _raycastFan;
        ProximityPins? pins = _proximityPins;
        ProximityPins? contactPins = _contactPins;
        if (policy == null || collision == null || fan == null || pins == null || contactPins == null)
        {
            return;
        }

        _tickCounter++;
        long tickStart = PluginTiming.Now();

        long nowMs = _sessionManager.ServerTimeMilliseconds;
        float dt = _lastUpdateTimeMs < 0
            ? 1f / _tickRateHz
            : Math.Clamp((nowMs - _lastUpdateTimeMs) / 1000f, 0f, _configuration.MaxRealDtSeconds);
        _lastUpdateTimeMs = nowMs;

        HumanCarSample human = SampleHumanCar();

        _driven.Clear();
        _trainSlots.Clear();
        int aiEntry = -1;
        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            if (car.AiMode != AiMode.None)
            {
                aiEntry++;         // counted whether a player has the slot or not, so trains keep their places
            }
            if (!car.AiControlled || car.Client != null)
            {
                continue;
            }

            _aiStates.Clear();
            car.GetInitializedStates(_aiStates);
            for (int k = 0; k < _aiStates.Count; k++)
            {
                _driven.Add((car.SessionId, _aiStates[k]));
                if (k == 0 && aiEntry >= 0)
                {
                    _trainSlots[_aiStates[k]] = aiEntry;
                }
            }
        }

        FormTrains();
        HoldSpacing(policy, dt);

        int aiCarsUpdated = 0;
        foreach (var (sessionId, aiState) in _driven)
        {
            UpdateCar(sessionId, aiState, dt, policy, collision, fan, pins, contactPins, in human);
            aiCarsUpdated++;
            if (_followers.Count > 0)
            {
                RecordLead(aiState, nowMs);
            }
        }
        int policyCars = aiCarsUpdated;
        _aheadLead = null;
        long followStart = PluginTiming.Now();
        foreach (var (sessionId, aiState, lead, place) in _followers)
        {
            Follow(sessionId, aiState, lead, place, nowMs, dt, collision, in human);
            aiCarsUpdated++;
        }
        _timing.Add(PluginTiming.Part.Followers, followStart);
        _health.Check(_driven, nowMs);

        // No AI car out (AssettoServer spawns them only once a player is on the racing line):
        // still log the human car, so a calibration drive near the pits is not lost. SessionId
        // 255 marks a row with no AI car in it.
        DriftTelemetry? telemetry = _telemetry;
        if (aiCarsUpdated == 0 && human.SessionId >= 0 && telemetry != null)
        {
            var row = new DriftTelemetryRow
            {
                Tick = _tickCounter,
                ServerTimeMs = nowMs,
                SessionId = HumanOnlySessionId,
                DtSeconds = dt,
                PolicyObservation = "",
                GapAheadM = -1f,
                FollowLeadId = -1,
            };
            telemetry.Write(in row, in human);
        }

        long flushStart = PluginTiming.Now();
        _telemetry?.FlushIfDue(_tickCounter);
        _timing.Add(PluginTiming.Part.Telemetry, flushStart);
        _timing.EndTick(tickStart, policyCars, aiCarsUpdated - policyCars, nowMs);
    }

    private void UpdateCar(byte sessionId, AiState aiState, float dt, DriftPolicy policy, TriangleGrid collision,
        RaycastFan fan, ProximityPins pins, ProximityPins contactPins, in HumanCarSample human)
    {
        _car = CarFor(aiState, policy);
        // What the stock AI did with this car this tick. Its velocity is used for one thing
        // only: to tell whether the server is moving the car at all.
        Vector3 stockVelocity = aiState.Status.Velocity;
        byte spawnCounter = aiState.SpawnCounter;

        if (!_cars.TryGetValue(aiState, out DriftCarState? state) || state.SpawnCounter != spawnCounter)
        {
            if (state != null)
            {
                VerboseLog.Information("GradientDrift: AI car {SessionId} was re-teleported by the server - starting it afresh", sessionId);
            }

            // Take the car over once the stock AI has it moving; until then it stays stock.
            if (stockVelocity.Length() < 1.0f)
            {
                _cars.Remove(aiState);
                return;
            }

            state = new DriftCarState(aiState.Status.Position, stockVelocity, spawnCounter)
            {
                RearSlidingGrip = _rearSlidingGrip,
                Pace = _basePace,
            };
            PlaceInLargestGap(sessionId, state);
            _cars[aiState] = state;
        }

        Vector3 position = state.Position;
        Vector3 velocity = state.Velocity;

        var row = new DriftTelemetryRow
        {
            Tick = _tickCounter,
            ServerTimeMs = _sessionManager.ServerTimeMilliseconds,
            SessionId = sessionId,
            SpawnCounter = spawnCounter,
            DtSeconds = dt,
            MaxSpeedMs = aiState.MaxSpeed,
            NearestPointId = -1,
            PolicyObservation = "",
            RearSlidingGrip = state.RearSlidingGrip,
            GapAheadM = state.GapAheadM,
            Pace = state.Pace,
            PaceOver = state.Policy.PaceOver,
            FollowLeadId = -1,
        };

        float crossTrackError = 0f;
        float steerCommand = 0f;
        // For the drivetrain visual: speed along the heading, how fast the front wheels roll,
        // how much faster than the heading speed the rear tread is moving, and the change in
        // forward speed. All zero for a stopped car.
        float drivetrainForwardSpeed = 0f;
        float drivetrainFrontWheelSpeed = 0f;
        float drivetrainRearSpin = 0f;
        float drivetrainLongitudinalAccel = 0f;
        float drivetrainPedal = float.NaN;          // the pedal, where the policy's speed output is one
        // The policy is asking for less speed than the car has (or braking, with the pedal). Null while the stock AI has
        // the car stopped: the lights are then left as it set them.
        bool? slowing = null;
        // The acceleration the car is making along and across itself, in g, for the body lean.
        float bodyAccelForwardG = 0f;
        float bodyAccelSidewaysG = 0f;
        // An axle's tires are slipping enough to leave rubber (SkidSlipOverPeak).
        bool frontSkidding = false;
        bool rearSkidding = false;

        float stockHorizontalSpeed = new Vector2(stockVelocity.X, stockVelocity.Z).Length();
        // With trains, a follower is a few metres behind its lead, and the stock AI measures the gap
        // to the car ahead between the positions this plugin sends: it would stop the lead for its
        // own follower. So a car the plugin drives in a train ignores the stock AI's stops.
        bool ignoreStockStop = _configuration.TrainSize > 1 && state.OwnSpeed.HasValue;
        int paceAskedPointId = -1;          // the point whose reference speed this tick is asked for
        state.HeldByStock = stockHorizontalSpeed <= 0.01f && !ignoreStockStop;
        if (!state.HeldByStock)
        {
            // ---- 1. Where the car is on the racing line ------------------------------------

            var (rawNearestPointId, _) = _spline.WorldToSpline(position);

            // Last tick's speed: both correctors ask "how far could the car have moved?".
            float speedForPlausibility = state.OwnSpeed ?? 0.5f;
            NearestPointCorrectionResult nearestPoint = NearestPointCorrector.Correct(rawNearestPointId,
                state.PreviousNearestPointId, state.NearestPointCorrectionStreak, speedForPlausibility, dt,
                position, _spline.Points);
            int nearestPointId = nearestPoint.CorrectedPointId;
            state.PreviousNearestPointId = nearestPointId;
            state.NearestPointCorrectionStreak = nearestPoint.ConsecutiveCorrectionStreak;
            paceAskedPointId = nearestPointId;

            var (segmentA, segmentB, segmentT) = FindInterpolationSegment(position, nearestPointId);
            Vector3 rawForward = Vector3.Lerp(
                _spline.Operations.GetForwardVector(segmentA),
                _spline.Operations.GetForwardVector(segmentB),
                segmentT);

            // GetForwardVector is zero at a spline end; fall back to the stock AI's direction.
            Vector3 rawLineDirection = rawForward.LengthSquared() > 0.0001f
                ? Vector3.Normalize(new Vector3(rawForward.X, 0f, rawForward.Z))
                : stockHorizontalSpeed > 0.01f
                    ? new Vector3(stockVelocity.X, 0f, stockVelocity.Z) / stockHorizontalSpeed
                    : new Vector3(MathF.Cos(state.Heading ?? 0f), 0f, MathF.Sin(state.Heading ?? 0f));
            float rawLineAngle = MathF.Atan2(rawLineDirection.Z, rawLineDirection.X);

            GlitchCorrectionResult glitch = GlitchCorrector.Correct(rawLineAngle, state.PreviousLineAngle,
                speedForPlausibility, dt, _configuration.MaxLateralAccelMs2);
            float lineAngle = glitch.CorrectedAngle;
            state.PreviousLineAngle = lineAngle;

            Vector3 lineDirection = new Vector3(MathF.Cos(lineAngle), 0f, MathF.Sin(lineAngle));
            Vector3 pathPosition = Vector3.Lerp(
                _spline.Points[segmentA].Position, _spline.Points[segmentB].Position, segmentT);
            crossTrackError = CrossTrackErrorCalculator.Compute(position, pathPosition, lineDirection);

            float radius = nearestPointId >= 0 && nearestPointId < _spline.Points.Length
                ? _spline.Points[nearestPointId].Radius
                : float.MaxValue;

            // The car's heading: its own state, or on its first tick the direction it is moving.
            bool firstTick = !state.Heading.HasValue;
            Vector3 horizontalVelocity = new Vector3(velocity.X, 0f, velocity.Z);
            Vector3 travelDirection = horizontalVelocity.LengthSquared() > 0.0001f
                ? Vector3.Normalize(horizontalVelocity)
                : lineDirection;
            float heading = state.Heading ?? MathF.Atan2(travelDirection.Z, travelDirection.X);

            // ---- 2. The policy -------------------------------------------------------------

            long networkStart = PluginTiming.Now();
            policy.BuildObservation(_observation, firstTick,
                firstTick ? 0f : (state.OwnSpeed ?? 0f),
                state.LateralVelocity, state.YawRate, crossTrackError, radius, nearestPointId, lineAngle,
                aiState.MaxSpeed, state.Policy, heading, state.PreviousPins, dt, state.WrongWayRecovering,
                _car.MaxSteerAngleRadians, state.RearSlidingGrip, _courseEdges, state.Pace, state.Policy.PaceOver,
                _car, _configuration.EnableTireLoadSensitivity, _skill);

            DriftPolicyAction action = policy.Evaluate(_observation);
            _timing.Add(PluginTiming.Part.Network, networkStart);
            float steerAction = action.Valid ? action.Steer : state.Policy.PreviousSteerAction;
            float speedAction = action.Valid ? action.Speed : state.Policy.PreviousSpeedAction;
            float brakeAction = action.Valid ? action.Brake : state.Policy.PreviousBrakeAction;
            if (policy.ReadsSkill && _skill < 1f)
            {
                (steerAction, speedAction, brakeAction) = HandicapBySkill(state, firstTick, steerAction, speedAction, brakeAction);
            }
            state.Policy.PreviousSteerAction = steerAction;
            state.Policy.PreviousSpeedAction = speedAction;
            state.Policy.PreviousBrakeAction = brakeAction;

            steerCommand = _car.MaxSteerAngleRadians * steerAction;
            if (_configuration.MaxSteerRateDegS > 0f && !firstTick)
            {
                float maxStep = _configuration.MaxSteerRateDegS * (MathF.PI / 180f) * dt;
                float lastSteer = state.Policy.LastSteerCommandRadians;
                steerCommand = MathF.Max(lastSteer - maxStep, MathF.Min(lastSteer + maxStep, steerCommand));
            }
            float desiredSpeed = aiState.MaxSpeed * 0.5f * (1f + speedAction);

            // ---- 3. Forward speed ----------------------------------------------------------

            float previousOwnSpeed = state.OwnSpeed ?? desiredSpeed;
            (_muFront, _muRear) = AxleFriction(state.OwnSpeed ?? 0f, state.YawRate);
            bool pedal = _configuration.EnableRearFrictionCircle;
            if (pedal && _configuration.DisableBrakes)
            {
                // lift-off at most and no brake action; the policy's own outputs, kept above, are
                // what it reads back next tick, as in training (fast_sim.step_post)
                speedAction = MathF.Max(0f, speedAction);
                brakeAction = 0f;
            }
            if (pedal && _configuration.FrontBrakeOnly)
            {
                // no foot brake; the brake action (front axle, with the throttle) stays (fast_sim.step_post)
                speedAction = MathF.Max(0f, speedAction);
            }
            slowing = pedal ? speedAction < 0f || brakeAction >= BrakeLightMinBrake : desiredSpeed < previousOwnSpeed;
            float ownSpeed;
            float thrustRearSpin = -1f;        // set by the rear thrust limit or the pedal when they run
            float rearCapacityN = -1f;         // sideways force the rear axle has left (pedal); < 0 = not in use
            float frontCapacityN = -1f;        // the same for the front axle while it brakes
            if (!state.OwnSpeed.HasValue)
            {
                ownSpeed = desiredSpeed;
            }
            else if (pedal && dt > 0.0001f)
            {
                ownSpeed = DriveByPedal(state, previousOwnSpeed, speedAction, brakeAction, aiState.MaxSpeed, dt,
                    out rearCapacityN, out frontCapacityN, out thrustRearSpin);
                drivetrainPedal = speedAction;
            }
            else
            {
                ownSpeed = state.OwnSpeed.Value;
                float rampRate = desiredSpeed > ownSpeed
                    ? aiState.EntryCar.AiAcceleration
                    : MathF.Min(aiState.EntryCar.AiDeceleration, _configuration.MaxLongitudinalDecelMs2);
                float maxDelta = rampRate * dt;
                ownSpeed += Math.Clamp(desiredSpeed - ownSpeed, -maxDelta, maxDelta);
                if (_configuration.EnableRearThrustLimit && dt > 0.0001f)
                {
                    ownSpeed = LimitByRearThrust(state, previousOwnSpeed, ownSpeed, dt, out thrustRearSpin);
                }
            }
            state.OwnSpeed = ownSpeed;

            float longitudinalAccel = dt > 0.0001f ? (ownSpeed - previousOwnSpeed) / dt : 0f;
            float forwardSpeed = MathF.Max(ownSpeed, 0.5f);

            // ---- 4. Lateral dynamics -------------------------------------------------------

            float vehicleMassKg = (_car.FrontStaticLoadN + _car.RearStaticLoadN) / 9.80665f;
            float frontPeakForceN = _muFront * _car.FrontStaticLoadN;
            float rearPeakForceN = _muRear * _car.RearStaticLoadN;

            // Friction ellipse: the force accelerating the car is taken out of what the rear
            // tires have left for cornering. With the rear thrust limit it is the other way
            // round - sideways force first, thrust from what is left - so this is off.
            float rearDriveForceN = longitudinalAccel > 0f && !_configuration.EnableRearThrustLimit && rearCapacityN < 0f
                ? vehicleMassKg * longitudinalAccel
                : 0f;
            float rearAvailableForceN = MathF.Sqrt(MathF.Max(0f,
                rearPeakForceN * rearPeakForceN - rearDriveForceN * rearDriveForceN));
            float effectiveRearPeakFriction = _car.RearStaticLoadN > 0.0001f
                ? rearAvailableForceN / _car.RearStaticLoadN
                : 0f;

            int subStepCount = Math.Max(1, (int)MathF.Ceiling(dt / _configuration.MaxLateralDynamicsSubStepSeconds));
            float subDt = dt / subStepCount;
            float cosSteer = MathF.Cos(steerCommand);
            float slipAngleSpeed = MathF.Max(forwardSpeed, _configuration.MinSlipAngleSpeedMs);

            float yawRate = state.YawRate;
            float lateralVelocity = state.LateralVelocity;
            float frontSlipAngle = 0f, rearSlipAngle = 0f, fyFront = 0f, fyRear = 0f;

            for (int subStep = 0; subStep < subStepCount; subStep++)
            {
                frontSlipAngle = MathF.Atan2(lateralVelocity + _car.DistanceCgToFrontAxleM * yawRate, slipAngleSpeed) - steerCommand;
                rearSlipAngle = MathF.Atan2(lateralVelocity - _car.DistanceCgToRearAxleM * yawRate, slipAngleSpeed);

                fyFront = frontCapacityN >= 0f
                    ? LateralForceWithin(frontSlipAngle, frontPeakForceN, _car.FrontPeakSlipAngleRadians, frontCapacityN)
                    : LateralTireForce(frontSlipAngle, _muFront,
                        _car.FrontStaticLoadN, _car.FrontPeakSlipAngleRadians, 1f);
                fyRear = rearCapacityN >= 0f
                    ? LateralForceWithin(rearSlipAngle, rearPeakForceN, _car.RearPeakSlipAngleRadians, rearCapacityN)
                    : LateralTireForce(rearSlipAngle, effectiveRearPeakFriction,
                        _car.RearStaticLoadN, _car.RearPeakSlipAngleRadians, state.RearSlidingGrip);

                float yawMomentNm = _car.DistanceCgToFrontAxleM * fyFront * cosSteer - _car.DistanceCgToRearAxleM * fyRear;
                float yawAngularAccel = yawMomentNm / _car.YawInertiaKgM2;
                float newYawRate = yawRate + yawAngularAccel * subDt;

                // Body frame: the "- forwardSpeed * yawRate" term is the frame itself turning.
                float lateralAccelMs2 = (fyFront * cosSteer + fyRear) / vehicleMassKg - forwardSpeed * yawRate;
                float newLateralVelocity = lateralVelocity + lateralAccelMs2 * subDt;

                float newHeading = heading + newYawRate * subDt;
                while (newHeading > MathF.PI) newHeading -= 2 * MathF.PI;
                while (newHeading < -MathF.PI) newHeading += 2 * MathF.PI;

                heading = newHeading;
                yawRate = newYawRate;
                lateralVelocity = newLateralVelocity;
            }

            // ---- 5. Wrong-way recovery (see the configuration) -----------------------------

            bool wrongWayActive = false;
            if (_configuration.EnableWrongWayRecovery)
            {
                if (!state.WrongWayRecovering)
                {
                    float offLineAngle = heading - lineAngle;
                    while (offLineAngle > MathF.PI) offLineAngle -= 2 * MathF.PI;
                    while (offLineAngle < -MathF.PI) offLineAngle += 2 * MathF.PI;

                    if (MathF.Abs(offLineAngle) * (180f / MathF.PI) > _configuration.WrongWayEntryThresholdDeg)
                    {
                        state.WrongWayStreak++;
                        if (state.WrongWayStreak >= _configuration.WrongWayEntryTicks)
                        {
                            state.WrongWayRecovering = true;
                        }
                    }
                    else
                    {
                        state.WrongWayStreak = 0;
                    }
                }

                if (state.WrongWayRecovering)
                {
                    float decayFactor = MathF.Exp(-dt / _configuration.WrongWayRecoveryTimeConstantSeconds);
                    yawRate *= decayFactor;
                    lateralVelocity *= decayFactor;
                    wrongWayActive = true;

                    // It only stops the rotation. Turning back to the line is the policy's job,
                    // so a car still pointing the wrong way is damped again a few ticks later.
                    float slideDeg = MathF.Atan2(MathF.Abs(lateralVelocity), forwardSpeed) * (180f / MathF.PI);
                    if (slideDeg < _configuration.WrongWayExitSlipDeg
                        && MathF.Abs(yawRate) * (180f / MathF.PI) < _configuration.WrongWayExitYawRateDegS)
                    {
                        state.WrongWayRecovering = false;
                        state.WrongWayStreak = 0;
                    }
                }
            }

            state.Policy.LastSteerCommandRadians = steerCommand;
            state.Policy.LastFrontSaturation = frontPeakForceN > 0.000001f ? MathF.Abs(fyFront) / frontPeakForceN : 0f;
            float rearForceLimitN = rearCapacityN >= 0f ? rearCapacityN : rearAvailableForceN;
            state.Policy.LastRearSaturation = rearForceLimitN > 0.000001f ? MathF.Abs(fyRear) / rearForceLimitN : 0f;

            state.Heading = heading;
            state.YawRate = yawRate;
            state.LateralVelocity = lateralVelocity;

            // Wheelspin implied by the tire model. A fully sliding tire's force opposes the
            // tread's slip over the road, so the tread slips lengthwise by (thrust / sideways
            // force) times its sideways slip. The rear thrust limit knows the thrust in use.
            // Without it, the thrust is taken to be all the friction circle leaves: for a rear
            // axle keeping a share s of its peak sideways force, sqrt(1 / s^2 - 1) times its
            // sideways slip. No spin below the peak slip angle either way.
            if (thrustRearSpin >= 0f)
            {
                drivetrainRearSpin = thrustRearSpin;
            }
            else
            {
                float rearGripShare = MathF.Abs(rearSlipAngle) >= _car.RearPeakSlipAngleRadians && rearAvailableForceN > 0.000001f
                    ? Math.Clamp(MathF.Abs(fyRear) / rearAvailableForceN, 0.05f, 1f)
                    : 1f;
                float rearSidewaysSpeed = MathF.Abs(lateralVelocity - _car.DistanceCgToRearAxleM * yawRate);
                drivetrainRearSpin = rearSidewaysSpeed * MathF.Sqrt(1f / (rearGripShare * rearGripShare) - 1f);
            }
            // Slip as the game measures it: sideways (tan of the slip angle) and lengthwise
            // (wheelspin over forward speed) together, over tan of the peak slip angle.
            frontSkidding = MathF.Tan(MathF.Min(MathF.Abs(frontSlipAngle), 1.5f))
                > SkidSlipOverPeak * MathF.Tan(_car.FrontPeakSlipAngleRadians);
            float rearSpinRatio = drivetrainRearSpin / MathF.Max(forwardSpeed, 1f);
            float rearSidewaysSlip = MathF.Tan(MathF.Min(MathF.Abs(rearSlipAngle), 1.5f));
            rearSkidding = MathF.Sqrt(rearSidewaysSlip * rearSidewaysSlip + rearSpinRatio * rearSpinRatio)
                > SkidSlipOverPeak * MathF.Tan(_car.RearPeakSlipAngleRadians);
            drivetrainForwardSpeed = forwardSpeed;
            // A front wheel rolls at the speed of its axle along the way the wheel points,
            // which for a countersteered car is close to the car's whole speed.
            drivetrainFrontWheelSpeed = forwardSpeed * cosSteer
                + (lateralVelocity + _car.DistanceCgToFrontAxleM * yawRate) * MathF.Sin(steerCommand);
            drivetrainLongitudinalAccel = longitudinalAccel;
            bodyAccelForwardG = Math.Clamp((longitudinalAccel - lateralVelocity * yawRate) / 9.80665f, -1.5f, 1.5f);
            bodyAccelSidewaysG = Math.Clamp((fyFront * cosSteer + fyRear) / vehicleMassKg / 9.80665f, -1.5f, 1.5f);

            velocity = new Vector3(
                forwardSpeed * MathF.Cos(heading) - lateralVelocity * MathF.Sin(heading),
                0f,
                forwardSpeed * MathF.Sin(heading) + lateralVelocity * MathF.Cos(heading));

            float travelAngle = velocity.LengthSquared() > 0.0001f
                ? MathF.Atan2(velocity.Z, velocity.X)
                : heading;
            float slipAngle = heading - travelAngle;
            while (slipAngle > MathF.PI) slipAngle -= 2 * MathF.PI;
            while (slipAngle < -MathF.PI) slipAngle += 2 * MathF.PI;

            row.OwnSpeedMs = ownSpeed;
            row.Heading = heading;
            row.YawRate = yawRate;
            row.LateralVelocityMs = lateralVelocity;
            row.SlipAngle = slipAngle;
            row.SteerCommand = steerCommand;
            row.FrontSlipAngle = frontSlipAngle;
            row.RearSlipAngle = rearSlipAngle;
            row.FyFrontN = fyFront;
            row.FyRearN = fyRear;
            row.LongitudinalAccelMs2 = longitudinalAccel;
            row.NearestPointId = nearestPointId;
            row.RawLineAngle = rawLineAngle;
            row.LineAngle = lineAngle;
            row.LineAngleCorrected = glitch.GlitchCorrected;
            row.NearestPointCorrected = nearestPoint.GlitchCorrected;
            row.WrongWayRecovering = wrongWayActive;
            row.PolicyActive = action.Valid;
            row.PolicySteerAction = steerAction;
            row.PolicySpeedAction = speedAction;
            row.PolicyBrakeAction = brakeAction;
            if (_telemetry != null && _configuration.LogLearnedPolicyObservations)
            {
                long formatStart = PluginTiming.Now();
                row.PolicyObservation = DriftPolicy.FormatObservation(_observation);
                _timing.Add(PluginTiming.Part.Telemetry, formatStart);
            }
        }
        else
        {
            // The stock AI has stopped the car (waiting, about to despawn): follow it.
            velocity = new Vector3(stockVelocity.X, 0f, stockVelocity.Z);
        }

        // ---- 6. Move; stay out of walls ------------------------------------------------------

        Vector3 previousPosition = position;
        position += velocity * dt;
        long pushStart = PluginTiming.Now();
        row.WallPushOutM = _configuration.CarOutlineContact
            ? PushOutlineOutOfWalls(collision, state.Heading ?? aiState.Status.Rotation.X + MathF.PI / 2f, ref position, ref velocity)
            : PushOutOfWalls(contactPins, aiState, ref position, ref velocity);
        _timing.Add(PluginTiming.Part.Sensors, pushStart);

        // ---- 7. Ride height, attitude, and what clients are sent -----------------------------

        var (verticalPointId, _) = _spline.WorldToSpline(position);
        float targetHeight;
        float heightSmoothingFactor;
        float attitudeHeading = state.Heading ?? aiState.Status.Rotation.X + MathF.PI / 2f;
        long groundStart = PluginTiming.Now();
        bool foundGround = TryFindGround(collision, position, out float groundY);
        _timing.Add(PluginTiming.Part.Sensors, groundStart);
        if (foundGround)
        {
            // Over a few metres along the way the car is going, the straight line that best fits
            // the ground: the body follows hills but not bumps shorter than the span (a rough road
            // surface showed as the car hopping a few cm every update).
            Vector3 along = new Vector3(velocity.X, 0f, velocity.Z);
            along = along.LengthSquared() > 1f
                ? Vector3.Normalize(along)
                : new Vector3(MathF.Cos(attitudeHeading), 0f, MathF.Sin(attitudeHeading));
            long rideStart = PluginTiming.Now();
            float rideGroundY = RideGroundHeight(collision, position, along, groundY);
            _timing.Add(PluginTiming.Part.Sensors, rideStart);
            targetHeight = rideGroundY + _configuration.GroundRayHeightOffsetM;
            heightSmoothingFactor = _configuration.GroundRayHeightSmoothingFactor;
            state.GroundMissTicks = 0;
            row.GroundRayUsed = true;
            row.GroundRayY = groundY;
        }
        else if (state.SmoothedHeight.HasValue && state.GroundMissTicks * dt < _configuration.GroundRayMissHoldSeconds)
        {
            // A ray that slips through a crack in the ground keeps the last height rather than
            // dropping to the racing line's for one update.
            state.GroundMissTicks++;
            targetHeight = state.SmoothedHeight.Value;
            heightSmoothingFactor = 1f;
        }
        else
        {
            targetHeight = FallbackHeight(position, verticalPointId, crossTrackError);
            heightSmoothingFactor = _configuration.HeightSmoothingFactor;
        }

        float height = targetHeight;
        if (state.SmoothedHeight.HasValue)
        {
            float heightGap = targetHeight - state.SmoothedHeight.Value;
            if (MathF.Abs(heightGap) <= _configuration.HeightSnapThresholdMeters)
            {
                height = state.SmoothedHeight.Value + heightSmoothingFactor * heightGap;
            }
        }
        state.SmoothedHeight = height;
        position.Y = height;

        // The vertical speed clients see is what the height actually did, so their own
        // extrapolation between updates follows the ground.
        float verticalRate = dt > 0.0001f ? (position.Y - previousPosition.Y) / dt : 0f;
        velocity = new Vector3(velocity.X, verticalRate, velocity.Z);

        state.Position = position;
        state.Velocity = velocity;
        // Clients draw the car with its body at the height they are sent, so each car model is
        // sent at its own CG height above the ground (its car file's CG_HEIGHT_M) rather than the
        // one ride height the bot drives and senses at: a low car no longer floats, a tall one
        // no longer sinks its nose into the road.
        aiState.Status.Position = position + new Vector3(0f, _car.CgHeightMeters - _configuration.GroundRayHeightOffsetM, 0f);
        if (_paceReferenceKmh != null)
        {
            UpdatePaceOver(state, position, paceAskedPointId, aiState.MaxSpeed, dt, policy.PaceAverageSeconds);
        }
        aiState.Status.Velocity = velocity;

        WriteSteeringVisual(aiState, state, steerCommand, dt, ref row);
        WriteDrivetrainVisual(aiState, state, drivetrainForwardSpeed, drivetrainFrontWheelSpeed, drivetrainRearSpin,
            drivetrainLongitudinalAccel, drivetrainPedal, dt, ref row);
        if (slowing.HasValue)
        {
            WriteStatusFlags(aiState, state, slowing.Value, frontSkidding, rearSkidding, dt);
        }

        // Pitch and roll: the slope of the ground under the car, along and across the way it
        // points, plus how far the body leans under the acceleration the car is making. Where
        // the rays find no ground the last attitude is held; a car that has none yet takes the
        // racing line's slope and camber.
        long attitudeStart = PluginTiming.Now();
        bool foundAttitude = TryFindGroundAttitude(collision, position, attitudeHeading, out float rawPitch, out float rawRoll);
        _timing.Add(PluginTiming.Part.Sensors, attitudeStart);
        if (foundAttitude)
        {
            rawPitch += _configuration.BodyPitchDegPerG * (MathF.PI / 180f) * bodyAccelForwardG;
            rawRoll += _configuration.BodyRollDegPerG * (MathF.PI / 180f) * bodyAccelSidewaysG;
        }
        else if (state.SmoothedPitch.HasValue && state.SmoothedRoll.HasValue)
        {
            rawPitch = state.SmoothedPitch.Value;
            rawRoll = state.SmoothedRoll.Value;
        }
        else if (verticalPointId >= 0 && verticalPointId < _spline.Points.Length)
        {
            Vector3 lineForward = _spline.Operations.GetForwardVector(verticalPointId);
            rawPitch = lineForward.LengthSquared() > 0.0001f
                ? (MathF.Atan2(new Vector2(lineForward.Z, lineForward.X).Length(), lineForward.Y) - MathF.PI / 2) * -1f
                : 0f;
            rawRoll = _spline.Operations.GetCamber(verticalPointId);
        }
        float pitch = state.SmoothedPitch.HasValue
            ? state.SmoothedPitch.Value + _configuration.PitchRollSmoothingFactor * (rawPitch - state.SmoothedPitch.Value)
            : rawPitch;
        float roll = state.SmoothedRoll.HasValue
            ? state.SmoothedRoll.Value + _configuration.PitchRollSmoothingFactor * (rawRoll - state.SmoothedRoll.Value)
            : rawRoll;
        state.SmoothedPitch = pitch;
        state.SmoothedRoll = roll;

        // Rotation.X is the game's yaw: this plugin's heading (atan2(Z, X)) minus 90 degrees.
        // A stationary car keeps the rotation the stock AI gave it.
        if (velocity.LengthSquared() > 0.0001f)
        {
            float yaw;
            if (state.Heading.HasValue)
            {
                yaw = state.Heading.Value - MathF.PI / 2;
            }
            else
            {
                Vector3 direction = Vector3.Normalize(velocity);
                yaw = MathF.Atan2(direction.Z, direction.X) - MathF.PI / 2;
            }
            aiState.Status.Rotation = new Vector3(yaw, pitch, roll);
        }

        // ---- 8. Sensors for the next tick ----------------------------------------------------

        // The fan looks where the car is going, the pins where it is pointing.
        float horizontalSpeedSq = velocity.X * velocity.X + velocity.Z * velocity.Z;
        float chassisAngle = aiState.Status.Rotation.X + MathF.PI / 2f;
        float fanCenterAngle = horizontalSpeedSq > 0.01f
            ? MathF.Atan2(velocity.Z, velocity.X)
            : chassisAngle;

        long sensorStart = PluginTiming.Now();
        FanRay[] fanRays = fan.Cast(position, fanCenterAngle);
        int fanRayCount = Math.Min(fanRays.Length, state.Policy.LastFanDistances.Length);
        for (int rayIndex = 0; rayIndex < fanRayCount; rayIndex++)
        {
            state.Policy.LastFanDistances[rayIndex] = fanRays[rayIndex].Distance;
        }
        state.Policy.LastFanCenterAngle = fanCenterAngle;
        state.Policy.HasFanCast = true;

        FanRay[] pinRays = pins.Cast(position, chassisAngle);
        state.PreviousPins = pinRays;
        _timing.Add(PluginTiming.Part.Sensors, sensorStart);

        if (_telemetry != null)
        {
            row.PosX = position.X;
            row.PosY = position.Y;
            row.PosZ = position.Z;
            row.SpeedMs = velocity.Length();
            row.CrossTrackErrorM = crossTrackError;
            row.FanRays = fanRays;
            row.FanCenterAngle = fanCenterAngle;
            row.Pins = pinRays;
            row.SentVelocityY = aiState.Status.Velocity.Y;
            row.SentPitch = aiState.Status.Rotation.Y;
            row.SentRoll = aiState.Status.Rotation.Z;
            row.SentStatusFlag = (uint)aiState.Status.StatusFlag;
            long writeStart = PluginTiming.Now();
            _telemetry.Write(in row, in human);
            _timing.Add(PluginTiming.Part.Telemetry, writeStart);
        }
    }

    // The forward speed the car can actually reach this tick (see EnableRearThrustLimit).
    // A car sliding sideways while it yaws loses forward speed at lateralVelocity x yawRate
    // unless thrust makes it up; the rear tires can give what the friction circle leaves
    // after their sideways force, and the front tires' sideways force has a part along the
    // car as well. Forces are those of the state the tick starts in. With EnableRearBrakeLimit
    // the same holds for slowing the car: no more pull back than there is push.
    // rearSpin: how much faster than the road the rear tread is moving with that thrust, for
    // the drivetrain visual only.
    private float LimitByRearThrust(DriftCarState state, float previousOwnSpeed, float rampedOwnSpeed, float dt,
        out float rearSpin)
    {
        float mass = (_car.FrontStaticLoadN + _car.RearStaticLoadN) / 9.80665f;
        float slipAngleSpeed = MathF.Max(previousOwnSpeed, _configuration.MinSlipAngleSpeedMs);
        float steer = state.Policy.LastSteerCommandRadians;
        float frontSlipAngle = MathF.Atan2(state.LateralVelocity + _car.DistanceCgToFrontAxleM * state.YawRate, slipAngleSpeed) - steer;
        float rearSlipAngle = MathF.Atan2(state.LateralVelocity - _car.DistanceCgToRearAxleM * state.YawRate, slipAngleSpeed);
        float fyFront = LateralTireForce(frontSlipAngle, _muFront,
            _car.FrontStaticLoadN, _car.FrontPeakSlipAngleRadians, 1f);
        float fyRear = LateralTireForce(rearSlipAngle, _muRear,
            _car.RearStaticLoadN, _car.RearPeakSlipAngleRadians, state.RearSlidingGrip);

        float rearPeakForceN = _muRear * _car.RearStaticLoadN;
        float thrustMaxN = MathF.Sqrt(MathF.Max(0f, rearPeakForceN * rearPeakForceN - fyRear * fyRear));
        float frontAlongN = -fyFront * MathF.Sin(steer);
        float slideLoss = state.LateralVelocity * state.YawRate;       // dOwnSpeed/dt with no push; negative in a slide
        float thrustNeededN = mass * ((rampedOwnSpeed - previousOwnSpeed) / dt - slideLoss) - frontAlongN;

        float thrustUsedN = float.IsFinite(thrustNeededN) ? Math.Clamp(thrustNeededN, 0f, thrustMaxN) : 0f;
        float rearSidewaysSpeed = MathF.Abs(state.LateralVelocity - _car.DistanceCgToRearAxleM * state.YawRate);
        rearSpin = MathF.Abs(rearSlipAngle) >= _car.RearPeakSlipAngleRadians && MathF.Abs(fyRear) > 1f
            ? rearSidewaysSpeed * thrustUsedN / MathF.Abs(fyRear)
            : 0f;

        if (thrustNeededN > thrustMaxN)
        {
            return MathF.Max(0f, previousOwnSpeed + (slideLoss + (thrustMaxN + frontAlongN) / mass) * dt);
        }
        if (_configuration.EnableRearBrakeLimit && thrustNeededN < -thrustMaxN)
        {
            return MathF.Max(0f, previousOwnSpeed + (slideLoss + (frontAlongN - thrustMaxN) / mass) * dt);
        }
        return rampedOwnSpeed;
    }

    // The forward speed the forces along the car make it this tick, with the policy's speed
    // output as a pedal (see EnableRearFrictionCircle): above zero the share of the thrust the
    // rear tires are driven with, below zero the share of the brakes. Forces are those of the
    // state the tick starts in.
    // brake: the policy's third output (0 for a policy with two), braking WITH the throttle - on
    // the front axle only, at up to the front's share of MaxLongitudinalDecelMs2 and never more
    // than the front tires can take; what goes into braking is not there for steering (the front
    // tires' friction circle). fast_sim.step_post does the same.
    // rearCapacityN / frontCapacityN: the sideways force each axle has left after what goes along
    // the car; frontCapacityN is -1 (not in use) while the front is not braking.
    // rearSpin: how much faster than the road the rear tread is moving, for the drivetrain
    // visual only.
    private float DriveByPedal(DriftCarState state, float previousOwnSpeed, float pedal, float brake, float maxSpeed, float dt,
        out float rearCapacityN, out float frontCapacityN, out float rearSpin)
    {
        float mass = (_car.FrontStaticLoadN + _car.RearStaticLoadN) / 9.80665f;
        float slipAngleSpeed = MathF.Max(previousOwnSpeed, _configuration.MinSlipAngleSpeedMs);
        float steer = state.Policy.LastSteerCommandRadians;
        float frontSlipAngle = MathF.Atan2(state.LateralVelocity + _car.DistanceCgToFrontAxleM * state.YawRate, slipAngleSpeed) - steer;
        float rearSlipAngle = MathF.Atan2(state.LateralVelocity - _car.DistanceCgToRearAxleM * state.YawRate, slipAngleSpeed);
        float fyFront = LateralTireForce(frontSlipAngle, _muFront,
            _car.FrontStaticLoadN, _car.FrontPeakSlipAngleRadians, 1f);

        float rearPeakForceN = _muRear * _car.RearStaticLoadN;
        float frontAlongN = -fyFront * MathF.Sin(steer);
        float slideLoss = state.LateralVelocity * state.YawRate;       // dOwnSpeed/dt with no force along the car

        float rearShare = _car.RearStaticLoadN / (_car.FrontStaticLoadN + _car.RearStaticLoadN);
        float frontPeakForceN = _muFront * _car.FrontStaticLoadN;
        brake = Math.Clamp(brake, 0f, 1f);
        float leftFootBrakeN = -brake * MathF.Min((1f - rearShare) * _configuration.MaxLongitudinalDecelMs2 * mass, frontPeakForceN);

        float rearAlongN;
        float frontBrakeN;
        if (pedal >= 0f)
        {
            rearAlongN = pedal * rearPeakForceN * MathF.Sqrt(MathF.Max(0f, 1f - state.RearSlidingGrip * state.RearSlidingGrip));
            // no more than the car's engine can put down at this speed (CarSpec.EngineForce; no
            // limit for a car without a curve): a car without the power keeps its rear grip
            rearAlongN = MathF.Min(rearAlongN, pedal * _car.EngineForce(previousOwnSpeed));
            // No thrust past the car's speed limit.
            float toLimitN = mass * ((maxSpeed - previousOwnSpeed) / dt - slideLoss) - frontAlongN - leftFootBrakeN;
            rearAlongN = MathF.Max(0f, MathF.Min(rearAlongN, toLimitN));
            frontBrakeN = leftFootBrakeN;
        }
        else
        {
            float brakeForceN = -pedal * _configuration.MaxLongitudinalDecelMs2 * mass;
            rearAlongN = -MathF.Min(rearShare * brakeForceN, rearPeakForceN);
            frontBrakeN = -(1f - rearShare) * brakeForceN + leftFootBrakeN;
        }
        frontCapacityN = -1f;
        if (frontBrakeN < 0f)       // braking, by the pedal or the brake action
        {
            frontBrakeN = MathF.Max(frontBrakeN, -frontPeakForceN);
            frontCapacityN = MathF.Sqrt(MathF.Max(0f, frontPeakForceN * frontPeakForceN - frontBrakeN * frontBrakeN));
        }
        rearCapacityN = MathF.Sqrt(MathF.Max(0f, rearPeakForceN * rearPeakForceN - rearAlongN * rearAlongN));

        // A rear axle at the edge of its circle slips lengthwise by (thrust / sideways force)
        // times its sideways slip.
        float fyRearN = MathF.Min(rearPeakForceN * (MathF.Abs(rearSlipAngle) / _car.RearPeakSlipAngleRadians), rearCapacityN);
        float rearSidewaysSpeed = MathF.Abs(state.LateralVelocity - _car.DistanceCgToRearAxleM * state.YawRate);
        rearSpin = rearAlongN > 0f && fyRearN >= rearCapacityN && fyRearN > 1f
            ? rearSidewaysSpeed * rearAlongN / fyRearN
            : 0f;

        return MathF.Max(0f, previousOwnSpeed + (slideLoss + (rearAlongN + frontBrakeN + frontAlongN) / mass) * dt);
    }

    // A driver below full skill (drift_env SKILL_*, as the policy was trained), by d = 1 - skill:
    // a slower reaction - the action applied is the one asked SkillDelayTicks x d ticks earlier
    // (rounded) - then smooth noise on the steering and pedal.
    private (float Steer, float Speed, float Brake) HandicapBySkill(DriftCarState state, bool firstTick,
        float steer, float speed, float brake)
    {
        const float SteerNoise = 0.10f, PedalNoise = 0.20f, NoiseKeep = 0.9f, SkillDelayTicks = 1.5f;
        float lack = 1f - Math.Clamp(_skill, 0f, 1f);
        float fresh = MathF.Sqrt(1f - NoiseKeep * NoiseKeep);
        // unit-variance uniform noise, as the trainer's
        state.SkillNoiseSteer = NoiseKeep * state.SkillNoiseSteer + fresh * ((float)_skillRandom.NextDouble() - 0.5f) * 3.4641016f;
        state.SkillNoisePedal = NoiseKeep * state.SkillNoisePedal + fresh * ((float)_skillRandom.NextDouble() - 0.5f) * 3.4641016f;
        if (firstTick)
            state.SkillAskedTicks = 0;
        int delay = (int)(SkillDelayTicks * lack + 0.5f);
        (float s, float v, float b) asked = (steer, speed, brake);
        if (delay >= 1 && state.SkillAskedTicks >= delay)
            (steer, speed, brake) = delay == 1 ? state.SkillAsked1 : state.SkillAsked2;
        state.SkillAsked2 = state.SkillAsked1;
        state.SkillAsked1 = asked;
        state.SkillAskedTicks++;
        steer = Math.Clamp(steer + SteerNoise * lack * state.SkillNoiseSteer, -1f, 1f);
        speed = Math.Clamp(speed + PedalNoise * lack * state.SkillNoisePedal, -1f, 1f);
        return (steer, speed, brake);
    }

    // Each axle's peak friction this tick (fast_sim.axle_friction), for the car being driven.
    private (float Front, float Rear) AxleFriction(float ownSpeed, float yawRate)
        => _car.AxleFriction(ownSpeed, yawRate, _configuration.EnableTireLoadSensitivity);

    // Sideways force of an axle that rises with slip angle as LateralTireForce does and stops
    // at capacityN, what the friction circle leaves it. Opposes the slip.
    private static float LateralForceWithin(float slipAngle, float peakForceN, float peakSlipAngleRadians, float capacityN)
    {
        float forceMagnitude = MathF.Min(peakForceN * (MathF.Abs(slipAngle) / peakSlipAngleRadians), capacityN);
        return slipAngle > 0f ? -forceMagnitude : slipAngle < 0f ? forceMagnitude : 0f;
    }

    // Lateral force of one axle: linear in slip angle up to the peak, then falling linearly to
    // slidingGrip x the peak at twice the peak slip angle, and constant beyond. Opposes the slip.
    // slidingGrip = 1 keeps the peak force however far the tire slides.
    private static float LateralTireForce(float slipAngle, float peakFriction, float staticLoadN, float peakSlipAngleRadians,
        float slidingGrip)
    {
        float peakForceN = peakFriction * staticLoadN;
        float slipMagnitude = MathF.Abs(slipAngle);
        float forceMagnitude;
        if (slipMagnitude < peakSlipAngleRadians)
        {
            forceMagnitude = peakForceN * (slipMagnitude / peakSlipAngleRadians);
        }
        else if (slidingGrip >= 1f)
        {
            forceMagnitude = peakForceN;
        }
        else
        {
            float pastPeak = (slipMagnitude - peakSlipAngleRadians) / peakSlipAngleRadians;
            forceMagnitude = pastPeak >= 1f
                ? peakForceN * slidingGrip
                : peakForceN * (1f - (1f - slidingGrip) * pastPeak);
        }
        return -MathF.Sign(slipAngle) * forceMagnitude;
    }

    // The segment of the racing line the car is abeam of: the nearest point with whichever
    // neighbour the car projects onto more closely, and how far along it (0..1). Uses the
    // spline's own links, so it does not assume points are stored in order or evenly spaced.
    private (int pointA, int pointB, float t) FindInterpolationSegment(Vector3 ourPosition, int nearestPointId)
    {
        if (nearestPointId < 0 || nearestPointId >= _spline.Points.Length)
            return (nearestPointId, nearestPointId, 0f);

        ref readonly var nearest = ref _spline.Points[nearestPointId];
        float bestDistSq = float.MaxValue;
        int bestA = nearestPointId;
        int bestB = nearestPointId;
        float bestT = 0f;

        if (nearest.NextId >= 0 && nearest.NextId < _spline.Points.Length)
        {
            Vector3 segment = _spline.Points[nearest.NextId].Position - nearest.Position;
            float segLenSq = segment.LengthSquared();
            if (segLenSq > 0.0001f)
            {
                float t = Vector3.Dot(ourPosition - nearest.Position, segment) / segLenSq;
                t = Math.Clamp(t, 0f, 1f);
                Vector3 projected = nearest.Position + segment * t;
                float distSq = Vector3.DistanceSquared(ourPosition, projected);
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    bestA = nearestPointId;
                    bestB = nearest.NextId;
                    bestT = t;
                }
            }
        }

        if (nearest.PreviousId >= 0 && nearest.PreviousId < _spline.Points.Length)
        {
            Vector3 prevPos = _spline.Points[nearest.PreviousId].Position;
            Vector3 segment = nearest.Position - prevPos;
            float segLenSq = segment.LengthSquared();
            if (segLenSq > 0.0001f)
            {
                float t = Vector3.Dot(ourPosition - prevPos, segment) / segLenSq;
                t = Math.Clamp(t, 0f, 1f);
                Vector3 projected = prevPos + segment * t;
                float distSq = Vector3.DistanceSquared(ourPosition, projected);
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    bestA = nearest.PreviousId;
                    bestB = nearestPointId;
                    bestT = t;
                }
            }
        }

        return (bestA, bestB, bestT);
    }

    // The pace error input for the next tick, as drift_env._step_one keeps it (counters 8 and 9):
    // the car's progress along the line this tick (clamped to -2 .. 5 m, as the reward does) over
    // the tick's length, and the speed asked of it, pace x the reference speed at the point it
    // started the tick at, each averaged over about averageSeconds; then how far over the asked
    // speed it is (0 while nothing is asked). The first tick starts both averages afresh.
    private void UpdatePaceOver(DriftCarState state, Vector3 position, int askedPointId, float maxSpeedMs, float dt,
        float averageSeconds)
    {
        DriftPolicyCarState p = state.Policy;
        float[] arcs = _lapArc!;
        var (rawPointId, _) = _spline.WorldToSpline(position);
        float arc = float.NaN;
        if (rawPointId >= 0 && rawPointId < arcs.Length)
        {
            var (a, b, t) = FindInterpolationSegment(position, rawPointId);
            arc = arcs[a] + t * Vector3.Distance(_spline.Points[a].Position, _spline.Points[b].Position);
        }
        float progress = 0f;
        if (!float.IsNaN(p.PaceArcM) && !float.IsNaN(arc))
        {
            progress = arc - p.PaceArcM;
            if (progress > _lapLength / 2f) progress -= _lapLength;
            else if (progress < -_lapLength / 2f) progress += _lapLength;
        }
        p.PaceArcM = arc;
        if (askedPointId < 0)
        {
            return;
        }

        float speed = Math.Clamp(progress, -2f, 5f) / MathF.Max(dt, 1e-3f);
        // never more than the car's speed limit (drift_env._step_one asks the same): a long
        // straight's reference can run far past what any bot is allowed to reach
        float askedKmh = state.Pace * MathF.Min(_paceReferenceKmh![askedPointId] * _car.PaceScale, maxSpeedMs * 3.6f);
        if (p.PaceTicks == 0)
        {
            p.PaceSpeedAverageMs = speed;
            p.PaceAskedAverageMs = askedKmh / 3.6f;
        }
        else
        {
            float k = MathF.Min(1f, dt / MathF.Max(averageSeconds, 1e-3f));
            p.PaceSpeedAverageMs += (speed - p.PaceSpeedAverageMs) * k;
            p.PaceAskedAverageMs += (askedKmh / 3.6f - p.PaceAskedAverageMs) * k;
        }
        p.PaceTicks++;
        p.PaceOver = askedKmh > 0f && p.PaceAskedAverageMs > 0.5f ? p.PaceSpeedAverageMs / p.PaceAskedAverageMs - 1f : 0f;
    }

    // Wall contact: pins on the collision triangles (the walls the game collides with), aimed
    // around the direction of travel, each push the car back along their own direction by
    // however far a Wall triangle is inside the contact distance; the velocity into the wall is
    // then removed. Returns the distance pushed.
    private float PushOutOfWalls(ProximityPins contactPins, AiState aiState, ref Vector3 position, ref Vector3 velocity)
    {
        float speedSq = velocity.X * velocity.X + velocity.Z * velocity.Z;
        float baseAngle = speedSq > 0.01f
            ? MathF.Atan2(velocity.Z, velocity.X)
            : aiState.Status.Rotation.X + MathF.PI / 2f;

        FanRay[] rays = contactPins.Cast(position, baseAngle);
        Vector3 pushOut = Vector3.Zero;
        for (int pinIndex = 0; pinIndex < rays.Length; pinIndex++)
        {
            if (rays[pinIndex].Surface != SurfaceType.Wall
                || rays[pinIndex].Distance >= _configuration.CollisionContactDistanceMeters)
            {
                continue;
            }

            float penetration = (_configuration.CollisionContactDistanceMeters + _configuration.CollisionPushOutMarginMeters)
                - rays[pinIndex].Distance;
            float pinAngle = baseAngle + ProximityPins.OffsetsRadians[pinIndex];
            pushOut += new Vector3(-MathF.Cos(pinAngle), 0f, -MathF.Sin(pinAngle)) * penetration;
        }

        if (pushOut.LengthSquared() <= 0.0001f)
        {
            return 0f;
        }

        position += pushOut;
        Vector3 contactNormal = Vector3.Normalize(pushOut);
        float velocityAlongNormal = Vector3.Dot(velocity, contactNormal);
        if (velocityAlongNormal < 0f)
        {
            velocity -= contactNormal * velocityAlongNormal;
        }
        return pushOut.Length();
    }

    // Wall contact with CarOutlineContact: a ray from the car's centre to each point of its
    // outline (CarOutline, turned with its heading); a Wall triangle nearer than the point is
    // inside the car, and pushes it back along that ray by how far it is in plus the margin. The
    // velocity into the wall is then removed, as PushOutOfWalls does (fast_sim's collision
    // response, EXT_CAR_OUTLINE_CONTACT). Returns the distance pushed.
    private float PushOutlineOutOfWalls(TriangleGrid collision, float heading, ref Vector3 position, ref Vector3 velocity)
    {
        var outline = new CarOutline(_car, _configuration.CarOutlineOverhangMeters, _configuration.CarOutlineSideMarginMeters);
        var origin = new Vector3(position.X, position.Y + _configuration.ProximityPinHeightOffsetMeters, position.Z);
        Vector3 pushOut = Vector3.Zero;
        for (int i = 0; i < CarOutline.PointCount; i++)
        {
            (float forward, float left) = outline.Point(i);
            float reach = MathF.Sqrt(forward * forward + left * left);
            float angle = heading + MathF.Atan2(left, forward);
            var direction = new Vector3(MathF.Cos(angle), 0f, MathF.Sin(angle));
            if (!collision.Raycast(origin, direction, reach, out RaycastHit hit, wallsOnly: true)
                || hit.Surface != SurfaceType.Wall || hit.Distance >= reach)
            {
                continue;
            }
            pushOut -= direction * (reach + _configuration.CollisionPushOutMarginMeters - hit.Distance);
        }

        if (pushOut.LengthSquared() <= 0.0001f)
        {
            return 0f;
        }

        position += pushOut;
        Vector3 contactNormal = Vector3.Normalize(pushOut);
        float velocityAlongNormal = Vector3.Dot(velocity, contactNormal);
        if (velocityAlongNormal < 0f)
        {
            velocity -= contactNormal * velocityAlongNormal;
        }
        return pushOut.Length();
    }

    // Height of the ground straight below the car: the first collision triangle that is not a
    // Wall. A barrier over the road is stepped through (1 mm past each hit, so the same triangle
    // is not found twice).
    private bool TryFindGround(TriangleGrid collision, Vector3 position, out float groundY)
    {
        float originY = position.Y + _configuration.GroundRayStartAboveMeters;
        float remaining = _configuration.GroundRayMaxDistanceMeters;
        for (int layer = 0; layer < GroundRayMaxWallLayers; layer++)
        {
            if (!collision.Raycast(new Vector3(position.X, originY, position.Z), -Vector3.UnitY, remaining, out RaycastHit hit))
            {
                break;
            }

            if (hit.Surface != SurfaceType.Wall)
            {
                groundY = originY - hit.Distance;
                return true;
            }

            originY -= hit.Distance + 0.001f;
            remaining -= hit.Distance + 0.001f;
            if (remaining <= 0f)
            {
                break;
            }
        }

        groundY = 0f;
        return false;
    }

    // The ground height under the car as the least-squares straight line through the ground at
    // RideHeightSamples points spread RideHeightSpanMeters either side of it along `along`, on
    // three lanes (the car's centre line and half a track to either side): exact on a steady
    // slope, and bumps shorter than the span average out. Points with no ground are left out;
    // with fewer than half of them, the ground straight below (centreY) is used.
    private float RideGroundHeight(TriangleGrid collision, Vector3 position, Vector3 along, float centreY)
    {
        float span = _configuration.RideHeightSpanMeters;
        if (span <= 0f)
        {
            return centreY;
        }

        var across = new Vector3(-along.Z, 0f, along.X);
        int n = 0;
        float sumS = 0f, sumY = 0f, sumSS = 0f, sumSY = 0f;
        for (int lane = -1; lane <= 1; lane++)
        {
            Vector3 laneStart = position + across * (lane * AttitudeHalfTrackM);
            for (int i = 0; i < RideHeightSamples; i++)
            {
                float s = span * (2f * i / (RideHeightSamples - 1) - 1f);
                if (TryFindGround(collision, laneStart + along * s, out float y))
                {
                    n++;
                    sumS += s;
                    sumY += y;
                    sumSS += s * s;
                    sumSY += s * y;
                }
            }
        }

        if (n < 3 * RideHeightSamples / 2)
        {
            return centreY;
        }

        float meanS = sumS / n;
        float meanY = sumY / n;
        float varS = sumSS / n - meanS * meanS;
        float slope = varS > 1e-6f ? (sumSY / n - meanS * meanY) / varS : 0f;
        return meanY - slope * meanS;
    }

    // Pitch and roll of the ground under the car, in the sense clients are sent them (as
    // measured from a human's car: HISTORY.md), from its height under the two axles and half
    // a track to either side. False where a ray finds no ground or the heights are a step apart.
    private bool TryFindGroundAttitude(TriangleGrid collision, Vector3 position, float heading, out float pitch, out float roll)
    {
        pitch = 0f;
        roll = 0f;
        var forward = new Vector3(MathF.Cos(heading), 0f, MathF.Sin(heading));
        var across = new Vector3(-forward.Z, 0f, forward.X);
        float toFront = _car.DistanceCgToFrontAxleM;
        float toRear = _car.DistanceCgToRearAxleM;
        if (!TryFindGround(collision, position + forward * toFront, out float frontY)
            || !TryFindGround(collision, position - forward * toRear, out float rearY)
            || !TryFindGround(collision, position + across * AttitudeHalfTrackM, out float nearY)
            || !TryFindGround(collision, position - across * AttitudeHalfTrackM, out float farY)
            || MathF.Abs(frontY - rearY) > AttitudeMaxStepM || MathF.Abs(nearY - farY) > AttitudeMaxStepM)
        {
            return false;
        }

        pitch = MathF.Atan2(frontY - rearY, toFront + toRear);
        roll = MathF.Atan2(nearY - farY, 2f * AttitudeHalfTrackM);
        return true;
    }

    // Ride height where the ground ray found nothing: the racing line's own height abeam the
    // car, blended toward the nearest road-mesh point the further the car is from the line.
    private float FallbackHeight(Vector3 position, int verticalPointId, float crossTrackError)
    {
        var (heightA, heightB, heightT) = FindInterpolationSegment(position, verticalPointId);
        float lineHeight = heightA >= 0 && heightA < _spline.Points.Length && heightB >= 0 && heightB < _spline.Points.Length
            ? _spline.Points[heightA].Position.Y + (_spline.Points[heightB].Position.Y - _spline.Points[heightA].Position.Y) * heightT
            : position.Y;

        if (_roadMeshTree == null)
        {
            return lineHeight;
        }

        var (meshIndex, meshDistSq) = _roadMeshTree.FindNearest(position);
        if (meshIndex < 0 || MathF.Sqrt(meshDistSq) >= _configuration.MaxTrustedMeshDistanceM)
        {
            return lineHeight;
        }

        float meshHeight = _roadMeshPoints[meshIndex].Y + _configuration.RoadMeshHeightOffsetM;
        float blend = Math.Clamp(
            (MathF.Abs(crossTrackError) - _configuration.CrossTrackBlendStartM)
                / (_configuration.CrossTrackBlendEndM - _configuration.CrossTrackBlendStartM),
            0f, 1f);
        return lineHeight + blend * (meshHeight - lineHeight);
    }

    // Overwrites the two steering bytes the stock AI left at 127 ("straight") with the angle
    // the vehicle model used, slewed so a one-tick lock-to-lock change does not show as a snap.
    // Cosmetic: nothing reads these back.
    private void WriteSteeringVisual(AiState aiState, DriftCarState state, float steerCommand, float dt, ref DriftTelemetryRow row)
    {
        row.VisualWheelAngleByte = 127;
        row.VisualSteerAngleByte = 127;
        if (!_configuration.EnableSteeringVisual)
        {
            return;
        }

        float steerDeg = steerCommand * (180f / MathF.PI);
        if (!float.IsFinite(steerDeg))
        {
            steerDeg = 0f;
        }

        if (_configuration.SteeringVisualMaxRateDegS > 0f && state.VisualSteerDeg.HasValue)
        {
            float maxStepDeg = _configuration.SteeringVisualMaxRateDegS * dt;
            steerDeg = state.VisualSteerDeg.Value
                + Math.Clamp(steerDeg - state.VisualSteerDeg.Value, -maxStepDeg, maxStepDeg);
        }
        state.VisualSteerDeg = steerDeg;

        row.VisualWheelAngleByte = (byte)Math.Clamp(MathF.Round(127f
            + _configuration.SteeringVisualSign * _configuration.SteeringVisualWheelByteUnitsPerDegree * steerDeg), 0f, 255f);
        // The steering wheel's byte covers its whole range, so it reaches its limit when the
        // road wheels reach full lock.
        float lockDeg = MathF.Max(_car.MaxSteerAngleRadians * (180f / MathF.PI), 1f);
        row.VisualSteerAngleByte = (byte)Math.Clamp(MathF.Round(127f
            + _configuration.SteeringVisualSign * 127f * steerDeg / lockDeg), 0f, 254f);

        aiState.Status.WheelAngle = row.VisualWheelAngleByte;
        aiState.Status.SteerAngle = row.VisualSteerAngleByte;
    }

    // Overwrites the wheel speeds, engine RPM, gear and throttle the stock AI wrote for its own
    // spline follower with ones that belong to this car (see EnableDrivetrainVisual).
    // Cosmetic: nothing reads these back.
    private void WriteDrivetrainVisual(AiState aiState, DriftCarState state, float forwardSpeed, float frontWheelSpeed,
        float rearSpin, float longitudinalAccel, float pedal, float dt, ref DriftTelemetryRow row)
    {
        var status = aiState.Status;
        List<float> gearRatios = _configuration.DrivetrainVisualGearRatios;
        if (!_configuration.EnableDrivetrainVisual || gearRatios is not { Count: > 0 }
            || !float.IsFinite(forwardSpeed) || !float.IsFinite(frontWheelSpeed) || !float.IsFinite(rearSpin))
        {
            // Left as the stock AI wrote them; logged so the telemetry shows what was sent.
            row.VisualFrontTyreByte = status.TyreAngularSpeed[0];
            row.VisualRearTyreByte = status.TyreAngularSpeed[2];
            row.VisualEngineRpm = status.EngineRpm;
            row.VisualGear = status.Gear - 1;
            row.VisualGasByte = status.Gas;
            return;
        }

        float tyreDiameter = MathF.Max(_configuration.DrivetrainVisualTyreDiameterMeters, 0.2f);
        float engineRpmPerWheelSpeed = 60f / (MathF.PI * tyreDiameter) * _configuration.DrivetrainVisualFinalDrive;

        // The gear is for the road speed, not for the wheelspin: up while the road speed alone
        // would have this gear over the upshift speed, down while it would have the next gear
        // down under the downshift speed. The gap between the two keeps it from hunting.
        int gear = Math.Clamp(state.VisualGear, 1, gearRatios.Count);
        float roadRpmPerRatio = forwardSpeed * engineRpmPerWheelSpeed;
        for (int shift = 0; shift < gearRatios.Count; shift++)
        {
            if (gear < gearRatios.Count && roadRpmPerRatio * gearRatios[gear - 1] > _configuration.DrivetrainVisualUpshiftRpm)
            {
                gear++;
            }
            else if (gear > 1 && roadRpmPerRatio * gearRatios[gear - 2] < _configuration.DrivetrainVisualDownshiftRpm)
            {
                gear--;
            }
            else
            {
                break;
            }
        }
        state.VisualGear = gear;

        // The wheels cannot spin faster than the rev limiter lets them in this gear, and they
        // follow a change in wheelspin with a short lag.
        float revLimitRpm = MathF.Max(_configuration.DrivetrainVisualIdleRpm, _configuration.DrivetrainVisualRevLimitRpm);
        float rpmPerWheelSpeed = MathF.Max(engineRpmPerWheelSpeed * gearRatios[gear - 1], 0.001f);
        float rearWheelSpeed = MathF.Max(forwardSpeed, MathF.Min(forwardSpeed + rearSpin, revLimitRpm / rpmPerWheelSpeed));
        float lag = _configuration.DrivetrainVisualWheelSpeedLagSeconds;
        if (lag > 0f && state.VisualRearWheelSpeed.HasValue)
        {
            rearWheelSpeed = state.VisualRearWheelSpeed.Value
                + (1f - MathF.Exp(-dt / lag)) * (rearWheelSpeed - state.VisualRearWheelSpeed.Value);
        }
        state.VisualRearWheelSpeed = rearWheelSpeed;
        float engineRpm = Math.Clamp(rearWheelSpeed * rpmPerWheelSpeed, _configuration.DrivetrainVisualIdleRpm, revLimitRpm);

        // Throttle: in proportion to wheelspin (full once the rear tread is 95% faster than
        // the road), at least 80% while gaining speed, none while slowing down without
        // wheelspin or stopped. Clients only receive it with CSP's own car updates.
        float spinShare = rearSpin / MathF.Max(forwardSpeed, 1f);
        float throttle = Math.Clamp(1.05f * spinShare, 0f, 1f);
        if (longitudinalAccel > 0.5f)
        {
            throttle = MathF.Max(throttle, 0.8f);
        }
        else if ((longitudinalAccel < -0.5f && spinShare < 0.05f) || forwardSpeed < 0.5f)
        {
            throttle = 0f;
        }
        if (float.IsFinite(pedal))
        {
            throttle = Math.Clamp(pedal, 0f, 1f);
        }

        byte frontTyreByte = EncodeTyreAngularSpeed(frontWheelSpeed, tyreDiameter);
        byte rearTyreByte = EncodeTyreAngularSpeed(rearWheelSpeed, tyreDiameter);
        status.TyreAngularSpeed[0] = frontTyreByte;
        status.TyreAngularSpeed[1] = frontTyreByte;
        status.TyreAngularSpeed[2] = rearTyreByte;
        status.TyreAngularSpeed[3] = rearTyreByte;
        status.EngineRpm = (ushort)engineRpm;
        status.Gear = (byte)(gear + 1);        // 0 = reverse, 1 = neutral, 2 = first: the stock AI sends 2
        status.Gas = (byte)MathF.Round(throttle * 255f);

        row.VisualFrontTyreByte = frontTyreByte;
        row.VisualRearTyreByte = rearTyreByte;
        row.VisualRearWheelSpeedMs = rearWheelSpeed;
        row.VisualEngineRpm = status.EngineRpm;
        row.VisualGear = gear;
        row.VisualGasByte = status.Gas;
    }

    // The stock AI sets the brake lights, hazards, indicators and horn from its own spline
    // follower, which is not this car. Brake lights instead while the policy is asking for
    // less speed than the car has, held on briefly so a one-tick dab does not flicker; the
    // other three off. Headlights and wipers are left as the stock AI set them. And the skid
    // bits of an axle whose tires are leaving rubber.
    private static void WriteStatusFlags(AiState aiState, DriftCarState state, bool slowing, bool frontSkidding,
        bool rearSkidding, float dt)
    {
        state.BrakeLightSeconds = slowing ? BrakeLightHoldSeconds : MathF.Max(0f, state.BrakeLightSeconds - dt);
        CarStatusFlags flags = aiState.Status.StatusFlag & ~(CarStatusFlags.BrakeLightsOn | CarStatusFlags.HazardsOn
            | CarStatusFlags.Horn | CarStatusFlags.IndicateLeft | CarStatusFlags.IndicateRight);
        if (state.BrakeLightSeconds > 0f)
        {
            flags |= CarStatusFlags.BrakeLightsOn;
        }
        flags |= (frontSkidding ? FrontTiresSkidding : 0) | (rearSkidding ? RearTiresSkidding : 0);
        aiState.Status.StatusFlag = flags;
    }

    // The stock AI's encoding (AiState.Update) of the wheel's angular speed in rad/s. (The
    // stock AI works that out as turns a second x 6; a human's game sends the real figure,
    // x 2 pi, and so does this.)
    private static byte EncodeTyreAngularSpeed(float treadSpeed, float tyreDiameter)
    {
        float angularSpeed = treadSpeed / (0.5f * tyreDiameter);
        if (!float.IsFinite(angularSpeed))
        {
            return 100;
        }
        return (byte)(Math.Clamp(MathF.Round(MathF.Log10(MathF.Abs(angularSpeed) + 1f) * 20f) * MathF.Sign(angularSpeed), -100f, 154f) + 100f);
    }

    // TrainSize > 1: the AI entries in entry order make the trains, TrainSize to a train, the same
    // way every tick whichever of them the server has out. _driven keeps the leads (and any car
    // whose train has no lead out, or an entry's extra overbooked states), _followers the rest,
    // each with its lead. A car that becomes a follower loses its own driving state and its
    // history as a lead, so it starts afresh if it ever leads again.
    private void FormTrains()
    {
        _followers.Clear();
        int size = _configuration.TrainSize;
        if (size <= 1)
        {
            return;
        }
        var leadOf = new Dictionary<int, AiState>();
        foreach (var (state, slot) in _trainSlots)
        {
            if (slot % size == 0)
            {
                leadOf[slot / size] = state;
            }
        }
        var all = _driven.ToArray();
        _driven.Clear();
        foreach (var (sessionId, state) in all)
        {
            if (_trainSlots.TryGetValue(state, out int slot) && slot % size != 0
                && leadOf.TryGetValue(slot / size, out AiState? lead) && ReadyToFollow(state, lead, slot % size))
            {
                _followers.Add((sessionId, state, lead, slot % size));
                _cars.Remove(state);
                _trainHistories.Remove(state);
            }
            else
            {
                _driven.Add((sessionId, state));
            }
        }
        // in train order, each train's places in order (Follow needs the car ahead done first)
        _followers.Sort((a, b) => a.Lead == b.Lead
            ? a.Place.CompareTo(b.Place)
            : _trainSlots[a.Lead].CompareTo(_trainSlots[b.Lead]));
    }

    // The lead's history, if it is this spawn's and the plugin is driving the lead.
    private TrainHistory? CurrentHistory(AiState lead)
    {
        return _trainHistories.TryGetValue(lead, out TrainHistory? history) && history.Count > 0
            && history.SpawnCounter == lead.SpawnCounter
            && _cars.TryGetValue(lead, out DriftCarState? state) && history.Owner == state
            ? history
            : null;
    }

    // A car joins its lead's train once the lead's history reaches back further than its place
    // will ever need (its whole gap at the longest, and a second for the minimum gap); until then
    // it drives alone. One already following stays (it falls back to the server's AI meanwhile).
    private bool ReadyToFollow(AiState car, AiState lead, int place)
    {
        if (!_cars.ContainsKey(car))
        {
            return true;
        }
        TrainHistory? history = CurrentHistory(lead);
        long need = (long)(1000f * place * _configuration.TandemGapSeconds * (1f + MathF.Abs(_configuration.TandemGapVariation))) + 1000;
        return history != null && _sessionManager.ServerTimeMilliseconds - history.OldestMs >= need;
    }

    // What a lead sent this tick, for its followers. Started afresh whenever the server places the
    // car somewhere new or the plugin starts driving it afresh, so followers never replay a jump.
    private void RecordLead(AiState lead, long nowMs)
    {
        if (!_cars.TryGetValue(lead, out DriftCarState? state) || state.SpawnCounter != lead.SpawnCounter)
        {
            return;
        }
        if (!_trainHistories.TryGetValue(lead, out TrainHistory? history))
        {
            history = new TrainHistory();
            _trainHistories[lead] = history;
        }
        if (history.SpawnCounter != lead.SpawnCounter || history.Owner != state)
        {
            history.Clear();
            history.SpawnCounter = lead.SpawnCounter;
            history.Owner = state;
        }
        history.Add(TrainSample.From(lead.Status, nowMs));
    }

    // A follower: the lead's state from its own delay ago - always further back than the car in
    // front, and never within TandemMinGapMeters of it - a little to one side, on the ground. While
    // that cannot be met (the lead's history too short) the server's own AI keeps the car, and the
    // ones behind it in the train too.
    private void Follow(byte sessionId, AiState follower, AiState lead, int place, long nowMs, float dt,
        TriangleGrid collision, in HumanCarSample human)
    {
        float delayLogged = -1f;
        float offsetLogged = 0f;
        try
        {
            FollowCore(sessionId, follower, lead, place, nowMs, collision, ref delayLogged, ref offsetLogged);
        }
        finally
        {
            DriftTelemetry? telemetry = _telemetry;
            if (telemetry != null)
            {
                var status = follower.Status;
                float heading = status.Rotation.X + MathF.PI / 2f;
                var row = new DriftTelemetryRow
                {
                    Tick = _tickCounter,
                    ServerTimeMs = nowMs,
                    SessionId = sessionId,
                    SpawnCounter = follower.SpawnCounter,
                    DtSeconds = dt,
                    PosX = status.Position.X,
                    PosY = status.Position.Y,
                    PosZ = status.Position.Z,
                    SpeedMs = status.Velocity.Length(),
                    MaxSpeedMs = follower.MaxSpeed,
                    Heading = MathF.Atan2(MathF.Sin(heading), MathF.Cos(heading)),
                    NearestPointId = -1,
                    PolicyObservation = "",
                    GapAheadM = -1f,
                    FollowLeadId = lead.EntryCar.SessionId,
                    FollowPlace = place,
                    FollowDelayMs = delayLogged,
                    FollowGapM = delayLogged >= 0f ? Vector3.Distance(status.Position, _aheadPositionBefore) : -1f,
                    FollowOffsetM = offsetLogged,
                };
                telemetry.Write(in row, in human);
            }
        }
    }

    private void FollowCore(byte sessionId, AiState follower, AiState lead, int place, long nowMs, TriangleGrid collision,
        ref float delayLogged, ref float offsetLogged)
    {
        if (_aheadLead != lead)
        {
            _aheadLead = lead;
            _aheadDelayMs = 0;
            _aheadPosition = lead.Status.Position;
        }
        TrainHistory? history = _aheadDelayMs < 0 ? null : CurrentHistory(lead);
        if (history == null)
        {
            _aheadDelayMs = -1;
            _followerLast.Remove(follower);
            return;
        }
        float timeS = nowMs / 1000f;
        float gapS = _configuration.TandemGapSeconds
            * (1f + _configuration.TandemGapVariation * TandemFollower.Wander(sessionId, timeS, 7.3f, 12.9f));
        long nominal = _aheadDelayMs + (long)(1000f * MathF.Max(gapS, 0.05f));
        Vector3 aheadPosition = _aheadPosition;
        _aheadPositionBefore = aheadPosition;
        float minGap2 = _configuration.TandemMinGapMeters * _configuration.TandemMinGapMeters;

        // the shortest delay from nominal on that keeps the gap: steps of 50 ms, then halved down to ~3 ms
        long longest = nowMs - history.OldestMs;
        long delay = -1;
        for (long d = nominal; d <= longest; d += 50)
        {
            if (history.TryAt(nowMs - d, out _, out Vector3 p) && Vector3.DistanceSquared(p, aheadPosition) >= minGap2)
            {
                long lo = Math.Max(nominal, d - 50);
                long hi = d;
                while (hi - lo > 3)
                {
                    long mid = (lo + hi) / 2;
                    if (history.TryAt(nowMs - mid, out _, out Vector3 q) && Vector3.DistanceSquared(q, aheadPosition) >= minGap2)
                    {
                        hi = mid;
                    }
                    else
                    {
                        lo = mid;
                    }
                }
                delay = hi;
                break;
            }
        }
        if (delay < 0 || !history.TryAt(nowMs - delay, out TrainSample sample, out Vector3 position))
        {
            _aheadDelayMs = -1;
            _followerLast.Remove(follower);
            return;
        }
        _aheadDelayMs = delay;
        delayLogged = delay;

        int side = (sessionId % 2 == 0) ? 1 : -1;
        float offset = _configuration.TandemLateralOffsetMeters
            * (0.6f + 0.4f * TandemFollower.Wander(sessionId + 17, timeS, 9.1f, 15.7f)) * side;
        position += TandemFollower.Side(sample) * offset;
        offsetLogged = offset;
        if (offset != 0f && TryFindGround(collision, position, out float groundY))
        {
            position.Y = groundY + _configuration.GroundRayHeightOffsetM;
        }
        // held back for the gap, the follower moves slower than the lead did then: send how it moves
        if (delay > nominal + 5 && _followerLast.TryGetValue(follower, out var last) && nowMs > last.TimeMs
            && nowMs - last.TimeMs <= 250)
        {
            sample.Velocity = (position - last.Position) / ((nowMs - last.TimeMs) / 1000f);
        }
        _followerLast[follower] = (position, nowMs);
        _aheadPosition = position;
        sample.WriteTo(follower.Status, position);
    }

    // Distance along the line from point 0 to every point (following NextId), the points in that
    // order, and the lap length. Nulls unless NextId leads once through every point and back.
    private static (float[]? Arc, int[]? Order, float Length) LapDistances(ReadOnlySpan<SplinePoint> points)
    {
        int n = points.Length;
        if (n < 2)
        {
            return (null, null, 0f);
        }

        var arc = new float[n];
        var order = new int[n];
        Array.Fill(arc, -1f);
        arc[0] = 0f;
        int id = 0;
        float length = 0f;
        for (int k = 1; k <= n; k++)
        {
            int next = points[id].NextId;
            if (next < 0 || next >= n)
            {
                return (null, null, 0f);
            }

            length += Vector3.Distance(points[id].Position, points[next].Position);
            if (next == 0)
            {
                return k == n ? (arc, order, length) : (null, null, 0f);
            }

            if (arc[next] >= 0f)
            {
                return (null, null, 0f);
            }

            arc[next] = length;
            order[k] = next;
            id = next;
        }
        return (null, null, 0f);
    }

    private float WrapAlongLap(float distance)
    {
        distance %= _lapLength;
        return distance < 0f ? distance + _lapLength : distance;
    }

    // Along the line from the i-th bot in _spaced (sorted by distance) to the next one; the last
    // one's gap runs over the lap's end. The gaps add up to the lap even for bots on one point.
    private float GapAhead(int i)
    {
        return i + 1 < _spaced.Count
            ? _spaced[i + 1].Arc - _spaced[i].Arc
            : _spaced[0].Arc + _lapLength - _spaced[i].Arc;
    }

    // The last point at or before this distance along the lap.
    private int PointAtDistance(float distance)
    {
        int[] order = _lapOrder!;
        float[] arc = _lapArc!;
        int lo = 0;
        int hi = order.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (arc[order[mid]] <= distance)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return order[lo];
    }

    // HoldBotSpacing, once a tick before any car moves: every bot's distance along the line, and
    // its pace (a policy that reads it; its grip then stays at the setting) or else its
    // RearSlidingGrip eased toward faster when there is more room ahead of it than behind
    // (slower the other way). A lone bot, or spacing off: eased back to the setting.
    private void HoldSpacing(DriftPolicy policy, float dt)
    {
        _spaced.Clear();
        foreach (var (_, aiState) in _driven)
        {
            if (_cars.TryGetValue(aiState, out DriftCarState? state) && state.SpawnCounter == aiState.SpawnCounter)
            {
                int point = state.PreviousNearestPointId ?? -1;
                if (_lapArc != null && point >= 0 && point < _lapArc.Length)
                {
                    _spaced.Add((state, _lapArc[point]));
                }
            }
        }
        _spaced.Sort((a, b) => a.Arc.CompareTo(b.Arc));

        int n = _spaced.Count;
        bool spacing = _configuration.HoldBotSpacing && n >= 2;
        bool byPace = policy.ReadsPace;
        float low = byPace ? _paceFloor : policy.RearSlidingGripLow;
        float high = byPace ? policy.PaceHigh : policy.RearSlidingGripHigh;
        float middle = byPace ? _basePace : _rearSlidingGrip;
        float halfRange = 0.5f * (high - low);
        float ease = Math.Min(1f, dt / SpacingEaseSeconds);
        for (int i = 0; i < n; i++)
        {
            DriftCarState state = _spaced[i].State;
            float target = middle;
            state.GapAheadM = -1f;
            if (spacing)
            {
                float ahead = GapAhead(i);
                float behind = GapAhead((i + n - 1) % n);
                float imbalance = (ahead - behind) / MathF.Max(ahead + behind, 1f);
                target = Math.Clamp(middle + SpacingGain * imbalance * halfRange, low, high);
                state.GapAheadM = ahead;
            }
            if (byPace)
            {
                state.Pace += (target - state.Pace) * ease;
                state.RearSlidingGrip = _rearSlidingGrip;
            }
            else
            {
                state.RearSlidingGrip += (target - state.RearSlidingGrip) * ease;
            }
        }
    }

    // HoldBotSpacing: a car just taken over goes to the middle of the largest gap between the
    // other bots (this tick's, from HoldSpacing) that the server would not object to
    // (SpotSuitsPlayers), along the line at the speed it has. Not for a lone bot.
    private void PlaceInLargestGap(byte sessionId, DriftCarState state)
    {
        int n = _spaced.Count;
        if (!_configuration.HoldBotSpacing || _lapArc == null || _lapOrder == null || n == 0)
        {
            return;
        }

        // The gaps largest first: the server spawns bots ahead of a player, so the largest is
        // usually the one the player is in, and its middle too near them.
        if (_gapOrder.Length < n)
        {
            _gapOrder = new int[2 * n];
        }
        for (int i = 0; i < n; i++)
        {
            _gapOrder[i] = i;
        }
        Array.Sort(_gapOrder, 0, n, Comparer<int>.Create((a, b) => GapAhead(b).CompareTo(GapAhead(a))));

        int pointId = -1;
        float placedGap = 0f;
        for (int k = 0; k < n && pointId < 0; k++)
        {
            int i = _gapOrder[k];
            float gap = GapAhead(i);
            int candidate = PointAtDistance(WrapAlongLap(_spaced[i].Arc + 0.5f * gap));
            Vector3 forward = _spline.Operations.GetForwardVector(candidate);
            if (new Vector2(forward.X, forward.Z).LengthSquared() > 0.0001f
                && SpotSuitsPlayers(_spline.Points[candidate].Position))
            {
                pointId = candidate;
                placedGap = gap;
            }
        }

        if (pointId >= 0)
        {
            Vector3 forward = _spline.Operations.GetForwardVector(pointId);
            Vector3 along = Vector3.Normalize(new Vector3(forward.X, 0f, forward.Z));
            float speed = new Vector2(state.Velocity.X, state.Velocity.Z).Length();
            state.Position = _spline.Points[pointId].Position;
            state.Velocity = along * speed;
            state.PreviousNearestPointId = pointId;
            VerboseLog.Information("GradientDrift: AI car {SessionId} placed at spline point {PointId}, the middle of a {Gap:F0} m gap between bots",
                sessionId, pointId, placedGap);
        }
        else
        {
            pointId = _spline.WorldToSpline(state.Position).PointId;
            if (pointId < 0 || pointId >= _lapArc.Length)
            {
                return;
            }
        }

        // so a second car taken over this tick sees this one
        float arc = _lapArc[pointId];
        int at = 0;
        while (at < n && _spaced[at].Arc <= arc)
        {
            at++;
        }
        _spaced.Insert(at, (state, arc));
    }

    // The spot as the server's AI judges players (AiBehavior): no nearer any active player's car
    // than SpawnSafetyDistanceToPlayerMeters, where it would not spawn a car either, and within
    // PlayerRadiusMeters (with a margin, as the car drives on) of the point
    // PlayerPositionOffsetMeters ahead of one, or the server takes the car away again and spawns
    // it back by a player. Any spot when no player is active.
    private bool SpotSuitsPlayers(Vector3 spot)
    {
        const float RadiusMargin = 0.8f;
        long now = _sessionManager.ServerTimeMilliseconds;
        bool anyPlayer = false;
        bool inRange = false;
        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            if (car.AiControlled || car.Client?.HasSentFirstUpdate != true
                || now - car.LastActiveTime >= _aiParams.PlayerAfkTimeoutMilliseconds)
            {
                continue;
            }

            anyPlayer = true;
            Vector3 position = car.Status.Position;
            if (Vector3.DistanceSquared(position, spot) < _aiParams.SpawnSafetyDistanceToPlayerSquared)
            {
                return false;
            }

            Vector3 ahead = position;
            if (car.Status.Velocity != Vector3.Zero)
            {
                ahead += Vector3.Normalize(car.Status.Velocity) * _aiParams.PlayerPositionOffsetMeters;
            }
            if (Vector3.DistanceSquared(ahead, spot) < RadiusMargin * RadiusMargin * _aiParams.PlayerRadiusSquared)
            {
                inRange = true;
            }
        }
        return !anyPlayer || inRange;
    }

    // The first connected human's car, as its own client reports it. Read-only.
    private HumanCarSample SampleHumanCar()
    {
        var sample = new HumanCarSample { SessionId = -1 };
        if (_telemetry == null || !_configuration.LogHumanCarForCalibration)
        {
            return sample;
        }

        foreach (EntryCar car in _entryCarManager.EntryCars)
        {
            if (car.Client == null)
            {
                continue;
            }

            sample.SessionId = car.SessionId;
            sample.PosX = car.Status.Position.X;
            sample.PosY = car.Status.Position.Y;
            sample.PosZ = car.Status.Position.Z;
            sample.VelocityY = car.Status.Velocity.Y;
            sample.RotationX = car.Status.Rotation.X;
            sample.Pitch = car.Status.Rotation.Y;
            sample.Roll = car.Status.Rotation.Z;
            sample.StatusFlag = (uint)car.Status.StatusFlag;
            sample.PerformanceDelta = car.Status.PerformanceDelta;
            sample.SpeedMs = car.Status.Velocity.Length();
            sample.SteerAngleByte = car.Status.SteerAngle;
            sample.WheelAngleByte = car.Status.WheelAngle;
            sample.EngineRpm = car.Status.EngineRpm;
            sample.GearByte = car.Status.Gear;
            sample.GasByte = car.Status.Gas;
            sample.TyreFLByte = car.Status.TyreAngularSpeed[0];
            sample.TyreFRByte = car.Status.TyreAngularSpeed[1];
            sample.TyreRLByte = car.Status.TyreAngularSpeed[2];
            sample.TyreRRByte = car.Status.TyreAngularSpeed[3];
            break;
        }

        return sample;
    }
}
