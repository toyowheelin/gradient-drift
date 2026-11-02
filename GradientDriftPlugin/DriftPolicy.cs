// The learned drift policy: loads a network trained offline (drift_rl/) and evaluates it once
// per car per tick. Nothing learns on the server.
//
//   x      = clamp((obs - obs_mean) / obs_std, -obs_clip, obs_clip)
//   h1     = tanh(W1 x  + b1)
//   h2     = tanh(W2 h1 + b2)
//   action = clamp(W3 h2 + b3, -1, 1)
//
//   action[0]  steering angle, as a fraction of full lock
//   action[1]  target speed: MaxSpeed * (1 + action[1]) / 2; or, for a policy trained with
//              EnableRearFrictionCircle, the pedal: throttle above 0, brakes below
//   action[2]  (policies with a third output) the brake WITH the throttle, clamped to [0, 1]:
//              left-foot braking on the front axle, the way a drifter holds a gap (DriveByPedal)
//
// The inputs, their order and their scale constants are part of the trained network just
// as much as its weights. drift_rl/drift_env.py build_obs() is the reference and
// BuildObservation() below is a line-for-line port of it; drift_rl/check_plugin_policy.py
// checks the port against real telemetry. Change either only together with a retrain.

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using System.Text.Json;
using AssettoServer.Server.Ai.Splines;
using Serilog;

namespace GradientDriftPlugin;

// What the next observation needs from the tick before. One per car, discarded when the car is
// re-teleported.
public sealed class DriftPolicyCarState
{
    // The actions applied last tick, after clamping to [-1, 1] (the brake to [0, 1]).
    public float PreviousSteerAction;
    public float PreviousSpeedAction;
    public float PreviousBrakeAction;
    // Last tick's steering command (also read by the rear thrust limit), and |tire force| /
    // available peak on each axle.
    public float LastSteerCommandRadians;
    public float LastFrontSaturation;
    public float LastRearSaturation;
    // Last tick's raycast fan. HasFanCast is false until the first cast after a (re)spawn.
    public float LastFanCenterAngle;
    public bool HasFanCast;
    public readonly float[] LastFanDistances = new float[DriftPolicy.FanRayCount];
    // The pace error input (GradientDriftPlugin.UpdatePaceOver): where the car was along the line last
    // tick (NaN before the first), its speed along the line and the speed asked of it, both
    // averaged, how many ticks they have run, and the result.
    public float PaceArcM = float.NaN;
    public float PaceSpeedAverageMs;
    public float PaceAskedAverageMs;
    public int PaceTicks;
    public float PaceOver;
    // The history block (DriftPolicy.AppendHistory): past frames of the history inputs, a ring with
    // HistoryHead at the latest. Null until the first tick.
    public float[]? History;
    public int HistoryHead;
}

// Valid is false when the network produced a non-finite value (the actions are then 0). Brake is
// 0 for a policy with two outputs.
public readonly struct DriftPolicyAction
{
    public readonly float Steer;
    public readonly float Speed;
    public readonly float Brake;
    public readonly bool Valid;

    public DriftPolicyAction(float steer, float speed, float brake, bool valid)
    {
        Steer = steer;
        Speed = speed;
        Brake = brake;
        Valid = valid;
    }
}

public sealed class DriftPolicy
{
    // Inputs 0 - 45; newer policies also read 46 - 51, the room to the course's
    // edges (CourseRoom), and the newest 52 - 53 too, the pace the car is asked to drive and last
    // tick's brake action. The observation buffer always holds all of them. Input 52 is the pace
    // as a fraction of the course's flat-out speed ("pace_fraction_range"), or for gen4 policies
    // ("pace_kmh_range") km/h / 50. The newest also read 54, how far over the speed asked of it the
    // car has been driving lately (PaceOver; needs the course's pace reference file).
    // The any-car policies read 55 - 65 as well, what tells this car from another (CarSpec.CarInputs),
    // and the newest 66, how good a driver to be (the skill, 0 - 1). A tandem policy reads 67 - 93
    // as well, the car ahead (ChaseInputs), and carries a residual network added to the actions only
    // while one is there.
    public const int ObservationSize = ChaseInputs.Slot + ChaseInputs.Count;
    public const int ObservationSizeV6 = ChaseInputs.Slot;
    public const int ObservationSizeV5 = 66;
    public const int ObservationSizeV4 = 55;
    public const int ObservationSizeV1 = 46;
    public const int ObservationSizeV2 = 52;
    public const int ObservationSizeV3 = 54;
    private const int CourseRoomSlot = 46;
    private const float CourseRoomScale = 10f;
    private const float CourseRoomMaxM = 20f;
    private const int PaceSlot = 52;
    private const float PaceScaleKmh = 50f;     // gen4 policies only
    private const int PaceOverSlot = 54;
    private const float PaceOverClip = 1f;
    private const int CarSlot = 55;
    private const int SkillSlot = 66;
    // steer and speed; newer policies also have the brake
    public const int ActionSize = 3;
    public const int ActionSizeV1 = 2;
    public const int FanRayCount = 15;
    public const int PinCount = 6;
    public const int LookaheadCount = 5;

    // Observation scaling, fixed by training. Literals on purpose: retuning a sensor range in the
    // configuration must not silently rescale what the network reads.
    private const float SpeedScale = 25f;
    private const float LateralVelocityScale = 10f;
    private const float YawRateScale = 2f;
    private const float CrossTrackScale = 15f;
    private const float FanDistanceScale = 60f;
    private const float PinDistanceScale = 12f;
    private const float DtScale = 0.1f;

    public string Track { get; }
    // Every real track it was trained on ("tracks"), and whether generated courses were part of
    // its training ("generated"): such a policy drives tracks it has not seen.
    public IReadOnlyList<string> Tracks { get; private set; } = Array.Empty<string>();
    public bool TrainedOnGeneratedCourses { get; private set; }

    // How many inputs this policy reads (ObservationSizeV1 ... V6 or ObservationSize).
    public int InputCount { get; }
    public bool ReadsCourseRoom => InputCount > ObservationSizeV1;
    // A policy that reads the pace and was trained with one: the plugin spaces bots by pace
    // (HoldBotSpacing), not grip. One with the input but trained without a pace reads 0 there, as
    // it did in training.
    public bool ReadsPace => InputCount > ObservationSizeV2 && PaceHigh > 0f;
    // A pace policy that also reads its pace error: needs the course's reference speed per point.
    public bool ReadsPaceOver => InputCount > ObservationSizeV3 && PaceHigh > 0f && PaceIsFraction;
    // An any-car policy: reads the car inputs, and each bot drives with its own car's numbers
    // (cars/<model>.json beside the plugin) and an angle kit of at least DriftSteerLockDeg.
    public bool ReadsCar => InputCount > ObservationSizeV4;
    public float DriftSteerLockDeg { get; private set; }
    // A policy that reads the skill input, and the skills it was trained over ("skill_range").
    public bool ReadsSkill => InputCount > ObservationSizeV5;
    public float SkillLow { get; private set; } = 1f;
    public float SkillHigh { get; private set; } = 1f;
    // A tandem policy: reads the car ahead, either as one network trained with chase episodes (a
    // "chase" block in the file) or through a residual network added to the actions while input
    // _residualGate is above _residualGateAbove (lead_present). ChasePaceMin: the bottom of the
    // chase pace in training, a share of PaceLow ("chase.controller.chase_pace_min").
    public bool ReadsChase => BaseInputCount > ObservationSizeV6 && (_residualWeights != null || _trainedToChase);
    // A policy that also reads a short history of some of its inputs (a "history" block in the file;
    // drift_rl/policy.py): after the BaseInputCount inputs come, for f = 1 .. HistoryFrames, the inputs
    // _historySlots as they were f x _historyEvery ticks ago. The largest block a file may ask for is
    // MaxHistoryInputs.
    public const int MaxHistoryInputs = 256;
    public int HistoryFrames { get; private set; }
    public int BaseInputCount => InputCount - HistoryFrames * _historySlots.Length;
    private int[] _historySlots = Array.Empty<int>();
    private int _historyEvery = 1;
    private int _historyResetSlot;
    private bool _trainedToChase;
    public float ChasePaceMin { get; private set; } = 1f;
    // How long the pace error averages over, seconds ("pace_average_s"; the reward's).
    public float PaceAverageSeconds { get; private set; } = 1f;
    // How many outputs it has (ActionSizeV1 or ActionSize).
    public int OutputCount { get; }
    public bool HasBrake => OutputCount > ActionSizeV1;
    public long TrainedTicks { get; }
    public string SourcePath { get; }

    // The RearSlidingGrip values the policy was trained over ("rear_sliding_grip_range" in the
    // file). 1 to 1 for a policy without the key: it has only ever driven with no loss of grip.
    public float RearSlidingGripLow { get; private set; } = 1f;
    public float RearSlidingGripHigh { get; private set; } = 1f;

    // The paces the policy was trained to drive: fractions of the course's flat-out speed
    // ("pace_fraction_range", PaceIsFraction), or km/h along the line for gen4 ("pace_kmh_range");
    // 0 to 0 for a policy that does not read the pace.
    public float PaceLow { get; private set; }
    public float PaceHigh { get; private set; }
    public bool PaceIsFraction { get; private set; }

    // Plugin settings the policy was trained with where they differ from this plugin's
    // defaults ("plugin_settings" in the file), by setting name. The plugin applies them.
    public IReadOnlyDictionary<string, float> PluginSettings { get; private set; } = new Dictionary<string, float>();

    private readonly float[] _obsMean;
    private readonly float[] _obsStd;
    private readonly float _obsClip;
    // _weights[l] is row-major [outputs x inputs]. _activations are scratch buffers reused every
    // call (OnUpdate is single-threaded), so evaluation allocates nothing.
    private readonly float[][] _weights;
    private readonly float[][] _biases;
    private readonly float[][] _activations;
    private float[][]? _residualWeights;
    private float[][]? _residualBiases;
    private float[][]? _residualActivations;
    private int _residualGate;
    private float _residualGateAbove;

    // Per spline point, built once at load: direction of travel, the point id each look-ahead
    // distance further on, and the tightest radius within the look-ahead window. Same
    // construction as drift_env.spline_lookahead_tables().
    private readonly float[] _splineDirection;
    private readonly int[] _lookaheadIndex;
    private readonly float[] _minRadiusAhead;

    private bool _warnedNonFinite;

    private DriftPolicy(string sourcePath, string track, long trainedTicks, int inputCount, int outputCount, float[] obsMean, float[] obsStd,
        float obsClip, float[][] weights, float[][] biases, float[] lookaheadMeters, float minRadiusWindowMeters,
        ReadOnlySpan<SplinePoint> points)
    {
        SourcePath = sourcePath;
        Track = track;
        InputCount = inputCount;
        OutputCount = outputCount;
        TrainedTicks = trainedTicks;
        _obsMean = obsMean;
        _obsStd = obsStd;
        _obsClip = obsClip;
        _weights = weights;
        _biases = biases;

        _activations = new float[biases.Length + 1][];
        _activations[0] = new float[inputCount];
        for (int layer = 0; layer < biases.Length; layer++)
        {
            _activations[layer + 1] = new float[biases[layer].Length];
        }

        int pointCount = points.Length;
        _splineDirection = new float[pointCount];
        _lookaheadIndex = new int[pointCount * LookaheadCount];
        _minRadiusAhead = new float[pointCount];
        for (int pointId = 0; pointId < pointCount; pointId++)
        {
            ref readonly var point = ref points[pointId];

            Vector3 forward = point.NextId >= 0 && point.NextId < pointCount
                ? points[point.NextId].Position - point.Position
                : Vector3.Zero;
            _splineDirection[pointId] = MathF.Atan2(forward.Z, forward.X);

            for (int lookahead = 0; lookahead < LookaheadCount; lookahead++)
            {
                int currentId = pointId;
                float accumulated = 0f;
                for (int step = 0; step < pointCount; step++)
                {
                    ref readonly var current = ref points[currentId];
                    if (current.NextId < 0 || current.NextId >= pointCount)
                        break;
                    if (accumulated + current.Length > lookaheadMeters[lookahead])
                        break;
                    accumulated += current.Length;
                    currentId = current.NextId;
                }
                _lookaheadIndex[pointId * LookaheadCount + lookahead] = currentId;
            }

            int walkId = pointId;
            float walked = 0f;
            float minRadius = point.Radius;
            for (int step = 0; step < pointCount && walked < minRadiusWindowMeters; step++)
            {
                ref readonly var current = ref points[walkId];
                if (current.NextId < 0 || current.NextId >= pointCount)
                    break;
                walked += current.Length;
                walkId = current.NextId;
                minRadius = MathF.Min(minRadius, points[walkId].Radius);
            }
            _minRadiusAhead[pointId] = minRadius;
        }
    }

    // Returns null, and logs why, on any problem. The caller then leaves the cars to the stock AI.
    public static DriftPolicy? TryLoad(string path, ReadOnlySpan<SplinePoint> points)
    {
        if (!File.Exists(path))
        {
            Log.Warning("GradientDrift: no policy file at {Path}", Path.GetFullPath(path));
            return null;
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;

            string format = root.GetProperty("format").GetString() ?? "";
            if (format != "drift-policy-v1")
            {
                Log.Warning("GradientDrift: policy file {Path} has format '{Format}', expected 'drift-policy-v1'",
                    Path.GetFullPath(path), format);
                return null;
            }

            int fileObsSize = root.GetProperty("obs_size").GetInt32();
            int actionSize = root.GetProperty("action_size").GetInt32();
            // a history block: the inputs after the env's own (validated with the block below)
            int[] historySlots = Array.Empty<int>();
            int historyFrames = 0, historyEvery = 1, historyResetSlot = 0;
            if (root.TryGetProperty("history", out JsonElement historyElement) && historyElement.ValueKind == JsonValueKind.Object)
            {
                float[] slotValues = ReadFloatArray(historyElement.GetProperty("slots"));
                historySlots = Array.ConvertAll(slotValues, v => (int)v);
                historyFrames = historyElement.GetProperty("frames").GetInt32();
                historyEvery = historyElement.GetProperty("every").GetInt32();
                historyResetSlot = historyElement.GetProperty("reset_slot").GetInt32();
                int count = historyFrames * historySlots.Length;
                int baseSize = fileObsSize - count;
                if (historyFrames < 1 || historyEvery < 1 || historySlots.Length == 0 || count > MaxHistoryInputs
                    || baseSize != ObservationSize || historyResetSlot < 0 || historyResetSlot >= baseSize
                    || Array.Exists(historySlots, j => j < 0 || j >= baseSize))
                {
                    Log.Warning("GradientDrift: policy file {Path} has a history block this build cannot run ({Frames} frames of {Slots} inputs every {Every} ticks after {Base} inputs; at most {Max} history inputs after {Size})",
                        Path.GetFullPath(path), historyFrames, historySlots.Length, historyEvery, baseSize, MaxHistoryInputs, ObservationSize);
                    return null;
                }
            }
            int obsSize = fileObsSize - historyFrames * historySlots.Length;
            if ((obsSize != ObservationSize && obsSize != ObservationSizeV6 && obsSize != ObservationSizeV5 && obsSize != ObservationSizeV4
                 && obsSize != ObservationSizeV3 && obsSize != ObservationSizeV2 && obsSize != ObservationSizeV1)
                || (actionSize != ActionSize && actionSize != ActionSizeV1))
            {
                Log.Warning("GradientDrift: policy file {Path} expects {ObsSize} observations / {ActionSize} actions, this build provides {BuildObs} (or {V6Obs}, {V5Obs}, {V4Obs}, {V3Obs}, {MidObs} or {OldObs}) / {BuildActions} (or {OldActions})",
                    Path.GetFullPath(path), obsSize, actionSize, ObservationSize, ObservationSizeV6, ObservationSizeV5, ObservationSizeV4, ObservationSizeV3, ObservationSizeV2, ObservationSizeV1, ActionSize, ActionSizeV1);
                return null;
            }

            float[] obsMean = ReadFloatArray(root.GetProperty("obs_mean"));
            float[] obsStd = ReadFloatArray(root.GetProperty("obs_std"));
            float obsClip = (float)root.GetProperty("obs_clip").GetDouble();
            float[] lookaheadMeters = ReadFloatArray(root.GetProperty("lookahead_m"));
            float minRadiusWindowMeters = (float)root.GetProperty("min_radius_window_m").GetDouble();
            string track = root.TryGetProperty("track", out JsonElement trackElement) ? trackElement.GetString() ?? "" : "";
            long trainedTicks = root.TryGetProperty("trained_ticks", out JsonElement ticksElement) ? ticksElement.GetInt64() : 0L;

            string controlMode = root.TryGetProperty("control_mode", out JsonElement modeElement) ? modeElement.GetString() ?? "" : "residual";
            if (controlMode != "direct")
            {
                Log.Warning("GradientDrift: policy file {Path} is a '{Mode}' policy; this plugin runs 'direct' policies only (train a new one with drift_rl/train_ppo_gpu.py)",
                    Path.GetFullPath(path), controlMode);
                return null;
            }

            if (obsMean.Length != fileObsSize || obsStd.Length != fileObsSize || lookaheadMeters.Length != LookaheadCount)
            {
                Log.Warning("GradientDrift: policy file {Path} has obs_mean/obs_std/lookahead_m of the wrong length",
                    Path.GetFullPath(path));
                return null;
            }
            for (int i = 0; i < fileObsSize; i++)
            {
                if (!float.IsFinite(obsMean[i]) || !float.IsFinite(obsStd[i]) || obsStd[i] <= 0f)
                {
                    Log.Warning("GradientDrift: policy file {Path} has a non-finite or non-positive normalisation value at observation {Index}",
                        Path.GetFullPath(path), i);
                    return null;
                }
            }

            if (!TryReadLayers(path, root.GetProperty("layers"), fileObsSize, "", out float[][] weights, out float[][] biases))
            {
                return null;
            }
            int layerCount = weights.Length;
            int inputs = layerCount > 0 ? biases[layerCount - 1].Length : 0;

            if (layerCount == 0 || inputs != actionSize)
            {
                Log.Warning("GradientDrift: policy file {Path} does not end in a {ActionSize}-unit layer",
                    Path.GetFullPath(path), actionSize);
                return null;
            }
            float[][]? residualWeights = null;
            float[][]? residualBiases = null;
            int residualGate = -1;
            float residualGateAbove = 0.5f;
            if (root.TryGetProperty("residual", out JsonElement residualElement))
            {
                residualGate = residualElement.GetProperty("gate_input").GetInt32();
                residualGateAbove = (float)residualElement.GetProperty("gate_above").GetDouble();
                if (residualGate != ChaseInputs.Slot || obsSize != ObservationSize || historyFrames > 0)
                {
                    Log.Warning("GradientDrift: policy file {Path} has a residual network gated by input {Gate} of {ObsSize}; this build runs one gated by the car ahead (input {Slot} of {BuildObs})",
                        Path.GetFullPath(path), residualGate, obsSize, ChaseInputs.Slot, ObservationSize);
                    return null;
                }
                if (!TryReadLayers(path, residualElement.GetProperty("layers"), obsSize, "residual ", out residualWeights, out residualBiases))
                {
                    return null;
                }
                if (residualWeights.Length == 0 || residualBiases[^1].Length != actionSize)
                {
                    Log.Warning("GradientDrift: policy file {Path} residual network does not end in a {ActionSize}-unit layer",
                        Path.GetFullPath(path), actionSize);
                    return null;
                }
            }

            var policy = new DriftPolicy(Path.GetFullPath(path), track, trainedTicks, fileObsSize, actionSize, obsMean, obsStd, obsClip,
                weights, biases, lookaheadMeters, minRadiusWindowMeters, points);
            policy.HistoryFrames = historyFrames;
            policy._historySlots = historySlots;
            policy._historyEvery = historyEvery;
            policy._historyResetSlot = historyResetSlot;
            if (residualWeights != null && residualBiases != null)
            {
                policy._residualWeights = residualWeights;
                policy._residualBiases = residualBiases;
                policy._residualActivations = new float[residualBiases.Length + 1][];
                policy._residualActivations[0] = policy._activations[0];
                for (int layer = 0; layer < residualBiases.Length; layer++)
                {
                    policy._residualActivations[layer + 1] = new float[residualBiases[layer].Length];
                }
                policy._residualGate = residualGate;
                policy._residualGateAbove = residualGateAbove;
            }
            if (root.TryGetProperty("chase", out JsonElement chaseElement) && chaseElement.ValueKind == JsonValueKind.Object)
            {
                policy._trainedToChase = obsSize == ObservationSize;
                if (chaseElement.TryGetProperty("controller", out JsonElement controllerElement)
                    && controllerElement.TryGetProperty("chase_pace_min", out JsonElement paceMinElement))
                {
                    policy.ChasePaceMin = Math.Clamp((float)paceMinElement.GetDouble(), 0.1f, 1f);
                }
            }
            bool fractionPaces = root.TryGetProperty("pace_fraction_range", out JsonElement paceRangeElement);
            if (fractionPaces || root.TryGetProperty("pace_kmh_range", out paceRangeElement))
            {
                float[] paceRange = ReadFloatArray(paceRangeElement);
                if (paceRange.Length != 2 || !(paceRange[0] > 0f) || !(paceRange[0] <= paceRange[1])
                    || (fractionPaces && !(paceRange[1] <= 1.5f)))
                {
                    Log.Warning("GradientDrift: policy file {Path} has a pace range that is not [low, high] with 0 < low <= high",
                        Path.GetFullPath(path));
                    return null;
                }
                policy.PaceLow = paceRange[0];
                policy.PaceHigh = paceRange[1];
                policy.PaceIsFraction = fractionPaces;
            }
            if (root.TryGetProperty("tracks", out JsonElement tracksElement) && tracksElement.ValueKind == JsonValueKind.Array)
            {
                policy.Tracks = tracksElement.EnumerateArray().Where(e => e.ValueKind == JsonValueKind.String)
                    .Select(e => e.GetString() ?? "").Where(t => t.Length > 0).ToArray();
            }
            policy.TrainedOnGeneratedCourses = root.TryGetProperty("generated", out JsonElement generatedElement)
                && generatedElement.ValueKind == JsonValueKind.Object;
            if (root.TryGetProperty("pace_average_s", out JsonElement paceAverageElement))
            {
                float paceAverage = (float)paceAverageElement.GetDouble();
                if (!(paceAverage > 0f))
                {
                    Log.Warning("GradientDrift: policy file {Path} has a pace_average_s that is not above 0", Path.GetFullPath(path));
                    return null;
                }
                policy.PaceAverageSeconds = paceAverage;
            }
            if (obsSize > ObservationSizeV3 && policy.PaceHigh > 0f && !policy.PaceIsFraction)
            {
                Log.Warning("GradientDrift: policy file {Path} reads the pace error but its paces are not fractions (pace_fraction_range)",
                    Path.GetFullPath(path));
                return null;
            }
            if (obsSize > ObservationSizeV4)
            {
                // an any-car policy says which angle kit every car was given in training
                if (!root.TryGetProperty("cars", out JsonElement carsElement)
                    || !carsElement.TryGetProperty("drift_steer_lock_deg", out JsonElement lockElement)
                    || !(lockElement.GetDouble() > 0.0))
                {
                    Log.Warning("GradientDrift: policy file {Path} reads the car inputs but has no cars.drift_steer_lock_deg",
                        Path.GetFullPath(path));
                    return null;
                }
                policy.DriftSteerLockDeg = (float)lockElement.GetDouble();
            }
            if (obsSize > ObservationSizeV5 && root.TryGetProperty("skill_range", out JsonElement skillElement))
            {
                float[] skillRange = ReadFloatArray(skillElement);
                if (skillRange.Length != 2 || !(skillRange[0] >= 0f) || !(skillRange[0] <= skillRange[1]) || !(skillRange[1] <= 1f))
                {
                    Log.Warning("GradientDrift: policy file {Path} has a skill_range that is not [low, high] within 0 - 1", Path.GetFullPath(path));
                    return null;
                }
                policy.SkillLow = skillRange[0];
                policy.SkillHigh = skillRange[1];
            }
            if (root.TryGetProperty("rear_sliding_grip_range", out JsonElement gripRangeElement))
            {
                float[] gripRange = ReadFloatArray(gripRangeElement);
                if (gripRange.Length != 2 || !(gripRange[0] > 0f) || !(gripRange[0] <= gripRange[1]) || !(gripRange[1] <= 1f))
                {
                    Log.Warning("GradientDrift: policy file {Path} has a rear_sliding_grip_range that is not [low, high] with 0 < low <= high <= 1",
                        Path.GetFullPath(path));
                    return null;
                }
                policy.RearSlidingGripLow = gripRange[0];
                policy.RearSlidingGripHigh = gripRange[1];
            }
            if (root.TryGetProperty("plugin_settings", out JsonElement settingsElement))
            {
                var settings = new Dictionary<string, float>();
                foreach (JsonProperty setting in settingsElement.EnumerateObject())
                {
                    settings[setting.Name] = (float)setting.Value.GetDouble();
                }
                policy.PluginSettings = settings;
            }
            Log.Information("GradientDrift: policy loaded from {Path} - track '{Track}', {Ticks} training ticks, {Layers} layers, look-ahead tables for {Points} spline points",
                policy.SourcePath, track, trainedTicks, layerCount, points.Length);
            return policy;
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "GradientDrift: failed to read policy file {Path}", Path.GetFullPath(path));
            return null;
        }
    }

    // The tanh layers and the linear last one of a network on inputCount inputs; false, logged, on any problem.
    private static bool TryReadLayers(string path, JsonElement layersElement, int inputCount, string what,
        out float[][] weights, out float[][] biases)
    {
        int layerCount = layersElement.GetArrayLength();
        weights = new float[layerCount][];
        biases = new float[layerCount][];
        int inputs = inputCount;
        int layerIndex = 0;
        foreach (JsonElement layerElement in layersElement.EnumerateArray())
        {
            string activation = layerElement.GetProperty("activation").GetString() ?? "";
            string expectedActivation = layerIndex < layerCount - 1 ? "tanh" : "linear";
            if (activation != expectedActivation)
            {
                Log.Warning("GradientDrift: policy file {Path} {What}layer {Layer} has activation '{Activation}', expected '{Expected}'",
                    Path.GetFullPath(path), what, layerIndex, activation, expectedActivation);
                return false;
            }

            float[] bias = ReadFloatArray(layerElement.GetProperty("b"));
            int outputs = bias.Length;
            JsonElement weightRows = layerElement.GetProperty("W");
            if (outputs == 0 || weightRows.GetArrayLength() != outputs)
            {
                Log.Warning("GradientDrift: policy file {Path} {What}layer {Layer} has mismatched W/b sizes",
                    Path.GetFullPath(path), what, layerIndex);
                return false;
            }

            var flat = new float[outputs * inputs];
            int rowIndex = 0;
            foreach (JsonElement rowElement in weightRows.EnumerateArray())
            {
                if (rowElement.GetArrayLength() != inputs)
                {
                    Log.Warning("GradientDrift: policy file {Path} {What}layer {Layer} row {Row} has {Got} weights, expected {Expected}",
                        Path.GetFullPath(path), what, layerIndex, rowIndex, rowElement.GetArrayLength(), inputs);
                    return false;
                }
                int column = 0;
                foreach (JsonElement value in rowElement.EnumerateArray())
                {
                    flat[rowIndex * inputs + column] = (float)value.GetDouble();
                    column++;
                }
                rowIndex++;
            }

            weights[layerIndex] = flat;
            biases[layerIndex] = bias;
            inputs = outputs;
            layerIndex++;
        }
        return true;
    }

    private static float[] ReadFloatArray(JsonElement element)
    {
        var result = new float[element.GetArrayLength()];
        int index = 0;
        foreach (JsonElement value in element.EnumerateArray())
        {
            result[index] = (float)value.GetDouble();
            index++;
        }
        return result;
    }

    // (x + pi) mod 2pi - pi, as the Python side does it, so the two agree even exactly at +-pi.
    private static float WrapPi(float angle)
    {
        float shifted = (angle + MathF.PI) % (2f * MathF.PI);
        if (shifted < 0f)
            shifted += 2f * MathF.PI;
        return shifted - MathF.PI;
    }

    // Fills obs[0..66]; slot order and scaling as drift_env.OBS_NAMES (a policy reads as many as it
    // was trained with). pace: the pace the car is asked to drive, in the policy's units (PaceIsFraction;
    // 0 = none); paceOver: how far over the asked speed it has been driving (0 = none). ownSpeedMs, lateralVelocityMs,
    // yawRateRadS, heading, the pins and carState are the values left by the PREVIOUS tick - what
    // the network saw in training. Slots 35 and 36 belonged to the removed hand-written
    // controller. 36 is always 0. 35 now carries the share of rear grip lost when sliding,
    // (1 - rearSlidingGrip) / 0.5, which is 0, as before, for a policy trained without it.
    public void BuildObservation(Span<float> obs, bool firstTick, float ownSpeedMs, float lateralVelocityMs,
        float yawRateRadS, float crossTrackErrorM, float radius, int nearestPointId, float lineAngle,
        float maxSpeedMs, DriftPolicyCarState carState, float heading, FanRay[]? previousPins, float dtSeconds,
        bool wrongWayRecovering, float maxSteerAngleRadians, float rearSlidingGrip, CourseEdges? course, float pace,
        float paceOver, CarSpec car, bool loadSensitivity, float skill)
    {
        obs[0] = ownSpeedMs / SpeedScale;
        obs[1] = lateralVelocityMs / LateralVelocityScale;
        obs[2] = yawRateRadS / YawRateScale;
        obs[3] = MathF.Atan2(lateralVelocityMs, MathF.Max(ownSpeedMs, 3f)) / (MathF.PI / 2f);
        obs[4] = WrapPi(lineAngle - heading) / MathF.PI;
        obs[5] = Math.Clamp(crossTrackErrorM / CrossTrackScale, -3f, 3f);
        obs[6] = MathF.Min(2f, 20f / MathF.Max(radius, 1f));

        bool pointValid = nearestPointId >= 0 && nearestPointId < _splineDirection.Length;
        for (int lookahead = 0; lookahead < LookaheadCount; lookahead++)
        {
            obs[7 + lookahead] = pointValid
                ? WrapPi(_splineDirection[_lookaheadIndex[nearestPointId * LookaheadCount + lookahead]] - lineAngle) / MathF.PI
                : 0f;
        }

        obs[12] = maxSpeedMs / SpeedScale;
        for (int ray = 0; ray < FanRayCount; ray++)
        {
            obs[13 + ray] = (carState.HasFanCast ? carState.LastFanDistances[ray] : FanDistanceScale) / FanDistanceScale;
        }
        obs[28] = firstTick || !carState.HasFanCast ? 0f : WrapPi(carState.LastFanCenterAngle - heading) / MathF.PI;
        for (int pin = 0; pin < PinCount; pin++)
        {
            float pinDistance = previousPins != null && pin < previousPins.Length ? previousPins[pin].Distance : PinDistanceScale;
            obs[29 + pin] = pinDistance / PinDistanceScale;
        }
        obs[35] = (1f - rearSlidingGrip) / 0.5f;
        obs[36] = 0f;
        obs[37] = carState.PreviousSteerAction;
        obs[38] = carState.PreviousSpeedAction;
        obs[39] = dtSeconds / DtScale;
        obs[40] = wrongWayRecovering ? 1f : 0f;
        obs[41] = carState.LastSteerCommandRadians / maxSteerAngleRadians;
        obs[42] = carState.LastFrontSaturation;
        obs[43] = carState.LastRearSaturation;
        obs[44] = pointValid ? MathF.Min(2f, 20f / MathF.Max(_minRadiusAhead[nearestPointId], 1f)) : 0f;
        obs[45] = firstTick ? 1f : 0f;

        // Room to the course's edges either side, at the car's point and 15 m and 30 m on (the car's
        // offset from the line carried ahead unchanged). drift_env.build_obs, the same.
        for (int j = 0; j < 3; j++)
        {
            float positiveRoom = CourseRoomMaxM;
            float negativeRoom = CourseRoomMaxM;
            if (course != null && pointValid && course.PointCount == _splineDirection.Length)
            {
                int point = j == 0 ? nearestPointId : _lookaheadIndex[nearestPointId * LookaheadCount + j];
                positiveRoom = Math.Clamp(course.PositiveSideM[point] - crossTrackErrorM, -10f, CourseRoomMaxM);
                negativeRoom = Math.Clamp(course.NegativeSideM[point] + crossTrackErrorM, -10f, CourseRoomMaxM);
            }
            obs[CourseRoomSlot + 2 * j] = positiveRoom / CourseRoomScale;
            obs[CourseRoomSlot + 2 * j + 1] = negativeRoom / CourseRoomScale;
        }

        obs[PaceSlot] = PaceIsFraction ? pace : pace / PaceScaleKmh;
        obs[PaceSlot + 1] = carState.PreviousBrakeAction;
        obs[PaceOverSlot] = Math.Clamp(paceOver, -PaceOverClip, PaceOverClip);
        car.CarInputs(obs, CarSlot, loadSensitivity);
        obs[SkillSlot] = skill;
    }

    // The history block into obs after the BaseInputCount inputs BuildObservation (and the chase
    // inputs) wrote, then this tick's frame kept for later ticks: exactly drift_rl/policy.py's
    // Policy.history_step. On a car's first tick (or its first since a re-teleport) every past
    // frame is the present one. Call once per tick per car, just before Evaluate.
    public void AppendHistory(float[] obs, DriftPolicyCarState carState)
    {
        if (HistoryFrames == 0)
        {
            return;
        }
        int n = _historySlots.Length;
        int length = HistoryFrames * _historyEvery;
        float[]? history = carState.History;
        bool fresh = history == null || history.Length != length * n || obs[_historyResetSlot] > 0.5f;
        if (history == null || history.Length != length * n)
        {
            history = new float[length * n];
            carState.History = history;
        }
        if (fresh)
        {
            for (int j = 0; j < length; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    history[j * n + i] = obs[_historySlots[i]];
                }
            }
            carState.HistoryHead = 0;
        }
        int head = carState.HistoryHead;
        int at = BaseInputCount;
        for (int f = 1; f <= HistoryFrames; f++)
        {
            int frame = (head + f * _historyEvery - 1) % length;
            for (int i = 0; i < n; i++)
            {
                obs[at++] = history[frame * n + i];
            }
        }
        head = (head - 1 + length) % length;
        for (int i = 0; i < n; i++)
        {
            history[head * n + i] = obs[_historySlots[i]];
        }
        carState.HistoryHead = head;
    }

    // A non-finite input would otherwise propagate through Math.Clamp into the car's state.
    public DriftPolicyAction Evaluate(ReadOnlySpan<float> obs)
    {
        float[] input = _activations[0];
        for (int i = 0; i < InputCount; i++)
        {
            input[i] = Math.Clamp((obs[i] - _obsMean[i]) / _obsStd[i], -_obsClip, _obsClip);
        }

        float[] output = NetworkMath.Forward(_weights, _biases, _activations);
        if (_residualWeights != null && obs[_residualGate] > _residualGateAbove)
        {
            float[] residual = NetworkMath.Forward(_residualWeights, _residualBiases!, _residualActivations!);
            for (int o = 0; o < output.Length; o++)
            {
                output[o] += residual[o];
            }
        }
        float steer = output[0];
        float speed = output[1];
        float brake = OutputCount > ActionSizeV1 ? output[2] : 0f;
        if (!float.IsFinite(steer) || !float.IsFinite(speed) || !float.IsFinite(brake))
        {
            if (!_warnedNonFinite)
            {
                _warnedNonFinite = true;
                Log.Warning("GradientDrift: the policy produced a non-finite action; the previous tick's action is repeated on such ticks. Logged once - see the PolicyObs telemetry column for the inputs.");
            }
            return new DriftPolicyAction(0f, 0f, 0f, false);
        }

        return new DriftPolicyAction(Math.Clamp(steer, -1f, 1f), Math.Clamp(speed, -1f, 1f), Math.Clamp(brake, 0f, 1f), true);
    }

    // "v0;v1;...;v53" for the PolicyObs telemetry column.
    public static string FormatObservation(ReadOnlySpan<float> obs)
    {
        var builder = new StringBuilder(obs.Length * 8);
        for (int i = 0; i < obs.Length; i++)
        {
            if (i > 0)
                builder.Append(';');
            builder.Append(obs[i].ToString("F4", CultureInfo.InvariantCulture));
        }
        return builder.ToString();
    }
}
