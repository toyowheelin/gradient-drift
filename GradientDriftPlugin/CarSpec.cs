using System.Text.Json;

namespace GradientDriftPlugin;

// The numbers of one car the vehicle model drives with (fast_sim's car parameters, by the same
// names in the car files setup writes: drift_env.car_param_values()). Every bot drives as a drift
// car whatever its files say: a welded diff (the model's rear axle is one solid axle) and an angle
// kit of at least the policy's drift lock - the rest (weight, geometry, tyres, engine) is the car's.
public sealed class CarSpec
{
    public const int EngineCurvePoints = 33;
    public const float EngineCurveStepMs = 2.5f;
    public const float DefaultTireFlex = 0.000852f;

    public string Name = "";
    public float YawInertiaKgM2;
    public float DistanceCgToFrontAxleM;
    public float DistanceCgToRearAxleM;
    public float FrontStaticLoadN;
    public float RearStaticLoadN;
    public float LateralDynamicsFrontPeakFriction;
    public float LateralDynamicsRearPeakFriction;
    public float FrontPeakSlipAngleRadians;
    public float RearPeakSlipAngleRadians;
    // tyres.ini FLEX, degrees of slip angle per N on a tire (RoundTireCurve only)
    public float FrontTireFlex = DefaultTireFlex;
    public float RearTireFlex = DefaultTireFlex;
    public float MaxSteerAngleRadians;
    public float CgHeightMeters;
    public float FrontTireLoadSensitivityPerKn;
    public float RearTireLoadSensitivityPerKn;
    public float FrontTrackMeters;
    public float RearTrackMeters;
    public float RollStiffnessFrontShare;
    // The newer tyre model (tyres.ini VERSION 5 and up): grip at a tire's load F is
    // peak x (F / FZ0)^(LS_EXPY - 1). 0 = the older model (TireLoadSensitivityPerKn).
    public float FrontLsExpy;
    public float RearLsExpy;
    public float FrontFz0N;
    public float RearFz0N;
    // The most force the drivetrain can put on the road at EngineCurveStepMs steps of forward
    // speed from 0, N; null = no engine limit (as before).
    public float[]? EngineForceN;
    // This car's flat-out speed over the reference car's: a pace asks for pace x reference x this.
    public float PaceScale = 1f;

    public float MassKg => (FrontStaticLoadN + RearStaticLoadN) / 9.80665f;

    // The plugin's own car (its configuration, after the policy file's settings).
    public static CarSpec FromConfiguration(GradientDriftConfiguration c) => new()
    {
        Name = "(configured)",
        YawInertiaKgM2 = c.YawInertiaKgM2,
        DistanceCgToFrontAxleM = c.DistanceCgToFrontAxleM,
        DistanceCgToRearAxleM = c.DistanceCgToRearAxleM,
        FrontStaticLoadN = c.FrontStaticLoadN,
        RearStaticLoadN = c.RearStaticLoadN,
        LateralDynamicsFrontPeakFriction = c.LateralDynamicsFrontPeakFriction,
        LateralDynamicsRearPeakFriction = c.LateralDynamicsRearPeakFriction,
        FrontPeakSlipAngleRadians = c.FrontPeakSlipAngleRadians,
        RearPeakSlipAngleRadians = c.RearPeakSlipAngleRadians,
        FrontTireFlex = c.FrontTireFlex,
        RearTireFlex = c.RearTireFlex,
        MaxSteerAngleRadians = c.MaxSteerAngleRadians,
        CgHeightMeters = c.CgHeightMeters,
        FrontTireLoadSensitivityPerKn = c.FrontTireLoadSensitivityPerKn,
        RearTireLoadSensitivityPerKn = c.RearTireLoadSensitivityPerKn,
        FrontTrackMeters = c.FrontTrackMeters,
        RearTrackMeters = c.RearTrackMeters,
        RollStiffnessFrontShare = c.RollStiffnessFrontShare,
    };

    // A car file (setup writes <plugin folder>/cars/<car model>.json): a JSON object of the
    // simulator's parameter names. Null, with the reason, if it is missing or incomplete.
    public static CarSpec? Load(string path, float minSteerLockRadians, out string? problem)
    {
        problem = null;
        if (!File.Exists(path))
        {
            problem = "no file";
            return null;
        }
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var v = new Dictionary<string, float>();
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (p.Value.ValueKind == JsonValueKind.Number)
                {
                    v[p.Name] = (float)p.Value.GetDouble();
                }
            }
            string[] required =
            {
                "YAW_INERTIA_KGM2", "DIST_CG_FRONT_M", "DIST_CG_REAR_M", "FRONT_STATIC_LOAD_N", "REAR_STATIC_LOAD_N",
                "LAT_FRONT_PEAK_FRICTION", "LAT_REAR_PEAK_FRICTION", "FRONT_PEAK_SLIP_RAD", "REAR_PEAK_SLIP_RAD",
                "MAX_STEER_RAD",
            };
            foreach (string key in required)
            {
                if (!v.TryGetValue(key, out float x) || !float.IsFinite(x) || x <= 0f)
                {
                    problem = $"{key} missing or not positive";
                    return null;
                }
            }
            float Get(string key, float fallback) => v.TryGetValue(key, out float x) && float.IsFinite(x) ? x : fallback;
            var spec = new CarSpec
            {
                Name = Path.GetFileNameWithoutExtension(path),
                YawInertiaKgM2 = v["YAW_INERTIA_KGM2"],
                DistanceCgToFrontAxleM = v["DIST_CG_FRONT_M"],
                DistanceCgToRearAxleM = v["DIST_CG_REAR_M"],
                FrontStaticLoadN = v["FRONT_STATIC_LOAD_N"],
                RearStaticLoadN = v["REAR_STATIC_LOAD_N"],
                LateralDynamicsFrontPeakFriction = v["LAT_FRONT_PEAK_FRICTION"],
                LateralDynamicsRearPeakFriction = v["LAT_REAR_PEAK_FRICTION"],
                FrontPeakSlipAngleRadians = v["FRONT_PEAK_SLIP_RAD"],
                RearPeakSlipAngleRadians = v["REAR_PEAK_SLIP_RAD"],
                FrontTireFlex = Get("FRONT_TIRE_FLEX", DefaultTireFlex),
                RearTireFlex = Get("REAR_TIRE_FLEX", DefaultTireFlex),
                MaxSteerAngleRadians = MathF.Max(v["MAX_STEER_RAD"], minSteerLockRadians),
                CgHeightMeters = Get("CG_HEIGHT_M", 0.40f),
                FrontTireLoadSensitivityPerKn = Get("FRONT_LOAD_SENSITIVITY_PER_KN", 0f),
                RearTireLoadSensitivityPerKn = Get("REAR_LOAD_SENSITIVITY_PER_KN", 0f),
                FrontTrackMeters = Get("FRONT_TRACK_M", 1.6f),
                RearTrackMeters = Get("REAR_TRACK_M", 1.6f),
                RollStiffnessFrontShare = Get("ROLL_STIFFNESS_FRONT_SHARE", 0.6f),
                FrontLsExpy = Get("FRONT_LS_EXPY", 0f),
                RearLsExpy = Get("REAR_LS_EXPY", 0f),
                FrontFz0N = Get("FRONT_FZ0_N", 0f),
                RearFz0N = Get("REAR_FZ0_N", 0f),
                PaceScale = Get("PACE_SCALE", 1f) > 0f ? Get("PACE_SCALE", 1f) : 1f,
            };
            if (Get("EXT_ENGINE", 0f) > 0.5f)
            {
                var curve = new float[EngineCurvePoints];
                for (int k = 0; k < EngineCurvePoints; k++)
                {
                    curve[k] = MathF.Max(0f, Get($"ENGINE_FORCE_{k:00}", 0f));
                }
                spec.EngineForceN = curve;
            }
            return spec;
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            problem = e.Message;
            return null;
        }
    }

    // The most force the drivetrain can put on the road at this forward speed (fast_sim.engine_force):
    // linearly between the curve's points, 0 past its last one; no limit without a curve.
    public float EngineForce(float speedMs)
    {
        if (EngineForceN == null)
        {
            return float.PositiveInfinity;
        }
        float x = MathF.Max(speedMs, 0f) / EngineCurveStepMs;
        int k = (int)x;
        if (k >= EngineCurvePoints - 1)
        {
            return 0f;
        }
        float w = x - k;
        return EngineForceN[k] * (1f - w) + EngineForceN[k + 1] * w;
    }

    // The slip angle at which each axle's force reaches its peak (fast_sim.tire_curve_angles): the
    // tyres' friction limit angle, plus with RoundTireCurve FLEX x the static load on one tire.
    public (float Front, float Rear) TireCurveAngles(bool roundTireCurve) => roundTireCurve
        ? (FrontPeakSlipAngleRadians + FrontTireFlex * 0.5f * FrontStaticLoadN * MathF.PI / 180f,
           RearPeakSlipAngleRadians + RearTireFlex * 0.5f * RearStaticLoadN * MathF.PI / 180f)
        : (FrontPeakSlipAngleRadians, RearPeakSlipAngleRadians);

    // Each axle's peak friction this tick (fast_sim.axle_friction). Off: the peaks as configured.
    public (float Front, float Rear) AxleFriction(float ownSpeed, float yawRate, bool loadSensitivity)
    {
        float muFront = LateralDynamicsFrontPeakFriction;
        float muRear = LateralDynamicsRearPeakFriction;
        if (!loadSensitivity)
        {
            return (muFront, muRear);
        }
        float frontLoad = FrontStaticLoadN;
        float rearLoad = RearStaticLoadN;
        float mass = (frontLoad + rearLoad) / 9.80665f;
        float sidewaysAccel = MathF.Min(MathF.Abs(ownSpeed * yawRate), 2f * 9.80665f);
        float moved = mass * sidewaysAccel * CgHeightMeters;
        float frontMoved = MathF.Min(0.5f * frontLoad, RollStiffnessFrontShare * moved / MathF.Max(FrontTrackMeters, 0.5f));
        float rearMoved = MathF.Min(0.5f * rearLoad, (1f - RollStiffnessFrontShare) * moved / MathF.Max(RearTrackMeters, 0.5f));
        muFront = FrontLsExpy > 0f
            ? PowerLawAxle(muFront, frontLoad, frontMoved, FrontFz0N, FrontLsExpy)
            : muFront + FrontTireLoadSensitivityPerKn * (0.5f * frontLoad + 2f * frontMoved * frontMoved / frontLoad) / 1000f;
        muRear = RearLsExpy > 0f
            ? PowerLawAxle(muRear, rearLoad, rearMoved, RearFz0N, RearLsExpy)
            : muRear + RearTireLoadSensitivityPerKn * (0.5f * rearLoad + 2f * rearMoved * rearMoved / rearLoad) / 1000f;
        return (MathF.Max(muFront, 0.1f), MathF.Max(muRear, 0.1f));
    }

    // fast_sim._power_law_axle: each tire's grip peak x (load / FZ0)^(LS_EXPY - 1), the axle's
    // the two tires' forces over its load.
    private static float PowerLawAxle(float peak, float axleLoad, float moved, float fz0, float lsExpy)
    {
        float outer = 0.5f * axleLoad + moved;
        float inner = MathF.Max(0.5f * axleLoad - moved, 1f);
        fz0 = MathF.Max(fz0, 1f);
        float force = outer * peak * MathF.Pow(outer / fz0, lsExpy - 1f) + inner * peak * MathF.Pow(inner / fz0, lsExpy - 1f);
        return force / MathF.Max(outer + inner, 1f);
    }

    // The car inputs (drift_env.car_inputs) into obs[k .. k + 10].
    public void CarInputs(Span<float> obs, int k, bool loadSensitivity)
    {
        float fLoad = FrontStaticLoadN;
        float rLoad = RearStaticLoadN;
        float mass = (fLoad + rLoad) / 9.80665f;
        float muF = LateralDynamicsFrontPeakFriction;
        float muR = LateralDynamicsRearPeakFriction;
        if (FrontLsExpy > 0f)
        {
            muF *= MathF.Pow(0.5f * fLoad / MathF.Max(FrontFz0N, 1f), FrontLsExpy - 1f);
        }
        else if (loadSensitivity)
        {
            muF += FrontTireLoadSensitivityPerKn * 0.5f * fLoad / 1000f;
        }
        if (RearLsExpy > 0f)
        {
            muR *= MathF.Pow(0.5f * rLoad / MathF.Max(RearFz0N, 1f), RearLsExpy - 1f);
        }
        else if (loadSensitivity)
        {
            muR += RearTireLoadSensitivityPerKn * 0.5f * rLoad / 1000f;
        }
        obs[k] = mass / 1300f - 1f;
        obs[k + 1] = YawInertiaKgM2 / MathF.Max(mass, 1f) / 1.95f - 1f;
        obs[k + 2] = (DistanceCgToFrontAxleM + DistanceCgToRearAxleM) / 2.5f - 1f;
        obs[k + 3] = (fLoad / MathF.Max(fLoad + rLoad, 1f) - 0.53f) * 10f;
        obs[k + 4] = MaxSteerAngleRadians * (180f / MathF.PI) / 40f - 1f;
        obs[k + 5] = muR / 1.3f - 1f;
        obs[k + 6] = (muF / MathF.Max(muR, 0.1f) - 1f) * 5f;
        obs[k + 7] = RearPeakSlipAngleRadians * (180f / MathF.PI) / 10f - 1f;
        float rearGrip = MathF.Max(muR * rLoad, 1f);
        obs[k + 8] = EngineForceN == null ? 1.5f : Math.Clamp(MathF.Log2(MathF.Max(EngineForce(50f / 3.6f), 1f) / rearGrip), -1.5f, 1.5f);
        obs[k + 9] = EngineForceN == null ? 1.5f : Math.Clamp(MathF.Log2(MathF.Max(EngineForce(90f / 3.6f), 1f) / rearGrip), -1.5f, 1.5f);
        obs[k + 10] = CgHeightMeters / 0.45f - 1f;
    }
}
