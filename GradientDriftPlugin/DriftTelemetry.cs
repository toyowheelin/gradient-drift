using System.Globalization;
using System.Text;
using Serilog;

namespace GradientDriftPlugin;

// One telemetry row: the state of one AI car at the end of one tick, plus what the policy saw
// and did. Angles are radians where the column name does not say "Deg".
internal struct DriftTelemetryRow
{
    public long Tick;
    public long ServerTimeMs;
    public byte SessionId;
    public byte SpawnCounter;
    public float DtSeconds;

    public float PosX, PosY, PosZ;
    public float SpeedMs;                  // length of the velocity sent to clients
    public float OwnSpeedMs;               // forward speed along the heading
    public float MaxSpeedMs;               // this car's speed limit (AiState.MaxSpeed)
    public float Heading;
    public float YawRate;
    public float LateralVelocityMs;
    public float SlipAngle;                // heading minus direction of travel
    public float SteerCommand;
    public float FrontSlipAngle, RearSlipAngle;
    public float FyFrontN, FyRearN;
    public float LongitudinalAccelMs2;

    public int NearestPointId;
    public float RawLineAngle, LineAngle;
    public bool LineAngleCorrected, NearestPointCorrected;
    public float CrossTrackErrorM;

    public FanRay[]? FanRays;
    public float FanCenterAngle;
    public FanRay[]? Pins;

    public float WallPushOutM;
    public bool GroundRayUsed;
    public float GroundRayY;
    public bool WrongWayRecovering;

    public bool PolicyActive;              // false = non-finite output, previous action repeated
    public float PolicySteerAction, PolicySpeedAction;
    public float PolicyBrakeAction;        // 0 for a policy with two outputs
    public string PolicyObservation;

    public byte VisualWheelAngleByte, VisualSteerAngleByte;

    // What clients were sent for the wheels and the engine (see EnableDrivetrainVisual).
    public byte VisualFrontTyreByte, VisualRearTyreByte;
    public float VisualRearWheelSpeedMs;
    public int VisualEngineRpm;
    public int VisualGear;                 // 1 = first
    public byte VisualGasByte;
    // The rest of what clients were sent, for comparing with a human's car.
    public float SentVelocityY, SentPitch, SentRoll;
    public uint SentStatusFlag;

    public float RearSlidingGrip;
    public float GapAheadM;                // along the line to the next bot ahead; -1 = none
    public float Pace;                     // the pace the car is asked to drive (fraction, or km/h for gen4); 0 = none
    public float PaceOver;                 // the pace error input this tick (GradientDriftPlugin.UpdatePaceOver); 0 = none

    // A tandem-train follower's row (TrainSize > 1); FollowLeadId -1 on every other row.
    public int FollowLeadId;               // the lead's session id
    public int FollowPlace;                // 1 = right behind the lead
    public float FollowDelayMs;            // how far behind the lead's replay it is; -1 = not following this tick
    public float FollowGapM;               // straight-line distance to the car in front
    public float FollowOffsetM;            // sideways from the lead's line, signed (each follower keeps to its own side)
}

// What a connected human's client is sending for its own car, as received. SessionId is -1
// when nobody is connected. See LogHumanCarForCalibration.
internal struct HumanCarSample
{
    public int SessionId;
    public float PosX, PosZ, RotationX, SpeedMs;
    public int SteerAngleByte, WheelAngleByte;
    public int EngineRpm, GearByte, GasByte;
    public int TyreFLByte, TyreFRByte, TyreRLByte, TyreRRByte;
    public float PosY, VelocityY, Pitch, Roll;
    public uint StatusFlag;
    public int PerformanceDelta;
}

// Writes drift_physics_telemetry_<date>_<time>.csv, a new file for every server start, so a
// restart never overwrites an earlier run's file. Numbers are always written with '.' as the decimal
// separator, whatever the server's locale. Flushed every FlushIntervalTicks ticks rather than
// every row: a synchronous flush per row on the server's update thread caused tick hitches.
// It stops at maxBytes, and at the first write that fails (a full disk, say), with one line in
// the server log: telemetry must never take the server down.
internal sealed class DriftTelemetry : IDisposable
{
    // The file this run writes, in the server's working directory.
    public string FilePath { get; } = $"drift_physics_telemetry_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
    private const int FlushIntervalTicks = 20;

    private const string Header =
        "Tick,ServerTimeMs,SessionId,SpawnCounter,RealDtMs," +
        "PosX,PosY,PosZ,SpeedKmh,OwnSpeedKmh,CeilingKmh,RealHeadingDeg,YawRateDegS,LateralVelocityMs,RealSlipAngleDeg," +
        "SteerCommandDeg,FrontSlipAngleDeg,RearSlipAngleDeg,FyFrontN,FyRearN,RealizedLongitudinalAccelMs2," +
        "NearestPointId,RawSplineAngle,CorrectedSplineAngle,AngleGlitchCorrected,NearestPointGlitchCorrected,CrossTrackErrorM," +
        "RayDistances,FanCenterDeg,PinDistances," +
        "CollisionPushOutM,GroundRayUsed,GroundRayY,WrongWayRecovering," +
        "PolicyActive,PolicySteerAction,PolicySpeedAction,PolicyObs," +
        "VisualWheelAngleByte,VisualSteerAngleByte," +
        "VisualFrontTyreByte,VisualRearTyreByte,VisualRearWheelSpeedKmh,VisualEngineRpm,VisualGear,VisualGasByte," +
        "HumanSessionId,HumanPosX,HumanPosZ,HumanRotationX,HumanSpeedKmh,HumanSteerAngleByte,HumanWheelAngleByte," +
        "HumanEngineRpm,HumanGearByte,HumanGasByte,HumanTyreFLByte,HumanTyreFRByte,HumanTyreRLByte,HumanTyreRRByte," +
        "SentVelocityY,SentPitch,SentRoll,SentStatusFlag," +
        "HumanPosY,HumanVelocityY,HumanPitch,HumanRoll,HumanStatusFlag,HumanPerformanceDelta," +
        "RearSlidingGrip,GapAheadM,FollowLeadId,FollowPlace,FollowDelayMs,FollowGapM,FollowOffsetM," +
        "PolicyBrakeAction,Pace,PaceOver";

    private const float RadToDeg = 180f / MathF.PI;

    private readonly StreamWriter _writer;
    private readonly StringBuilder _line = new(1024);
    private readonly long _maxBytes;
    private long _bytes;
    private bool _stopped;

    // maxBytes 0 = no limit
    public DriftTelemetry(long maxBytes)
    {
        _maxBytes = maxBytes;
        _writer = new StreamWriter(FilePath, append: false) { AutoFlush = false };
        _writer.WriteLine(Header);
        _bytes = Header.Length + 2;
    }

    public void Write(in DriftTelemetryRow row, in HumanCarSample human)
    {
        if (_stopped)
        {
            return;
        }

        StringBuilder line = _line;
        line.Clear();

        Integer(line, row.Tick);
        Integer(line, row.ServerTimeMs);
        Integer(line, row.SessionId);
        Integer(line, row.SpawnCounter);
        Number(line, row.DtSeconds * 1000f, "F1");

        Number(line, row.PosX, "F2");
        Number(line, row.PosY, "F2");
        Number(line, row.PosZ, "F2");
        Number(line, row.SpeedMs * 3.6f, "F1");
        Number(line, row.OwnSpeedMs * 3.6f, "F2");
        Number(line, row.MaxSpeedMs * 3.6f, "F1");
        Number(line, row.Heading * RadToDeg, "F2");
        Number(line, row.YawRate * RadToDeg, "F2");
        Number(line, row.LateralVelocityMs, "F3");
        Number(line, row.SlipAngle * RadToDeg, "F2");

        Number(line, row.SteerCommand * RadToDeg, "F2");
        Number(line, row.FrontSlipAngle * RadToDeg, "F2");
        Number(line, row.RearSlipAngle * RadToDeg, "F2");
        Number(line, row.FyFrontN, "F1");
        Number(line, row.FyRearN, "F1");
        Number(line, row.LongitudinalAccelMs2, "F3");

        Integer(line, row.NearestPointId);
        Number(line, row.RawLineAngle, "F4");
        Number(line, row.LineAngle, "F4");
        Flag(line, row.LineAngleCorrected);
        Flag(line, row.NearestPointCorrected);
        Number(line, row.CrossTrackErrorM, "F3");

        Distances(line, row.FanRays);
        Number(line, row.FanCenterAngle * RadToDeg, "F1");
        Distances(line, row.Pins);

        Number(line, row.WallPushOutM, "F3");
        Flag(line, row.GroundRayUsed);
        Number(line, row.GroundRayY, "F3");
        Flag(line, row.WrongWayRecovering);

        Flag(line, row.PolicyActive);
        Number(line, row.PolicySteerAction, "F4");
        Number(line, row.PolicySpeedAction, "F4");
        line.Append(row.PolicyObservation).Append(',');

        Integer(line, row.VisualWheelAngleByte);
        Integer(line, row.VisualSteerAngleByte);

        Integer(line, row.VisualFrontTyreByte);
        Integer(line, row.VisualRearTyreByte);
        Number(line, row.VisualRearWheelSpeedMs * 3.6f, "F1");
        Integer(line, row.VisualEngineRpm);
        Integer(line, row.VisualGear);
        Integer(line, row.VisualGasByte);

        Integer(line, human.SessionId);
        Number(line, human.PosX, "F2");
        Number(line, human.PosZ, "F2");
        Number(line, human.RotationX, "F4");
        Number(line, human.SpeedMs * 3.6f, "F1");
        Integer(line, human.SteerAngleByte);
        Integer(line, human.WheelAngleByte);
        Integer(line, human.EngineRpm);
        Integer(line, human.GearByte);
        Integer(line, human.GasByte);
        Integer(line, human.TyreFLByte);
        Integer(line, human.TyreFRByte);
        Integer(line, human.TyreRLByte);
        Integer(line, human.TyreRRByte);

        Number(line, row.SentVelocityY, "F2");
        Number(line, row.SentPitch, "F4");
        Number(line, row.SentRoll, "F4");
        Integer(line, row.SentStatusFlag);

        Number(line, human.PosY, "F3");
        Number(line, human.VelocityY, "F2");
        Number(line, human.Pitch, "F4");
        Number(line, human.Roll, "F4");
        Integer(line, human.StatusFlag);
        Integer(line, human.PerformanceDelta);

        Number(line, row.RearSlidingGrip, "F3");
        Number(line, row.GapAheadM, "F1");
        Integer(line, row.FollowLeadId);
        Integer(line, row.FollowPlace);
        Number(line, row.FollowDelayMs, "F0");
        Number(line, row.FollowGapM, "F2");
        Number(line, row.FollowOffsetM, "F2");
        Number(line, row.PolicyBrakeAction, "F4");
        Number(line, row.Pace, "F4");
        line.Append(row.PaceOver.ToString("F4", CultureInfo.InvariantCulture));

        if (_maxBytes > 0 && _bytes + line.Length + 2 > _maxBytes)
        {
            Stop($"the file reached TelemetryMaxFileMegabytes ({_maxBytes / (1024 * 1024)} MB)");
            return;
        }
        try
        {
            _writer.WriteLine(line.ToString());
            _bytes += line.Length + 2;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
            Stop(e.Message);
        }
    }

    public void FlushIfDue(long tick)
    {
        if (_stopped || tick % FlushIntervalTicks != 0)
        {
            return;
        }
        try
        {
            _writer.Flush();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
            Stop(e.Message);
        }
    }

    private void Stop(string why)
    {
        _stopped = true;
        Log.Warning("GradientDrift: telemetry stopped ({Why}); {Path} has what was written before", why, FilePath);
        try
        {
            _writer.Flush();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
        }
    }

    public void Dispose()
    {
        try
        {
            _writer.Dispose();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    private static void Number(StringBuilder line, float value, string format)
    {
        line.Append(value.ToString(format, CultureInfo.InvariantCulture)).Append(',');
    }

    private static void Integer(StringBuilder line, long value)
    {
        line.Append(value.ToString(CultureInfo.InvariantCulture)).Append(',');
    }

    private static void Flag(StringBuilder line, bool value)
    {
        line.Append(value ? "True" : "False").Append(',');
    }

    // "d0;d1;...;dN" - the hit distance of each ray, in the order the sensor returns them.
    private static void Distances(StringBuilder line, FanRay[]? rays)
    {
        if (rays != null)
        {
            for (int i = 0; i < rays.Length; i++)
            {
                if (i > 0)
                {
                    line.Append(';');
                }
                line.Append(rays[i].Distance.ToString("F1", CultureInfo.InvariantCulture));
            }
        }
        line.Append(',');
    }
}
