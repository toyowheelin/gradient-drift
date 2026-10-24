using System.Numerics;

namespace GradientDriftPlugin;

// Everything the plugin remembers about one AI car between ticks. Keyed by the AiState object,
// not the session id: with AI overbooking one slot can have several AiState instances alive at
// once. A new instance is created whenever the server (re)teleports the car, so nothing here
// ever describes the car's previous location.
//
// A null value means "not known yet this spawn"; the first tick then starts from that tick's
// own reading instead of ramping or smoothing from a stale one.
internal sealed class DriftCarState
{
    public DriftCarState(Vector3 position, Vector3 velocity, byte spawnCounter)
    {
        Position = position;
        Velocity = velocity;
        SpawnCounter = spawnCounter;
    }

    // AiState.SpawnCounter when this state was created. The server increments it only inside
    // Teleport(), so a different value means the car was placed somewhere else.
    public readonly byte SpawnCounter;

    // Where the plugin's own vehicle model has the car, and how fast it is moving (m/s, world).
    public Vector3 Position;
    public Vector3 Velocity;

    // Forward speed along the car's heading.
    public float? OwnSpeed;

    // Where the car points (radians, atan2(Z, X)), its yaw rate (rad/s) and its sideways speed
    // in the body frame (m/s). Heading and direction of travel differ by the slip angle.
    public float? Heading;
    public float YawRate;
    public float LateralVelocity;

    // History for GlitchCorrector and NearestPointCorrector.
    public float? PreviousLineAngle;
    public int? PreviousNearestPointId;
    public int NearestPointCorrectionStreak;

    // Wrong-way recovery: consecutive ticks pointing too far off the line, and whether the
    // recovery is currently damping the car.
    public int WrongWayStreak;
    public bool WrongWayRecovering;

    // Smoothed ride height, pitch and roll.
    public float? SmoothedHeight;
    public float? SmoothedPitch;
    public float? SmoothedRoll;

    // Front-wheel angle last sent to clients, degrees (see EnableSteeringVisual).
    public float? VisualSteerDeg;

    // Rear tread speed (m/s) and gear last sent to clients (see EnableDrivetrainVisual).
    public float? VisualRearWheelSpeed;
    public int VisualGear = 1;

    // The server's own AI has the car stopped this tick (waiting, about to despawn): the plugin follows it.
    public bool HeldByStock;

    // How much longer the brake lights stay on, seconds.
    public float BrakeLightSeconds;

    // This car's RearSlidingGrip (the setting, or what HoldBotSpacing has eased it to), and
    // the distance along the line to the next bot ahead (-1 = no other bot).
    public float RearSlidingGrip;
    public float GapAheadM = -1f;
    // The pace this car is asked to drive, in the policy's units (a fraction of the course's
    // flat-out speed, or km/h for gen4): the setting, or what HoldBotSpacing has eased it to; 0 = none.
    public float Pace;
    // LeadPlayers: a player is chasing this car, and its pace waits for them.
    public bool LeadingPlayer;

    // The wandering noise of a driver below full skill (BotSkill): steering and pedal.
    public float SkillNoiseSteer;
    public float SkillNoisePedal;
    // what the policy asked for the last two ticks, for the slower reaction of a lower skill
    public (float Steer, float Speed, float Brake) SkillAsked1;
    public (float Steer, float Speed, float Brake) SkillAsked2;
    public int SkillAskedTicks;

    // Last tick's proximity pins, and the rest of what the policy's next observation needs.
    public FanRay[]? PreviousPins;
    public readonly DriftPolicyCarState Policy = new();
}
