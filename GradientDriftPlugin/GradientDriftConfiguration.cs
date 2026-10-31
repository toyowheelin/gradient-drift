using System;

namespace GradientDriftPlugin;

// Settings, read from the "--- !GradientDriftConfiguration" block of extra_cfg.yml. Every value
// has a working default; a block with nothing in it is a valid configuration.
//
// IMPORTANT: everything from "The car" down to "Ride height" describes the vehicle and the
// sensors the policy was TRAINED with. The trainer (drift_rl/) simulates exactly these numbers.
// Change one here and the policy is driving a car it has never seen: retrain, or leave it.
// The one exception is RearSlidingGrip. A policy file can also carry the steering lock, the
// rear thrust limit and the wrong-way thresholds it was trained with; those then replace the
// values here.
public class GradientDriftConfiguration
{
    // ---- The policy ---------------------------------------------------------------------

    // Off, or no usable policy file, or missing track data (geometry, course edges, pace
    // reference), or a policy with a setting this build cannot apply, or one trained only on
    // other real tracks (a policy trained on generated courses drives any track): the plugin
    // does nothing and the server's own spline-following AI drives the cars.
    public bool EnableLearnedPolicy { get; set; } = true;

    // Empty = "GradientDriftPlugin.policy.json" next to GradientDriftPlugin.dll (the build copies it there).
    public string LearnedPolicyPath { get; set; } = "";

    // ---- Several bots ---------------------------------------------------------------------

    // With more than one bot out: a car the plugin takes over is moved to the middle of the
    // largest gap between the others (if that is within the server's PlayerRadiusMeters of a
    // player), and each car is then sped up or slowed down a little so the gaps stay even. A
    // policy that reads the pace (input 52) is asked for a faster or slower pace,
    // inside the range it was trained for, around BotPacePercent; it slows by braking with the
    // throttle, keeping its angle, and every car keeps the same RearSlidingGrip. An older policy
    // has its RearSlidingGrip eased up or down inside its trained range instead (RearSlidingGrip
    // below is then the middle). The bots do not see each other: one that spins can still be
    // driven through by the car behind.
    public bool HoldBotSpacing { get; set; } = true;

    // The pace the bots are asked to drive (a policy that reads it), as a percentage of the
    // course's reference speed (<track>_pace_reference.csv: a quick human's speed at each point,
    // a little faster; drift_rl/pace_model.py): 90 = 90% of that speed at every point of the lap,
    // on any track. Held inside the range the policy was trained for
    // ("pace_fraction_range" in the policy file); 0 = the top of that range. Lower = calmer
    // bots that brake more through the corners. A policy that also reads how far over its asked
    // speed it is needs <track>_pace_reference.csv (the trainer or setup writes it) in the
    // server's folder, beside the course edges.
    public float BotPacePercent { get; set; } = 0f;

    // The slowest pace spacing may ask a bot for, % (a policy whose paces are shares of flat-out
    // speed); 0 = the bottom of the policy's range. Slower bots brake more and slide less: gen5
    // slides 82% of the time at 95 - 100% but 61% at 80%, so the default keeps them at 85 or above.
    public float BotPaceMinPercent { get; set; } = 85f;

    // The same for a gen4 policy, whose paces are km/h along the line ("pace_kmh_range").
    public float BotPaceKmh { get; set; } = 0f;

    // Tandem with a player chasing: a bot (or train) with a player close behind waits for them -
    // full pace with the player right on its tail, slowing to BotPaceMinPercent as they drop back
    // 40 m; further back, spacing as usual. A policy that reads its pace only.
    public bool LeadPlayers { get; set; } = false;

    // Tandem with a bot chasing a player: a player who pulls in within ChaseOptInMeters ahead of a
    // bot's nose is chased until they get ChaseOptOutMeters ahead. The bot's pace comes from the
    // Intelligent Driver Model, aiming ChaseGapMeters behind them. A policy that reads its pace only.
    public bool ChasePlayers { get; set; } = false;
    public float ChaseGapMeters { get; set; } = 2f;
    // The model's time gap (drift_env chase_time_gap_blind_s: a player's plan is unknown), how far
    // ahead its acceleration becomes the pace asked (chase_response_s), and the top of that pace as
    // a percentage of the policy's top pace (chase_pace_max): a chaser needs some speed over the
    // player's to catch up.
    public float ChaseTimeGapSeconds { get; set; } = 0.2f;
    public float ChaseResponseSeconds { get; set; } = 0.6f;
    public float ChasePaceMaxPercent { get; set; } = 110f;

    // Tandem trains: bots in order of their entry, this many to a train. The first of each train is
    // driven by the policy; the others replay what it sent a moment earlier (TandemTrains.cs), so
    // they follow its line exactly. 1 = every bot drives alone. With HoldBotSpacing the trains are
    // spread round the lap, each as one.
    public int TrainSize { get; set; } = 1;

    // Seconds behind the car in front, per place in the train (0.35 is a little over a car length
    // at drift speed), varied by up to this share either way, slowly and differently for each car.
    public float TandemGapSeconds { get; set; } = 0.35f;
    public float TandemGapVariation { get; set; } = 0.25f;

    // Never closer than this to the car in front, nose to nose (a car is about 4.4 m long).
    public float TandemMinGapMeters { get; set; } = 4.8f;

    // How far a follower sits to one side of the lead's line at most; it wanders between about a
    // fifth of this and all of it, on a side of its own.
    public float TandemLateralOffsetMeters { get; set; } = 0.6f;

    // ---- Telemetry ----------------------------------------------------------------------

    // Writes drift_physics_telemetry_<date>_<time>.csv (one row per AI car per tick, a new file
    // for every server start) into the server's working directory. drift_rl/check_plugin_policy.py and replay_live.py read it.
    // Off by default: the plugin writes no files unless asked.
    public bool EnableTelemetryLogging { get; set; } = false;

    // The telemetry file stops growing at this size (about 110 MB an hour with six bots and a
    // player; 0 = no limit). It also stops, with a line in the log, if writing it fails.
    public float TelemetryMaxFileMegabytes { get; set; } = 500f;

    // Adds the network's inputs to every row (about 330 bytes). Both tools need them.
    public bool LogLearnedPolicyObservations { get; set; } = true;

    // The plugin's routine lines in the server log as well as its warnings: what it loaded, each
    // setting the policy file brings, each bot car's numbers, spawns and placements, and the timing
    // line every 60 s (unless LogTimingIntervalSeconds says otherwise). Off: warnings only.
    public bool VerboseLogging { get; set; } = false;

    // Show each bot in game under its entry list DRIVERNAME (and this country code's flag, e.g.
    // ITA; empty = none) instead of AssettoServer's "<NamePrefix> <slot>". A bot with no
    // DRIVERNAME keeps AssettoServer's name.
    public bool ShowEntryListNames { get; set; } = true;
    public string BotNation { get; set; } = "";

    // Every this many seconds, a line in the server log with the time the plugin took on the
    // server's update thread: milliseconds a tick, split into the network, the sensors (rays,
    // ground and wall checks), telemetry and the tandem followers, the worst tick, and the share
    // of one CPU core. 0 = off.
    public float LogTimingIntervalSeconds { get; set; } = 0f;

    // Adds to the telemetry what the first connected player's client sends for its own car:
    // position, steering, engine RPM, gear, throttle and the four wheel speeds (no name, IP or
    // Steam ID). How the two visuals' encodings were measured, and judge.py's human driver.
    public bool LogHumanCarForCalibration { get; set; } = false;

    // ---- Steering visual (cosmetic) -------------------------------------------------------

    // Sends the steering angle the vehicle model used to clients, so the front wheels and the
    // steering wheel turn on screen (the stock AI always sends "straight"). Changes nothing
    // about how the car drives.
    public bool EnableSteeringVisual { get; set; } = true;

    // Encoding of the road wheels' byte around its centre value 127, as measured from a human
    // driver's client (HISTORY.md): the sign, and byte units per degree. The steering wheel's
    // byte reaches its limit at full lock.
    public float SteeringVisualSign { get; set; } = 1f;
    public float SteeringVisualWheelByteUnitsPerDegree { get; set; } = 2f;

    // Fastest the displayed angle may change. The policy can go lock to lock in one tick,
    // which would show as the wheels snapping. 0 = no limit.
    public float SteeringVisualMaxRateDegS { get; set; } = 360f;

    // ---- Drivetrain visual (cosmetic) ------------------------------------------------------

    // Sends wheel speeds, engine RPM, gear and throttle that belong to THIS car, so it looks
    // and sounds like what it is doing. (The stock AI sends the same speed for all four
    // wheels, an RPM between its idle and maximum settings, first gear and no throttle, all
    // worked out from where its own spline follower would be.) Changes nothing about how the
    // car drives.
    //   Front wheels: the speed of their axle along the way they point.
    //   Rear wheels: the car's forward speed plus the wheelspin the tire model implies. A sliding
    //     tire's force points against its tread's slip over the road, so thrust and sideways
    //     force in a given ratio mean lengthwise and sideways slip in that ratio.
    //     Never faster than the rev limiter allows in the gear the car is in.
    //   Engine RPM: rear wheel speed through the gearbox below.
    // The numbers come from a human driver's recording on this course (HISTORY.md).
    public bool EnableDrivetrainVisual { get; set; } = true;

    // The car's gearbox (drivetrain.ini GEAR_1.. and FINAL), rev range (engine.ini MINIMUM
    // and LIMITER) and driven-tire diameter (tyres.ini RADIUS x 2). Defaults: the Gravy
    // Garage Miata, the car the vehicle model above is.
    public List<float> DrivetrainVisualGearRatios { get; set; } = new() { 3.321f, 1.902f, 1.408f, 1.150f, 0.889f };
    public float DrivetrainVisualFinalDrive { get; set; } = 3.8f;
    public float DrivetrainVisualIdleRpm { get; set; } = 1000f;
    public float DrivetrainVisualRevLimitRpm { get; set; } = 8750f;
    public float DrivetrainVisualTyreDiameterMeters { get; set; } = 0.52f;

    // Gear changes go by the car's forward speed, not by wheelspin: up when that speed alone
    // would have this gear above the first, down when it would have the next gear down below
    // the second. A drift car holds its gear and sits on the limiter rather than change up
    // mid-slide: a gear higher, the engine is off boost at road speed and cannot keep the
    // rear tires spinning (HISTORY.md).
    public float DrivetrainVisualUpshiftRpm { get; set; } = 6000f;
    public float DrivetrainVisualDownshiftRpm { get; set; } = 2500f;

    // How quickly the rear wheels (and with them the engine) follow a change in wheelspin.
    public float DrivetrainVisualWheelSpeedLagSeconds { get; set; } = 0.15f;

    // ---- Body lean (cosmetic) -------------------------------------------------------------

    // How far the body leans on its suspension, degrees per g: outward in a turn and nose up
    // under power, on top of the slope of the ground under the car. Measured on a
    // human-driven car of this family and scaled by weight, height and spring rates to the
    // car above (HISTORY.md). 0 = none.
    public float BodyRollDegPerG { get; set; } = 0.85f;
    public float BodyPitchDegPerG { get; set; } = 0.57f;

    // ---- Wrong-way recovery ---------------------------------------------------------------

    // The one safety net. The vehicle model holds forward speed while lateral velocity is
    // free, so a car that is rotating faster than its tires can stop never slows down on its
    // own. When the car points more than WrongWayEntryThresholdDeg away from the racing
    // line's direction for WrongWayEntryTicks ticks in a row, yaw rate and lateral velocity
    // are decayed (time constant below) until the car is neither sliding (body slip under
    // WrongWayExitSlipDeg) nor rotating (yaw rate under WrongWayExitYawRateDegS). A held
    // drift stays well inside the threshold. The policy is trained with this in place.
    public bool EnableWrongWayRecovery { get; set; } = true;
    public float WrongWayEntryThresholdDeg { get; set; } = 67f;
    public int WrongWayEntryTicks { get; set; } = 3;
    public float WrongWayExitSlipDeg { get; set; } = 30f;
    public float WrongWayExitYawRateDegS { get; set; } = 20f;
    public float WrongWayRecoveryTimeConstantSeconds { get; set; } = 0.3f;

    // ---- The car ----------------------------------------------------------------------------
    // A single-track ("bicycle") model: yaw rate and lateral velocity are integrated from the
    // lateral tire forces on two axles. Each axle's force rises linearly with slip angle up to
    // its peak. The front then stays at the peak; the rear falls to RearSlidingGrip x the peak.

    public float YawInertiaKgM2 { get; set; } = 2199.6f;
    public float DistanceCgToFrontAxleM { get; set; } = 1.035f;
    public float DistanceCgToRearAxleM { get; set; } = 1.265f;
    public float FrontStaticLoadN { get; set; } = 6132.6f;
    public float RearStaticLoadN { get; set; } = 5017.6f;
    public float LateralDynamicsFrontPeakFriction { get; set; } = 1.3745f;
    public float LateralDynamicsRearPeakFriction { get; set; } = 1.3385f;

    // Tire load sensitivity (a policy file trained with it turns it on, "EnableTireLoadSensitivity";
    // off otherwise, whatever is set here). The peak frictions above are a tire's grip with no
    // load on it (tyres.ini DY0); each kN on a tire takes TireLoadSensitivityPerKn off (DY1). An
    // axle's load is shared evenly at rest; cornering moves mass x sideways acceleration x
    // CgHeightMeters / track onto the outside tire, split between the axles by
    // RollStiffnessFrontShare. The numbers are the Miata's (tyres.ini, suspensions.ini); a policy
    // file can carry its own.
    public bool EnableTireLoadSensitivity { get; set; } = false;
    public float FrontTireLoadSensitivityPerKn { get; set; } = -0.044f;
    public float RearTireLoadSensitivityPerKn { get; set; } = -0.049f;
    public float FrontTrackMeters { get; set; } = 1.67f;
    public float RearTrackMeters { get; set; } = 1.50f;
    public float RollStiffnessFrontShare { get; set; } = 0.67f;
    public float CgHeightMeters { get; set; } = 0.40f;
    public float FrontPeakSlipAngleRadians { get; set; } = 11.7f * MathF.PI / 180f;
    public float RearPeakSlipAngleRadians { get; set; } = 12.0f * MathF.PI / 180f;
    // The steering lock at the road wheels. A policy file that was trained with another lock
    // says so ("plugin_settings": MaxSteerAngleDeg) and that value is used instead; the same
    // goes for MaxSteerRateDegS, the two rear limits and the wrong-way recovery's four thresholds.
    public float MaxSteerAngleRadians { get; set; } = 28.125f * MathF.PI / 180f;

    // How fast the road wheels can be steered, degrees per second. 0 = no limit: they are
    // wherever the policy says within one tick. A policy file trained with a limit sets it.
    public float MaxSteerRateDegS { get; set; } = 0f;

    // How good a driver the bots are, 0 - 1, for a policy trained with a skill input ("skill_range";
    // others ignore it): 1 is the best it can do; 0 a driver who would still be let out with others -
    // holds a drift and rarely spins, but with a shallower angle, a cautious line, slower and
    // sloppier hands. Kept inside the range the policy was trained for.
    public float BotSkill { get; set; } = 1f;

    // THE DIAL. Share of its peak sideways grip the rear axle keeps once it is sliding: the
    // force falls from the peak at RearPeakSlipAngleRadians to this share at twice that angle.
    // 1 = no loss. For a policy trained without the rear thrust limit, less grip means a
    // slower drift through the same corner. For one trained with it the speed barely moves
    // and less grip means more angle (it leaves more thrust to hold the slide with). Unlike the
    // other values in this section it may be changed without retraining, but only inside the
    // range the policy was trained over ("rear_sliding_grip_range" in the policy file; 1 to 1
    // for a policy without it). Outside that range the nearest end is used and a warning logged.
    // drift_rl/dial_table.py prints the speed and angle each setting gives for a policy.
    public float RearSlidingGrip { get; set; } = 0.5f;

    // Braking is limited to this even if the car's own AiDeceleration is higher. Acceleration
    // uses the car's AiAcceleration. Without the thrust limit below, driving force uses up
    // rear grip (friction ellipse).
    public float MaxLongitudinalDecelMs2 { get; set; } = 5.6f;

    // A sliding car loses forward speed (at lateral velocity x yaw rate) unless it is pushed,
    // the more the larger its slip angle. With this on, forward speed can only be held or
    // gained with the push that is actually there: what the friction circle leaves the rear
    // tires after their sideways force, plus the part of the front tires' sideways force that
    // points along the car. Without it a car with enough steering lock holds any angle at any
    // speed. Off for policies trained without it; a policy file trained with it turns it on
    // ("plugin_settings": EnableRearThrustLimit).
    public bool EnableRearThrustLimit { get; set; } = false;

    // The thrust limit's other side: the rear tires cannot hold the car back with more than
    // they could push it with. Only with EnableRearThrustLimit; set by the policy file.
    public bool EnableRearBrakeLimit { get; set; } = false;

    // The policy's speed output is a pedal instead of a target speed (throttle above zero,
    // brakes below), forward speed follows from the forces along the car, and the rear tires
    // share one friction circle between thrust and sideways grip: lift and a sliding rear axle
    // grips at its peak, full throttle leaves it RearSlidingGrip x the peak. Takes the place
    // of the two limits above. Set by the policy file.
    public bool EnableRearFrictionCircle { get; set; } = false;

    // With EnableRearFrictionCircle: the pedal cannot go below zero and the brake action does
    // nothing, so the car sheds speed only by lifting and with its angle. Set by the policy file.
    public bool DisableBrakes { get; set; } = false;

    // With EnableRearFrictionCircle: the pedal cannot go below zero (no braking on all four
    // wheels), but the brake action - the front axle, with the throttle - still works, so the car
    // can scrub speed and keep the rear loose. Set by the policy file.
    public bool FrontBrakeOnly { get; set; } = false;

    // Floor on the speed in the slip-angle formula, which is singular at zero speed.
    public float MinSlipAngleSpeedMs { get; set; } = 3.0f;

    // The lateral dynamics are integrated in steps no longer than this; a server tick longer
    // than MaxRealDtSeconds is treated as that long.
    public float MaxLateralDynamicsSubStepSeconds { get; set; } = 0.05f;
    public float MaxRealDtSeconds { get; set; } = 0.2f;

    // ---- The racing line --------------------------------------------------------------------

    // Bounds how fast the line's direction at the car may change from one tick to the next
    // (see GlitchCorrector): twice this lateral acceleration at the car's current speed.
    public float MaxLateralAccelMs2 { get; set; } = 13.3f;

    // ---- Sensors ----------------------------------------------------------------------------

    // Horizontal ray fan centred on the car's direction of travel. The policy reads 15 rays.
    public int RaycastFanRayCount { get; set; } = 15;
    public float RaycastFanAngleDegrees { get; set; } = 120f;
    public float RaycastFanMaxDistanceMeters { get; set; } = 60f;
    public float RaycastHeightOffsetMeters { get; set; } = 0.5f;

    // Six short rays fixed to the chassis (see ProximityPins).
    public float ProximityPinMaxDistanceMeters { get; set; } = 12f;
    public float ProximityPinHeightOffsetMeters { get; set; } = 0.15f;

    // Cell size of the ray-cast grid, as triangles per cell. Speed only.
    public int RaycastGridTargetTrianglesPerCell { get; set; } = 8;

    // ---- Walls --------------------------------------------------------------------------------

    // A Wall triangle closer than the contact distance along a proximity pin pushes the car
    // back out to contact distance plus the margin and removes its velocity into the wall.
    public float CollisionContactDistanceMeters { get; set; } = 1.6f;
    public float CollisionPushOutMarginMeters { get; set; } = 0.3f;

    // Walls touch the car's own outline instead of the contact distance's circle: a rectangle
    // turned with the car's heading, reaching each axle's distance from the centre plus the
    // overhang, and half the wider track plus the side margin to each side (CarOutline). A
    // sliding car's tail reaches a wall long before its centre does. Set by the policy file.
    public bool CarOutlineContact { get; set; } = false;
    public float CarOutlineOverhangMeters { get; set; } = 0.8f;
    public float CarOutlineSideMarginMeters { get; set; } = 0.1f;

    // ---- Ride height ----------------------------------------------------------------------------

    // The car rides GroundRayHeightOffsetM above the ground straight below it (the first
    // non-wall triangle of what the game collides with), found with a ray that starts
    // GroundRayStartAboveMeters above the car and reaches GroundRayMaxDistanceMeters down. The sensors are cast from this height, so it matters
    // to the policy, not only to how the car looks.
    public float GroundRayHeightOffsetM { get; set; } = 0.40f;
    public float GroundRayStartAboveMeters { get; set; } = 1.5f;
    public float GroundRayMaxDistanceMeters { get; set; } = 6.0f;
    public float GroundRayHeightSmoothingFactor { get; set; } = 1.0f;
    // The ground the car rides on is the straight line best fitting the ground this far ahead of
    // and behind it along the way it is going, so bumps shorter than about twice this are
    // smoothed out while hills are followed exactly. 0 = the ground straight below only.
    public float RideHeightSpanMeters { get; set; } = 3.0f;
    // Where the ray finds nothing (a crack in the ground mesh), the last height is held this long
    // before the fallback below takes over.
    public float GroundRayMissHoldSeconds { get; set; } = 0.5f;

    // Only when that ray finds no ground: the racing line's own height, blended toward the
    // nearest road-mesh point (plus RoadMeshHeightOffsetM) as the car gets between
    // CrossTrackBlendStartM and CrossTrackBlendEndM from the line, if a mesh point lies within
    // MaxTrustedMeshDistanceM. Smoothed by HeightSmoothingFactor per tick unless the change
    // exceeds HeightSnapThresholdMeters.
    public float RoadMeshHeightOffsetM { get; set; } = 0.44f;
    public float CrossTrackBlendStartM { get; set; } = 5.0f;
    public float CrossTrackBlendEndM { get; set; } = 12.0f;
    public float MaxTrustedMeshDistanceM { get; set; } = 8.0f;
    public float HeightSmoothingFactor { get; set; } = 0.15f;
    public float HeightSnapThresholdMeters { get; set; } = 0.15f;

    // Pitch and roll follow the ground under the car and are smoothed by this much per tick.
    public float PitchRollSmoothingFactor { get; set; } = 0.5f;
}
