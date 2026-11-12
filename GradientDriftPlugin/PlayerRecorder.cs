using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using AssettoServer.Server;
using Serilog;

namespace GradientDriftPlugin;

// RecordPlayers: every connected player's car, as their own client reports it, into
// gradientdrift_players_<date>_<time>.csv.gz in the server's working directory (a new file for
// every server start). What the drift_rl style catalogue needs to learn how people drift on this
// server: tools/players_to_monitor.py turns a file into the AI Car Monitor's format.
//
// A player shows only as a salted hash of their Steam ID (DriverId), the same on this server
// across sessions and server starts and meaningless anywhere else: the salt is made once, at
// random, into gradientdrift_players_salt.txt beside the files. No names, IPs or Steam IDs.
internal sealed class PlayerRecorder : IDisposable
{
    private const string Header =
        "ServerTimeMs,SessionId,DriverId,CarModel,PosX,PosY,PosZ,HeadingRad,VelX,VelY,VelZ,SpeedKmh," +
        "WheelAngleByte,SteerAngleByte,GasByte,Gear,EngineRpm,StatusFlag";
    private const string SaltFile = "gradientdrift_players_salt.txt";
    private const int FlushEveryRows = 2000;

    public string FilePath { get; } = $"gradientdrift_players_{DateTime.Now:yyyyMMdd_HHmmss}.csv.gz";

    private readonly StreamWriter _writer;
    private readonly long _maxBytes;
    private readonly long _intervalMs;
    private readonly byte[] _salt;
    private readonly Dictionary<ulong, string> _ids = new();
    private readonly StringBuilder _line = new(256);
    private long _bytes;
    private long _lastMs = long.MinValue;
    private int _rows;
    private bool _stopped;

    // rateHz: rows per player per second at most; maxBytes: uncompressed size the file stops at (0 = no limit)
    public PlayerRecorder(float rateHz, long maxBytes)
    {
        _maxBytes = maxBytes;
        _intervalMs = rateHz > 0f ? (long)MathF.Round(1000f / rateHz) : 0L;
        _salt = LoadSalt();
        var gzip = new GZipStream(File.Create(FilePath), CompressionLevel.Fastest);
        _writer = new StreamWriter(gzip, new UTF8Encoding(false)) { AutoFlush = false };
        _writer.WriteLine(Header);
        _bytes = Header.Length + 1;
    }

    // One row per connected player, at most every 1 / rateHz seconds. Read-only on the cars.
    public void Sample(EntryCarManager cars, long nowMs)
    {
        if (_stopped || nowMs - _lastMs < _intervalMs)
        {
            return;
        }
        _lastMs = nowMs;
        try
        {
            foreach (EntryCar car in cars.EntryCars)
            {
                ACTcpClient? client = car.Client;
                if (client == null || car.AiControlled || !client.HasSentFirstUpdate)
                {
                    continue;
                }
                var status = car.Status;
                Vector3 v = status.Velocity;
                StringBuilder line = _line;
                line.Clear();
                line.Append(nowMs).Append(',').Append(car.SessionId).Append(',').Append(IdFor(client.Guid)).Append(',');
                line.Append(Clean(car.Model)).Append(',');
                Num(line, status.Position.X, "F2"); Num(line, status.Position.Y, "F2"); Num(line, status.Position.Z, "F2");
                // the plugin's heading convention, atan2(z, x): the game's yaw plus 90 degrees
                Num(line, status.Rotation.X + MathF.PI / 2f, "F4");
                Num(line, v.X, "F2"); Num(line, v.Y, "F2"); Num(line, v.Z, "F2");
                Num(line, v.Length() * 3.6f, "F1");
                line.Append(status.WheelAngle).Append(',').Append(status.SteerAngle).Append(',').Append(status.Gas).Append(',');
                line.Append(status.Gear).Append(',').Append(status.EngineRpm).Append(',').Append((uint)status.StatusFlag);
                if (_maxBytes > 0 && _bytes + line.Length + 1 > _maxBytes)
                {
                    Stop($"the file reached RecordPlayersMaxMegabytes ({_maxBytes / (1024 * 1024)} MB)");
                    return;
                }
                _writer.WriteLine(line);
                _bytes += line.Length + 1;
                if (++_rows % FlushEveryRows == 0)
                {
                    _writer.Flush();
                }
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
            Stop(e.Message);
        }
    }

    private string IdFor(ulong guid)
    {
        if (!_ids.TryGetValue(guid, out string? id))
        {
            byte[] data = new byte[_salt.Length + 8];
            _salt.CopyTo(data, 0);
            BitConverter.TryWriteBytes(data.AsSpan(_salt.Length), guid);
            id = "p" + Convert.ToHexString(SHA256.HashData(data))[..12].ToLowerInvariant();
            _ids[guid] = id;
        }
        return id;
    }

    private static byte[] LoadSalt()
    {
        try
        {
            if (File.Exists(SaltFile))
            {
                byte[] saved = Convert.FromHexString(File.ReadAllText(SaltFile).Trim());
                if (saved.Length >= 16)
                {
                    return saved;
                }
            }
        }
        catch (FormatException)
        {
        }
        byte[] salt = RandomNumberGenerator.GetBytes(32);
        File.WriteAllText(SaltFile, Convert.ToHexString(salt) + Environment.NewLine);
        return salt;
    }

    private static string Clean(string? s) => (s ?? "").Replace(',', '_').Replace('\n', '_').Replace('\r', '_');

    private static void Num(StringBuilder line, float value, string format)
    {
        line.Append(float.IsFinite(value) ? value.ToString(format, CultureInfo.InvariantCulture) : "nan").Append(',');
    }

    private void Stop(string why)
    {
        if (_stopped)
        {
            return;
        }
        _stopped = true;
        Log.Warning("GradientDrift: player recording stopped: {Why}", why);
        try
        {
            _writer.Flush();
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
        }
    }

    public void Dispose()
    {
        _stopped = true;
        try
        {
            _writer.Dispose();
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
        }
    }
}
