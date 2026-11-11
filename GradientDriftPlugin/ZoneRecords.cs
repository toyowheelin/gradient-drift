using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Text;
using Serilog;

namespace GradientDriftPlugin;

// The server's zone run records: a small SQLite database file of its own on each server (beside
// its other files), so each server keeps its own leaderboard. Every scored run is kept; the board
// is each driver's best run on this track with these zones (an edited zone makes a new board, as
// old scores were for other zones).
//
// SQLite is the one the system has: winsqlite3.dll on Windows 10 / Server 2016 and later,
// libsqlite3 on Linux. No extra file ships with the plugin. Without it the scores still show in
// game and in chat; nothing is kept.
//
// Writes go to one background thread, so a slow disk never holds up the server's tick; the board
// is kept in memory for the game's HUD and the chat commands, refreshed after each write.
public sealed class ZoneRecords : IDisposable
{
    public sealed record Entry(int Rank, string Driver, string Car, float Score, float AngleDeg, float SpeedOfPace, string When);

    private readonly string _path;
    private readonly string _track;
    private readonly string _zones;
    private readonly int _size;
    private readonly BlockingCollection<Action> _work = new();
    private readonly Thread? _thread;
    private IntPtr _db;
    private volatile IReadOnlyList<Entry> _board = Array.Empty<Entry>();
    private readonly ConcurrentDictionary<string, (float Best, int Rank)> _personal = new();

    public bool Open => _db != IntPtr.Zero;
    private volatile int _version;
    public int Version => _version;
    public IReadOnlyList<Entry> Board => _board;

    // track: the track (and layout) name; zones: a fingerprint of the zones (DriftZones.Fingerprint).
    public ZoneRecords(string path, string track, string zones, int size)
    {
        _path = path;
        _track = track;
        _zones = zones;
        _size = Math.Max(1, size);
        try
        {
            Sqlite.Init();
            string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(folder)) Directory.CreateDirectory(folder);
            Check(Sqlite.sqlite3_open_v2(Encoding.UTF8.GetBytes(Path.GetFullPath(path) + "\0"), out _db,
                Sqlite.OpenReadWrite | Sqlite.OpenCreate | Sqlite.OpenFullMutex, IntPtr.Zero), "open");
            Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
            Exec("""
                CREATE TABLE IF NOT EXISTS runs (
                  id INTEGER PRIMARY KEY,
                  at TEXT NOT NULL,
                  track TEXT NOT NULL,
                  zones TEXT NOT NULL,
                  driver TEXT NOT NULL,
                  guid TEXT NOT NULL,
                  car TEXT NOT NULL,
                  score REAL NOT NULL,
                  angle REAL NOT NULL,
                  speed REAL NOT NULL,
                  fill REAL NOT NULL,
                  walls INTEGER NOT NULL,
                  zone_points TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_board ON runs (track, zones, guid, score);
                """);
            LoadBoard();
            _thread = new Thread(Work) { IsBackground = true, Name = "GradientDrift zone records" };
            _thread.Start();
            Log.Information("GradientDrift: zone run records in {Path} ({Count} drivers on this track's board)",
                Path.GetFullPath(path), _personal.Count);
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or IOException
                                      or UnauthorizedAccessException or InvalidOperationException)
        {
            Log.Warning("GradientDrift: zone runs are scored but not kept - no SQLite database at {Path} ({Message})",
                path, e.Message);
            Close();
        }
    }

    // A driver's best score and place on the board (0, 0 = none yet).
    public (float Best, int Rank) Personal(string guid) => _personal.TryGetValue(guid, out var p) ? p : (0f, 0);

    // Keeps a scored run; then (on the records thread) done(new best?, place, server record?).
    public void Add(string driver, string guid, string car, ZoneJudge.RunResult run, Action<bool, int, bool>? done)
    {
        if (!Open)
        {
            return;
        }
        string at = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss");
        _work.Add(() =>
        {
            float before = Personal(guid).Best;
            float top = _board.Count > 0 ? _board[0].Score : 0f;
            IntPtr st = Prepare("""
                INSERT INTO runs (at, track, zones, driver, guid, car, score, angle, speed, fill, walls, zone_points)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                """);
            try
            {
                BindText(st, 1, at);
                BindText(st, 2, _track);
                BindText(st, 3, _zones);
                BindText(st, 4, driver);
                BindText(st, 5, guid);
                BindText(st, 6, car);
                Sqlite.sqlite3_bind_double(st, 7, run.Score);
                Sqlite.sqlite3_bind_double(st, 8, run.MeanAngleDeg);
                Sqlite.sqlite3_bind_double(st, 9, run.MeanSpeedOfPace);
                Sqlite.sqlite3_bind_double(st, 10, run.MeanFill);
                Sqlite.sqlite3_bind_int(st, 11, run.Walls);
                BindText(st, 12, string.Join(" ", run.ZonePoints.Select(p => p.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture))));
                if (Sqlite.sqlite3_step(st) != Sqlite.Done) throw new InvalidOperationException(Error());
            }
            finally
            {
                Sqlite.sqlite3_finalize(st);
            }
            LoadBoard();
            int rank = Personal(guid).Rank;
            done?.Invoke(run.Score > before, rank, rank == 1 && run.Score > top);
        });
    }

    // Each driver's best run on this track with these zones, best first.
    private void LoadBoard()
    {
        IntPtr st = Prepare("""
            SELECT driver, car, score, angle, speed, at, guid FROM runs r
            WHERE track = ?1 AND zones = ?2 AND id = (
              SELECT id FROM runs b WHERE b.track = r.track AND b.zones = r.zones AND b.guid = r.guid
              ORDER BY score DESC, id ASC LIMIT 1)
            ORDER BY score DESC, id ASC
            """);
        var board = new List<Entry>();
        try
        {
            BindText(st, 1, _track);
            BindText(st, 2, _zones);
            int rank = 0;
            while (Sqlite.sqlite3_step(st) == Sqlite.Row)
            {
                rank++;
                float score = (float)Sqlite.sqlite3_column_double(st, 2);
                string guid = Text(st, 6);
                _personal[guid] = (score, rank);
                if (board.Count < _size)
                {
                    board.Add(new Entry(rank, Text(st, 0), Text(st, 1), score, (float)Sqlite.sqlite3_column_double(st, 3),
                        (float)Sqlite.sqlite3_column_double(st, 4), Text(st, 5)));
                }
            }
        }
        finally
        {
            Sqlite.sqlite3_finalize(st);
        }
        _board = board;
        _version++;
    }

    private void Work()
    {
        foreach (Action a in _work.GetConsumingEnumerable())
        {
            try
            {
                a();
            }
            catch (Exception e)
            {
                Log.Warning(e, "GradientDrift: could not keep a zone run in {Path}", _path);
            }
        }
        Close();
    }

    private void Exec(string sql)
    {
        if (Sqlite.sqlite3_exec(_db, Encoding.UTF8.GetBytes(sql + "\0"), IntPtr.Zero, IntPtr.Zero, out IntPtr err) != Sqlite.Ok)
        {
            string message = err != IntPtr.Zero ? Marshal.PtrToStringUTF8(err) ?? "" : Error();
            if (err != IntPtr.Zero) Sqlite.sqlite3_free(err);
            throw new InvalidOperationException(message);
        }
    }

    private IntPtr Prepare(string sql)
    {
        Check(Sqlite.sqlite3_prepare_v2(_db, Encoding.UTF8.GetBytes(sql + "\0"), -1, out IntPtr st, IntPtr.Zero), "prepare");
        return st;
    }

    private static void BindText(IntPtr st, int i, string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        Sqlite.sqlite3_bind_text(st, i, bytes, bytes.Length, Sqlite.Transient);
    }

    private static string Text(IntPtr st, int i)
    {
        IntPtr p = Sqlite.sqlite3_column_text(st, i);
        return p == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(p) ?? "";
    }

    private void Check(int code, string what)
    {
        if (code != Sqlite.Ok)
        {
            throw new InvalidOperationException($"SQLite {what}: {(_db != IntPtr.Zero ? Error() : "code " + code)}");
        }
    }

    private string Error() => Marshal.PtrToStringUTF8(Sqlite.sqlite3_errmsg(_db)) ?? "";

    private void Close()
    {
        if (_db != IntPtr.Zero)
        {
            Sqlite.sqlite3_close_v2(_db);
            _db = IntPtr.Zero;
        }
    }

    public void Dispose()
    {
        _work.CompleteAdding();
        if (_thread == null || !_thread.Join(TimeSpan.FromSeconds(5)))
        {
            Close();
        }
    }

    // The few SQLite calls the records need, from the system's own SQLite library.
    private static class Sqlite
    {
        private const string Library = "sqlite3";
        public const int Ok = 0, Row = 100, Done = 101;
        public const int OpenReadWrite = 0x2, OpenCreate = 0x4, OpenFullMutex = 0x10000;
        public static readonly IntPtr Transient = new(-1);
        private static int _init;

        // "sqlite3" is winsqlite3.dll on Windows, libsqlite3.so.0 on Linux, libsqlite3.dylib on a Mac
        public static void Init()
        {
            if (Interlocked.Exchange(ref _init, 1) == 1) return;
            NativeLibrary.SetDllImportResolver(typeof(Sqlite).Assembly, (name, assembly, path) =>
            {
                if (name != Library) return IntPtr.Zero;
                string[] names = OperatingSystem.IsWindows() ? new[] { "winsqlite3", "sqlite3", "e_sqlite3" }
                    : OperatingSystem.IsMacOS() ? new[] { "libsqlite3.dylib", "libsqlite3.0.dylib" }
                    : new[] { "libsqlite3.so.0", "libsqlite3.so", "libe_sqlite3.so" };
                foreach (string n in names)
                {
                    if (NativeLibrary.TryLoad(n, assembly, path, out IntPtr handle)) return handle;
                }
                return IntPtr.Zero;
            });
        }

        [DllImport(Library)] public static extern int sqlite3_open_v2(byte[] filename, out IntPtr db, int flags, IntPtr vfs);
        [DllImport(Library)] public static extern int sqlite3_close_v2(IntPtr db);
        [DllImport(Library)] public static extern int sqlite3_exec(IntPtr db, byte[] sql, IntPtr callback, IntPtr arg, out IntPtr err);
        [DllImport(Library)] public static extern void sqlite3_free(IntPtr p);
        [DllImport(Library)] public static extern IntPtr sqlite3_errmsg(IntPtr db);
        [DllImport(Library)] public static extern int sqlite3_prepare_v2(IntPtr db, byte[] sql, int bytes, out IntPtr st, IntPtr tail);
        [DllImport(Library)] public static extern int sqlite3_step(IntPtr st);
        [DllImport(Library)] public static extern int sqlite3_finalize(IntPtr st);
        [DllImport(Library)] public static extern int sqlite3_bind_text(IntPtr st, int i, byte[] text, int bytes, IntPtr destructor);
        [DllImport(Library)] public static extern int sqlite3_bind_double(IntPtr st, int i, double value);
        [DllImport(Library)] public static extern int sqlite3_bind_int(IntPtr st, int i, int value);
        [DllImport(Library)] public static extern IntPtr sqlite3_column_text(IntPtr st, int i);
        [DllImport(Library)] public static extern double sqlite3_column_double(IntPtr st, int i);
    }
}
