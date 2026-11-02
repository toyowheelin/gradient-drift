using System.IO.Compression;
using Serilog;

namespace GradientDriftPlugin;

// The plugin's data files - the policy and the car files beside the plugin, and the track's data in
// the server's folder - read from one file, GradientDriftPlugin.bundle.zip beside the plugin, when it
// has them, else from the loose files where they have always been. The bundle holds them under the
// same names: the plugin folder's by their path in it (GradientDriftPlugin.policy.json,
// cars/<model>.json), the track's by their file name (<track>_triangles.csv.gz, ...), so a bundle
// can carry several tracks' data. Setup writes it. A loose file is read instead of the bundle's
// copy of it, so a policy copied in by hand still takes over (a warning at start names them).
//
// The bundle is read once for its list of files and then opened for each file as it is needed, so
// the server holds no lock on it between reads.
public static class PluginData
{
    public const string BundleFileName = "GradientDriftPlugin.bundle.zip";

    private static readonly object Lock = new();
    // the file's name as the plugin asks for it (cars/x.json) -> its name in the zip
    private static readonly Dictionary<string, string> Entries = new(StringComparer.OrdinalIgnoreCase);
    private static string _pluginDirectory = "";
    private static string? _bundlePath;

    public static void Init(string pluginDirectory)
    {
        _pluginDirectory = Path.TrimEndingDirectorySeparator(Path.GetFullPath(pluginDirectory));
        _bundlePath = null;
        Entries.Clear();
        string path = Path.Combine(_pluginDirectory, BundleFileName);
        if (!File.Exists(path))
        {
            return;
        }
        try
        {
            using ZipArchive zip = ZipFile.OpenRead(path);
            foreach (ZipArchiveEntry entry in zip.Entries)
            {
                if (entry.Name.Length > 0)
                {
                    Entries[entry.FullName.Replace('\\', '/')] = entry.FullName;
                }
            }
            _bundlePath = path;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Entries.Clear();
            Log.Warning(e, "GradientDrift: could not read {Path} - using the loose files instead", path);
            return;
        }

        string serverDirectory = Directory.GetCurrentDirectory();
        string[] shadowed = Entries.Keys
            .Where(name => File.Exists(Path.Combine(_pluginDirectory, name)) || (!name.Contains('/') && File.Exists(Path.Combine(serverDirectory, name))))
            .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (shadowed.Length > 0)
        {
            Log.Warning("GradientDrift: these loose files are read instead of their copies in {Bundle}: {Files}",
                BundleFileName, string.Join(", ", shadowed));
        }
    }

    public static bool Exists(string path) => File.Exists(path) || EntryFor(path) != null;

    // The loose file's contents, or the bundle's copy. IOException if it cannot be read.
    public static Stream Open(string path)
    {
        string? entryName = File.Exists(path) ? null : EntryFor(path);
        if (entryName == null)
        {
            return File.OpenRead(path);
        }
        lock (Lock)
        {
            try
            {
                using ZipArchive zip = ZipFile.OpenRead(_bundlePath!);
                ZipArchiveEntry entry = zip.GetEntry(entryName)
                    ?? throw new IOException($"{entryName} is no longer in {_bundlePath}");
                var copy = new MemoryStream();
                using (Stream stream = entry.Open())
                {
                    stream.CopyTo(copy);
                }
                copy.Position = 0;
                return copy;
            }
            catch (InvalidDataException e)
            {
                throw new IOException($"{_bundlePath} is damaged: {e.Message}", e);
            }
        }
    }

    public static string ReadAllText(string path)
    {
        using var reader = new StreamReader(Open(path));
        return reader.ReadToEnd();
    }

    public static IEnumerable<string> ReadLines(string path)
    {
        using var reader = new StreamReader(Open(path));
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            yield return line;
        }
    }

    // Where the file is read from, for the log: "<bundle> (name)" or the loose file's full path.
    public static string Describe(string path)
    {
        string? entryName = File.Exists(path) ? null : EntryFor(path);
        return entryName != null ? $"{_bundlePath} ({entryName})" : Path.GetFullPath(path);
    }

    // The bundle's name for a file the plugin asks for, or null: a file in the plugin's folder by its
    // path in it, one in the server's folder by its name.
    private static string? EntryFor(string path)
    {
        if (_bundlePath == null)
        {
            return null;
        }
        string full = Path.GetFullPath(path);
        string relative = Path.GetRelativePath(_pluginDirectory, full);
        string name;
        if (relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar) && !Path.IsPathRooted(relative))
        {
            name = relative.Replace('\\', '/');
        }
        else if (string.Equals(Path.GetDirectoryName(full),
                     Path.TrimEndingDirectorySeparator(Directory.GetCurrentDirectory()), StringComparison.OrdinalIgnoreCase))
        {
            name = Path.GetFileName(full);
        }
        else
        {
            return null;
        }
        return Entries.TryGetValue(name, out string? entryName) ? entryName : null;
    }
}
