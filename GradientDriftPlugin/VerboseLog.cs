using Serilog;

namespace GradientDriftPlugin;

// The plugin's routine lines (what it loaded, each setting the policy file brings, each bot car's
// numbers, spawns and placements): written only with VerboseLogging on. The log is otherwise quiet
// unless something is wrong.
internal static class VerboseLog
{
    public static bool On;

    public static void Information(string template, params object?[] values)
    {
        if (On)
        {
            Log.Information(template, values);
        }
    }
}
