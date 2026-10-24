using AssettoServer.Network.Tcp;
using AssettoServer.Server;
using AssettoServer.Server.Configuration;
using AssettoServer.Shared.Network.Packets.Outgoing;
using Serilog;

namespace GradientDriftPlugin;

// Shows each bot in game under its entry-list DRIVERNAME. AssettoServer names every AI car
// "<NamePrefix> <slot>" ("Traffic 16") and sends that in two places: to everyone when a slot
// becomes an AI car, and in the driver list a player gets on joining. This sends the entry
// list's name right after each: to every player when a bot's slot turns AI, and to a joining
// player once their first update (with the driver list) has gone out. A bot with no DRIVERNAME
// keeps AssettoServer's name.
internal sealed class BotNames
{
    private readonly EntryCarManager _entryCarManager;
    private readonly ACServerConfiguration _serverConfiguration;
    private readonly string _nation;
    private string?[]? _names;          // per slot, from the first tick (the entry cars exist by then)
    private bool[] _wasAi = Array.Empty<bool>();

    public BotNames(EntryCarManager entryCarManager, ACServerConfiguration serverConfiguration, string nation)
    {
        _entryCarManager = entryCarManager;
        _serverConfiguration = serverConfiguration;
        _nation = nation;
        entryCarManager.ClientConnected += (client, _) => client.FirstUpdateSent += OnFirstUpdateSent;
    }

    private string?[] Names()
    {
        if (_names != null)
        {
            return _names;
        }
        EntryCar[] cars = _entryCarManager.EntryCars;
        var names = new string?[cars.Length];
        for (int i = 0; i < cars.Length && i < _serverConfiguration.EntryList.Cars.Count; i++)
        {
            string? name = _serverConfiguration.EntryList.Cars[i].DriverName?.Trim();
            if (cars[i].AiMode != AiMode.None && !string.IsNullOrEmpty(name))
            {
                names[i] = name;
            }
        }
        _wasAi = new bool[cars.Length];
        return _names = names;
    }

    // Each tick: a slot that has just become an AI car was announced under AssettoServer's name.
    public void Update()
    {
        EntryCar[] cars = _entryCarManager.EntryCars;
        string?[] names = Names();
        for (int i = 0; i < cars.Length && i < names.Length; i++)
        {
            bool ai = cars[i].AiControlled;
            if (ai && !_wasAi[i] && names[i] != null)
            {
                _entryCarManager.BroadcastPacket(Packet(cars[i]));
            }
            _wasAi[i] = ai;
        }
    }

    private void OnFirstUpdateSent(ACTcpClient client, EventArgs args)
    {
        try
        {
            EntryCar[] cars = _entryCarManager.EntryCars;
            string?[]? names = _names;      // none sent before the first tick has read them
            for (int i = 0; names != null && i < cars.Length && i < names.Length; i++)
            {
                if (cars[i].AiControlled && names[i] != null)
                {
                    client.SendPacket(Packet(cars[i]));
                }
            }
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "GradientDrift: could not send the bots' names to {Client}", client.Name);
        }
    }

    private CarConnected Packet(EntryCar car) =>
        new() { SessionId = car.SessionId, Name = _names![car.SessionId], Nation = _nation };
}
