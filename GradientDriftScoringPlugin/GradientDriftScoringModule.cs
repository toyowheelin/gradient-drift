using AssettoServer.Server.Plugin;
using Autofac;

namespace GradientDriftScoring;

// Plugin entry point. The base class reads the "--- !GradientDriftScoringConfiguration" block of
// extra_cfg.yml and registers the resulting object; ZoneScoring falls back to the defaults when the
// block is not there.
public class GradientDriftScoringModule : AssettoServerModule<GradientDriftScoringConfiguration>
{
    protected override void Load(ContainerBuilder builder)
    {
        // the players' drift scores, the HUD, the server's board and the painted zones (ZoneScoring.cs)
        builder.RegisterType<ZoneScoring>().AsSelf().As<IAssettoServerAutostart>().SingleInstance();
    }
}
