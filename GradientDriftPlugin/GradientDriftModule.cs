using AssettoServer.Server.Plugin;
using Autofac;

namespace GradientDriftPlugin;

// Plugin entry point. The base class reads the "--- !GradientDriftConfiguration" block of
// extra_cfg.yml and registers the resulting object.
public class GradientDriftModule : AssettoServerModule<GradientDriftConfiguration>
{
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<GradientDriftPlugin>().AsSelf().As<IAssettoServerAutostart>().SingleInstance();
    }
}
