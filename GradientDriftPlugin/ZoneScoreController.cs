using Microsoft.AspNetCore.Mvc;

namespace GradientDriftPlugin;

// What the zone HUD (ZoneHud) asks the server for, on AssettoServer's HTTP port: one car's live
// numbers and latest scores, and the server's board. Plain tab-separated lines (ZoneScoring).
[ApiController]
public class ZoneScoreController : ControllerBase
{
    private readonly ZoneScoring _scoring;

    public ZoneScoreController(ZoneScoring scoring)
    {
        _scoring = scoring;
    }

    [HttpGet("/gradientdrift/zones/{car:int}")]
    public IActionResult Live(int car, [FromQuery] int since = 0)
    {
        if (!_scoring.On || car < 0 || car > 255) return NotFound();
        return Content(_scoring.LiveText((byte)car, since), "text/plain");
    }

    [HttpGet("/gradientdrift/board")]
    public IActionResult Board()
    {
        if (!_scoring.On) return NotFound();
        return Content(_scoring.BoardText(), "text/plain");
    }
}
