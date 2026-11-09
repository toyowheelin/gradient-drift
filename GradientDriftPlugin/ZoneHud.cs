namespace GradientDriftPlugin;

// The zone score HUD for players with CSP: a CSP online script (sent like DriftZones' paint) that
// asks the server for the player's numbers a few times a second (ZoneScoreController) and draws
//   - a panel: the zone the car is in and its points so far, the slip angle and speed right now,
//     the run so far, the player's best and place, and the server's top drivers;
//   - a popup as each zone is judged (its points, and line / angle / speed / tandem), for a spin
//     or a wall hit, and for a run's score.
// All scoring is the server's (ZoneJudge); the script only shows it.
public static class ZoneHud
{
    public const string ScriptName = "gradientdrift_zone_hud.lua";

    public static string Script(int zoneCount) => """
        -- Gradient Drift: drift zone scores, sent by the GradientDrift server plugin
        -- (https://github.com/toyowheelin/gradient-drift, GradientDriftPlugin/ZoneHud.cs). The server judges;
        -- this only shows what it says.
        local base = 'http://' .. ac.getServerIP() .. ':' .. ac.getServerPortHTTP() .. '/gradientdrift/'
        local me = ac.getCar(0)
        local live = nil
        local board = {}
        local boardVersion = -1
        local since = -1
        local popups = {}
        local waiting, waitingBoard = false, false
        local untilPoll = 0

        local gold = rgbm(0.96, 0.77, 0, 1)
        local green = rgbm(0.45, 0.9, 0.4, 1)
        local white = rgbm(1, 1, 1, 1)
        local grey = rgbm(0.7, 0.7, 0.7, 1)
        local red = rgbm(1, 0.35, 0.3, 1)
        local back = rgbm(0, 0, 0, 0.55)

        local function fields(line)
          local f = {}
          for x in (line .. '\t'):gmatch('([^\t]*)\t') do f[#f + 1] = x end
          return f
        end

        local function pct(x) return string.format('%d%%', math.floor(tonumber(x) * 100 + 0.5)) end

        local function popup(text, sub, colour)
          table.insert(popups, 1, { text = text, sub = sub, colour = colour, age = 0 })
          while #popups > 4 do table.remove(popups) end
        end

        local function told(f)
          local kind, zone, points = f[3], f[4], tonumber(f[5])
          local parts = 'Line ' .. pct(f[6]) .. '   Angle ' .. string.format('%.0f°', tonumber(f[7])) .. '   Speed ' .. pct(f[8])
          if tonumber(f[9]) > 0.005 then parts = parts .. '   Tandem ' .. pct(f[9]) end
          if kind == 'zone' then
            popup(string.format('ZONE %s   +%.0f', zone, points), parts, points >= 75 and gold or points >= 40 and green or white)
          elseif kind == 'spin' then
            popup('ZONE ' .. zone .. '   SPIN   0', nil, red)
          elseif kind == 'wall' then
            popup(string.format('WALL   %.0f', points), nil, red)
          elseif kind == 'run' then
            popup(string.format('RUN   %.1f', points), parts, gold)
            ui.toast(ui.Icons.Confirm, string.format('Zone run: %.1f points', points))
          elseif kind == 'ended' then
            popup('Run ended: ' .. f[10], nil, grey)
          end
        end

        local function getBoard()
          if waitingBoard then return end
          waitingBoard = true
          web.get(base .. 'board', function (err, response)
            waitingBoard = false
            if err or not response or response.status ~= 200 then return end
            local b = {}
            for line in response.body:gmatch('[^\n]+') do
              local f = fields(line)
              if f[1] == 'B' then b[#b + 1] = { rank = f[2], driver = f[3], score = tonumber(f[5]) } end
            end
            board = b
          end)
        end

        local function poll()
          if waiting then return end
          waiting = true
          web.get(base .. 'zones/' .. me.sessionID .. '?since=' .. math.max(since, 0), function (err, response)
            waiting = false
            if err or not response or response.status ~= 200 then return end
            local first = since < 0
            for line in response.body:gmatch('[^\n]+') do
              local f = fields(line)
              if f[1] == 'L' then
                live = { running = f[2] == '1', zone = tonumber(f[3]), zones = tonumber(f[4]), fill = tonumber(f[5]),
                  points = tonumber(f[6]), angle = tonumber(f[7]), kmh = tonumber(f[8]), ofPace = tonumber(f[9]),
                  run = tonumber(f[10]), runZones = tonumber(f[11]), best = tonumber(f[12]), rank = tonumber(f[13]) }
                local v = tonumber(f[14])
                if v ~= boardVersion then boardVersion = v; getBoard() end
              elseif f[1] == 'E' then
                local seq = tonumber(f[2])
                if seq > since then
                  if not first then told(f) end
                  since = seq
                end
              end
            end
            if since < 0 then since = 0 end
          end)
        end

        function script.update(dt)
          for i = #popups, 1, -1 do
            popups[i].age = popups[i].age + dt
            if popups[i].age > 4 then table.remove(popups, i) end
          end
          untilPoll = untilPoll - dt
          if untilPoll <= 0 then
            untilPoll = 0.25
            poll()
          end
        end

        local function text(s, size, x, y, colour)
          ui.dwriteDrawText(s, size, vec2(x, y), colour)
        end

        function script.drawUI()
          if not live then return end
          local screen = ac.getUI().windowSize
          -- popups, top middle, newest on top, fading out
          local y = 150
          for _, p in ipairs(popups) do
            local a = math.min(1, 4 - p.age)
            local c = rgbm(p.colour.r, p.colour.g, p.colour.b, a)
            local w = ui.measureDWriteText(p.text, 34).x
            text(p.text, 34, screen.x / 2 - w / 2, y - math.min(p.age, 0.3) * 30, c)
            y = y + 44
            if p.sub then
              local sw = ui.measureDWriteText(p.sub, 17).x
              text(p.sub, 17, screen.x / 2 - sw / 2, y - 6, rgbm(1, 1, 1, 0.85 * a))
              y = y + 24
            end
          end
          -- the panel, top right
          local x0, y0, w = screen.x - 290, 110, 270
          local rows = 6 + math.min(#board, 5)
          ui.drawRectFilled(vec2(x0 - 10, y0 - 8), vec2(x0 + w, y0 + rows * 22 + 12), back, 6)
          text('DRIFT ZONES', 16, x0, y0, gold)
          local yy = y0 + 24
          if live.zone > 0 then
            text(string.format('Zone %d/%d   line %s   %.0f pts', live.zone, live.zones, pct(live.fill), live.points), 15, x0, yy, white)
          else
            text(live.running and string.format('Run: %d/%d zones', live.runZones, live.zones) or 'Next run starts at zone 1', 15, x0, yy, grey)
          end
          yy = yy + 22
          local angle = live.kmh >= 25 and live.angle or 0
          local ac_ = angle >= 45 and gold or angle >= 30 and green or angle >= 20 and white or grey
          text(string.format('Angle %.0f°', angle), 15, x0, yy, ac_)
          text(string.format('%.0f km/h  %s', live.kmh, pct(live.ofPace)), 15, x0 + 110, yy, white)
          yy = yy + 22
          text(live.running and string.format('Run %.1f   (%d/%d zones)', live.run, live.runZones, live.zones) or 'Run -', 15, x0, yy, white)
          yy = yy + 22
          text(live.rank > 0 and string.format('Best %.1f   #%d here', live.best, live.rank) or 'Best -', 15, x0, yy, white)
          yy = yy + 26
          if #board > 0 then
            text('SERVER TOP', 13, x0, yy, gold)
            yy = yy + 20
            for i = 1, math.min(#board, 5) do
              local b = board[i]
              text(b.rank .. '  ' .. b.driver, 14, x0, yy, white)
              text(string.format('%.1f', b.score), 14, x0 + w - 60, yy, white)
              yy = yy + 20
            end
          end
        end
        """ + $"\n-- {zoneCount} zones\n";
}
