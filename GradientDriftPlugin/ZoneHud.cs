namespace GradientDriftPlugin;

// The zone score HUD for players with CSP: a CSP online script (sent like DriftZones' paint) that
// asks the server for the player's numbers a few times a second (ZoneScoreController) and draws
//   - a panel: an angle gauge (needle and tier colours), the zone the car is in with a live fill
//     bar and its points so far, speed against the track's pace, the run so far (counting up),
//     the player's best and place, and the server's top drivers. Its edge glows in the zones'
//     yellow while the car is in a zone. Drag it with the left mouse button, size it with the
//     wheel; remembered on the player's PC;
//   - a popup as each zone is judged: a grade word (PERFECT / GREAT / GOOD / OK) and points that
//     pop in, with line / angle / speed / tandem under it; a shaking red one for a spin or a wall
//     hit; a run's score, and NEW BEST when it beats the player's best.
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
        local clock = 0
        local shownRun, shownFill, shownAngle = 0, 0, 0   -- eased toward the server's numbers
        local glow = 0                                     -- the panel edge, while in a zone

        local yellow = rgbm(0.96, 0.77, 0, 1)
        local gold = rgbm(1, 0.84, 0.2, 1)
        local green = rgbm(0.4, 0.92, 0.45, 1)
        local cyan = rgbm(0.3, 0.85, 1, 1)
        local white = rgbm(1, 1, 1, 1)
        local grey = rgbm(0.62, 0.62, 0.66, 1)
        local red = rgbm(1, 0.3, 0.28, 1)
        local back = rgbm(0.04, 0.04, 0.06, 0.72)
        local track = rgbm(1, 1, 1, 0.12)

        local function fields(line)
          local f = {}
          for x in (line .. '\t'):gmatch('([^\t]*)\t') do f[#f + 1] = x end
          return f
        end

        local function pct(x) return string.format('%d%%', math.floor(tonumber(x) * 100 + 0.5)) end
        local function alpha(c, a) return rgbm(c.r, c.g, c.b, c.mult * a) end
        local function mix(a, b, t) return rgbm(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, 1) end
        local function ease(now, target, rate, dt) return now + (target - now) * math.min(1, rate * dt) end

        -- colour for a share 0..1: grey, white, green, gold
        local function grade(x)
          if x >= 0.75 then return gold elseif x >= 0.5 then return green elseif x >= 0.25 then return white end
          return grey
        end
        local function angleColour(a)
          if a >= 45 then return gold elseif a >= 30 then return green elseif a >= 20 then return white end
          return grey
        end

        local function popup(p)
          p.age = 0
          table.insert(popups, 1, p)
          while #popups > 4 do table.remove(popups) end
        end

        local function told(f)
          local kind, zone, points = f[3], f[4], tonumber(f[5])
          local parts = 'LINE ' .. pct(f[6]) .. '    ANGLE ' .. string.format('%.0f°', tonumber(f[7])) .. '    SPEED ' .. pct(f[8])
          if tonumber(f[9]) > 0.005 then parts = parts .. '    TANDEM ' .. pct(f[9]) end
          if kind == 'zone' then
            local word = points >= 90 and 'PERFECT' or points >= 75 and 'GREAT' or points >= 50 and 'GOOD' or 'OK'
            popup({ word = word, text = string.format('+%.0f', points), sub = 'ZONE ' .. zone .. '    ' .. parts,
              colour = grade(points / 100), big = points >= 75 })
          elseif kind == 'spin' then
            popup({ word = 'SPIN', text = 'ZONE ' .. zone .. '  0', colour = red, shake = true })
          elseif kind == 'wall' then
            popup({ word = 'WALL', text = string.format('%.0f', points), colour = red, shake = true })
          elseif kind == 'run' then
            local best = live and live.rank > 0 and points > live.best
            popup({ word = best and 'NEW BEST' or 'RUN', text = string.format('%.1f', points), sub = parts,
              colour = best and gold or cyan, big = true, rainbow = best })
            ui.toast(ui.Icons.Confirm, string.format('Zone run: %.1f points%s', points, best and ' - new best!' or ''))
          elseif kind == 'ended' then
            popup({ word = 'RUN ENDED', text = f[10], colour = grey, small = true })
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
          clock = clock + dt
          for i = #popups, 1, -1 do
            popups[i].age = popups[i].age + dt
            if popups[i].age > 4 then table.remove(popups, i) end
          end
          if live then
            shownRun = ease(shownRun, live.running and live.run or 0, 6, dt)
            shownFill = ease(shownFill, live.zone > 0 and live.fill or 0, 10, dt)
            shownAngle = ease(shownAngle, live.kmh >= 25 and live.angle or 0, 12, dt)
            glow = ease(glow, live.zone > 0 and 1 or 0, 5, dt)
          end
          untilPoll = untilPoll - dt
          if untilPoll <= 0 then
            untilPoll = 0.25
            poll()
          end
        end

        -- the panel can be dragged with the left mouse button and sized with the wheel over it,
        -- remembered on this PC (ac.store / ac.load, as other server HUDs do)
        local panel = nil
        local scale = 1
        local dragFrom = nil
        if ac.load then
          local px, py, ps = ac.load('gdZonesPanelX'), ac.load('gdZonesPanelY'), ac.load('gdZonesPanelScale')
          if px and py then panel = vec2(px, py) end
          if ps and ps >= 0.6 and ps <= 2 then scale = ps end
        end
        local function remember()
          if not ac.store then return end
          ac.store('gdZonesPanelX', panel.x)
          ac.store('gdZonesPanelY', panel.y)
          ac.store('gdZonesPanelScale', scale)
        end

        local function text(s, size, pos, colour)
          ui.dwriteDrawText(s, size, pos, colour)
        end
        local function centred(s, size, x, y, colour)
          local w = ui.measureDWriteText(s, size).x
          ui.dwriteDrawText(s, size, vec2(x - w / 2, y), colour)
        end

        local function drawPopups(screen)
          local y = 140
          for _, p in ipairs(popups) do
            local a = math.min(1, (4 - p.age) * 1.5)
            -- pop in: overshoot then settle; shake for a spin or a wall
            local pop = p.age < 0.25 and 1 + 0.35 * math.sin(p.age / 0.25 * math.pi) or 1
            local dx = p.shake and p.age < 0.5 and math.sin(p.age * 70) * 9 * (1 - p.age / 0.5) or 0
            local colour = p.colour
            if p.rainbow then colour = mix(gold, cyan, 0.5 + 0.5 * math.sin(clock * 6)) end
            local wordSize = (p.small and 16 or 20) * pop
            local size = (p.small and 20 or p.big and 46 or 36) * pop
            centred(p.word, wordSize, screen.x / 2 + dx + 2, y + 2, rgbm(0, 0, 0, 0.6 * a))
            centred(p.word, wordSize, screen.x / 2 + dx, y, alpha(colour, a))
            y = y + wordSize + 2
            centred(p.text, size, screen.x / 2 + dx + 2, y + 2, rgbm(0, 0, 0, 0.6 * a))
            centred(p.text, size, screen.x / 2 + dx, y, alpha(white, a))
            y = y + size + 4
            if p.sub then
              centred(p.sub, 15, screen.x / 2, y, rgbm(1, 1, 1, 0.8 * a))
              y = y + 22
            end
            y = y + 10
          end
        end

        -- an arc from angle a0 to a1 (radians, 0 = right, clockwise on screen) as short lines
        local function arc(c, r, a0, a1, colour, thick)
          local steps = math.max(2, math.floor(math.abs(a1 - a0) / 0.08))
          local last = vec2(c.x + math.cos(a0) * r, c.y + math.sin(a0) * r)
          for i = 1, steps do
            local t = a0 + (a1 - a0) * i / steps
            local p = vec2(c.x + math.cos(t) * r, c.y + math.sin(t) * r)
            ui.drawLine(last, p, colour, thick)
            last = p
          end
        end

        function script.drawUI()
          if not live then return end
          local screen = ac.getUI().windowSize
          drawPopups(screen)

          if not panel then panel = vec2(screen.x - 310, 96) end
          local k = scale
          local rows = math.min(#board, 5)
          local size = vec2(290, 214 + (rows > 0 and 24 + rows * 20 or 0)) * k
          local mouse = ui.mousePos()
          local over = mouse.x >= panel.x and mouse.x <= panel.x + size.x and mouse.y >= panel.y and mouse.y <= panel.y + size.y
          if ui.mouseDown(0) then
            if dragFrom then
              panel = panel + (mouse - dragFrom)
              dragFrom = mouse
            elseif over then
              dragFrom = mouse
            end
          elseif dragFrom then
            dragFrom = nil
            remember()
          end
          if over and ui.mouseWheel() ~= 0 then
            scale = math.max(0.6, math.min(2, scale + ui.mouseWheel() * 0.05))
            remember()
          end
          local function P(dx, dy) return vec2(panel.x + dx * k, panel.y + dy * k) end
          local function T(s, sz, dx, dy, colour) text(s, sz * k, P(dx, dy), colour) end

          -- body, the zones' yellow down the left, and an edge that glows (and breathes) in a zone
          ui.drawRectFilled(P(0, 0), P(290, size.y / k), back, 8 * k)
          ui.drawRectFilled(P(0, 0), P(4, size.y / k), yellow, 2 * k)
          if glow > 0.02 then
            local breathe = 0.55 + 0.45 * math.sin(clock * 5)
            ui.drawRect(P(0, 0), P(290, size.y / k), alpha(yellow, glow * breathe), 8 * k, nil, 2 * k)
          end
          T('DRIFT ZONES', 13, 16, 10, yellow)
          T(live.zones .. ' ZONES', 11, 222, 12, grey)

          -- angle gauge: a half ring from 0 to 60 degrees, tier marks at 20, 30 and 45, a needle
          local c, r = P(78, 104), 52 * k
          local function at(deg) return math.pi + math.min(deg, 60) / 60 * math.pi end
          arc(c, r, math.pi, 2 * math.pi, track, 7 * k)
          if shownAngle > 0.5 then arc(c, r, math.pi, at(shownAngle), angleColour(shownAngle), 7 * k) end
          for _, m in ipairs({ 20, 30, 45 }) do
            local t = at(m)
            ui.drawLine(vec2(c.x + math.cos(t) * (r - 9 * k), c.y + math.sin(t) * (r - 9 * k)),
              vec2(c.x + math.cos(t) * (r + 5 * k), c.y + math.sin(t) * (r + 5 * k)), rgbm(1, 1, 1, 0.5), 1.5 * k)
          end
          local n = at(shownAngle)
          ui.drawLine(c, vec2(c.x + math.cos(n) * (r - 4 * k), c.y + math.sin(n) * (r - 4 * k)), white, 2.5 * k)
          ui.drawCircleFilled(c, 4 * k, white, 12)
          centred(string.format('%.0f°', shownAngle), 26 * k, c.x, c.y - 34 * k, angleColour(shownAngle))
          centred('ANGLE', 10 * k, c.x, c.y + 8 * k, grey)

          -- speed, right of the gauge
          T(string.format('%.0f', live.kmh), 30, 160, 54, white)
          T('KM/H', 10, 162, 88, grey)
          local pace = math.min(1.2, live.ofPace)
          ui.drawRectFilled(P(160, 104), P(274, 110), track, 3 * k)
          ui.drawRectFilled(P(160, 104), P(160 + 114 * math.min(1, pace), 110), grade((pace - 0.6) / 0.4), 3 * k)
          T('OF PACE ' .. pct(live.ofPace), 10, 160, 114, grey)

          -- the zone the car is in: a fill bar and the points so far
          local y = 136
          if live.zone > 0 then
            T('ZONE ' .. live.zone, 14, 16, y, yellow)
            T(string.format('%.0f PTS', live.points), 14, 200, y, white)
            ui.drawRectFilled(P(16, y + 22), P(274, y + 32), track, 4 * k)
            ui.drawRectFilled(P(16, y + 22), P(16 + 258 * math.min(1, shownFill), y + 32), grade(shownFill), 4 * k)
            T('LINE ' .. pct(shownFill), 10, 16, y + 35, grey)
          else
            T(live.running and string.format('RUN  %d/%d ZONES', live.runZones, live.zones) or 'NEXT RUN STARTS AT ZONE 1', 13, 16, y + 8, grey)
          end

          -- the run (counting up) and the player's best
          y = 186
          T('RUN', 10, 16, y, grey)
          T(live.running and string.format('%.1f', shownRun) or '-', 20, 16, y + 10, live.running and cyan or grey)
          T('BEST', 10, 120, y, grey)
          T(live.rank > 0 and string.format('%.1f', live.best) or '-', 20, 120, y + 10, gold)
          if live.rank > 0 then T('#' .. live.rank, 20, 222, y + 10, white) end

          -- the server's top five
          if rows > 0 then
            y = 222
            ui.drawLine(P(16, y), P(274, y), track, 1)
            T('SERVER TOP', 10, 16, y + 6, yellow)
            for i = 1, rows do
              local b = board[i]
              local yy = y + 6 + i * 20
              local colour = i == 1 and gold or white
              T(b.rank, 13, 16, yy, colour)
              T(#b.driver > 22 and b.driver:sub(1, 21) .. '…' or b.driver, 13, 40, yy, colour)
              T(string.format('%.1f', b.score), 13, 230, yy, colour)
            end
          end
        end
        """ + $"\n-- {zoneCount} zones\n";
}
