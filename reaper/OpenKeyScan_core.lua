-- Общий код скриптов OpenKeyScan для REAPER. Это не отдельное действие:
-- его подключают файлы «OpenKeyScan - ….lua», загружать в список действий не нужно.
--
-- Что делает: берёт исходные файлы выбранных элементов, отдаёт их консольной
-- OpenKeyScan (okscan-cli.exe) и озвучивает результат через OSARA.
-- Анализ идёт в отдельном процессе, REAPER при этом не замирает.

local M = {}

local TITLE = "OpenKeyScan"
-- Первый запуск после установки грузит модели дольше, поэтому запас щедрый.
local TIMEOUT_BASE, TIMEOUT_PER_FILE = 90, 30

local function say(msg)
  if reaper.osara_outputMessage then
    reaper.osara_outputMessage(msg)
  else
    -- Без OSARA сообщаем окном: его прочитает любой скринридер.
    reaper.MB(msg, TITLE, 0)
  end
end

-- Где искать okscan-cli.exe, по порядку:
-- 1) путь, сохранённый в ExtState (OpenKeyScan / cli_path);
-- 2) установленная программа в профиле пользователя;
-- 3) рядом со скриптом: копия скриптов в папке программы (reaper-scripts\..)
--    или папка проекта со свежей сборкой (reaper\..\dist\OpenKeyScan).
local function find_cli(script_dir)
  local default = (os.getenv("LOCALAPPDATA") or "") .. "\\Programs\\OpenKeyScan-Personal\\okscan-cli.exe"
  local candidates = { reaper.GetExtState(TITLE, "cli_path"), default }
  if script_dir then
    candidates[#candidates + 1] = script_dir .. "..\\okscan-cli.exe"
    candidates[#candidates + 1] = script_dir .. "..\\dist\\OpenKeyScan\\okscan-cli.exe"
  end
  for _, path in ipairs(candidates) do
    if path ~= "" and reaper.file_exists(path) then return path end
  end
  return nil, default
end

-- Исходный файл активного дубля. У обрезанных и развёрнутых элементов источник
-- обёрнут в «секцию», поэтому поднимаемся до самого верхнего родителя.
local function item_file(item)
  local take = reaper.GetActiveTake(item)
  if not take or reaper.TakeIsMIDI(take) then return nil end
  local src = reaper.GetMediaItemTake_Source(take)
  local parent = reaper.GetMediaSourceParent(src)
  while parent do
    src = parent
    parent = reaper.GetMediaSourceParent(src)
  end
  local file = reaper.GetMediaSourceFileName(src)
  if not file or file == "" then return nil end
  return file, reaper.GetTakeName(take)
end

local function split_tabs(line)
  local t = {}
  for field in (line .. "\t"):gmatch("(.-)\t") do t[#t + 1] = field end
  return t
end

-- Дописывает тональность в конец имени дорожки: «instr» -> «instr [D minor, 7A]».
-- Старая пометка такого же вида сначала убирается, чтобы при повторном запуске
-- скобки не копились. Дорожка с несколькими элементами получает тональность первого.
local function tag_track_names(items, results)
  local done, count = {}, 0
  for _, it in ipairs(items) do
    local r = results[it.file]
    local track = it.track
    if r and (r.err or "") == "" and r.camelot ~= "" and track and not done[track]
       and reaper.ValidatePtr2(0, track, "MediaTrack*") then
      done[track] = true
      local _, name = reaper.GetSetMediaTrackInfo_String(track, "P_NAME", "", false)
      -- «[что угодно, 7A]» в самом конце имени — наша прошлая пометка
      name = name:gsub("%s*%[[^%[%]]*%d+[AB]%]$", "")
      local tag = "[" .. r.key .. ", " .. r.camelot .. "]"
      reaper.GetSetMediaTrackInfo_String(track, "P_NAME", name ~= "" and (name .. " " .. tag) or tag, true)
      count = count + 1
    end
  end
  return count
end

-- Сдвигает элемент item целиком на delta секунд. То, что уезжает левее начала
-- проекта, подрезается (сдвигается начало звука внутри элемента), а не теряется.
local function shift_item(item, delta)
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION") + delta
  if pos >= 0 then
    reaper.SetMediaItemInfo_Value(item, "D_POSITION", pos)
    return false
  end
  local cut = -pos
  local len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
  if cut >= len then
    -- элемент целиком ушёл бы до начала проекта — оставляем его в нуле как есть
    reaper.SetMediaItemInfo_Value(item, "D_POSITION", 0)
    return false
  end
  for t = 0, reaper.CountTakes(item) - 1 do
    local take = reaper.GetTake(item, t)
    if take then
      local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
      local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
      reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", offs + cut * rate)
    end
  end
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH", len - cut)
  reaper.SetMediaItemInfo_Value(item, "D_POSITION", 0)
  return true
end

local function describe_shift(delta)
  local ms = math.floor(math.abs(delta) * 1000 + 0.5)
  local amount = ms < 1000 and (ms .. " мс") or (string.format("%.2f", math.abs(delta)) .. " с")
  return (delta > 0 and "вправо на " or "влево на ") .. amount
end

-- Ставит первую сильную долю бита (секунды от начала файла) на ближайшую линию
-- такта и сдвигает на столько же ВСЕ элементы проекта, чтобы ничего не разъехалось.
-- Возвращает фразу для озвучки.
local function align_project_to_bar(it, downbeat, bpm)
  if not downbeat then
    return "Первую сильную долю найти не удалось, элементы не сдвинуты"
  end
  local item = it.item
  if not reaper.ValidatePtr2(0, item, "MediaItem*") then
    return "Элемент пропал из проекта, выравнивание пропущено"
  end
  local take = reaper.GetActiveTake(item)
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  -- Если начало элемента обрезано дальше первой доли, берём первый такт внутри элемента.
  if downbeat < offs and bpm then
    local bar = 4 * 60 / bpm
    downbeat = downbeat + math.ceil((offs - downbeat) / bar) * bar
  end
  local beat_time = pos + (downbeat - offs) / rate

  -- ближайшая линия такта по текущей сетке проекта (темп уже выставлен выше)
  local _, measure = reaper.TimeMap2_timeToBeats(0, beat_time)
  local here = reaper.TimeMap2_beatsToTime(0, 0, measure)
  local nextbar = reaper.TimeMap2_beatsToTime(0, 0, measure + 1)
  local target = (beat_time - here <= nextbar - beat_time) and here or nextbar
  local delta = target - beat_time
  if math.abs(delta) < 0.0005 then
    return "Бит уже стоит по сетке"
  end

  local trimmed = false
  for i = 0, reaper.CountMediaItems(0) - 1 do
    if shift_item(reaper.GetMediaItem(0, i), delta) then trimmed = true end
  end
  return "Все элементы сдвинуты " .. describe_shift(delta) .. ", сильная доля бита на такте " ..
         (measure + (target == here and 1 or 2)) .. (trimmed and ", начало подрезано" or "")
end

local function busy_since()
  local v = tonumber(reaper.GetExtState(TITLE, "busy"))
  -- Метка старше 15 минут — остаток от прерванного запуска, её не считаем.
  if v and os.time() - v < 900 then return v end
  return nil
end

-- mode.set_tempo = true — после анализа выставить темп проекта по первому элементу.
-- mode.script_dir — папка скрипта (со слешем на конце), чтобы найти программу рядом.
function M.run(mode)
  if busy_since() then
    say("Анализ OpenKeyScan уже идёт, дождитесь результата.")
    return
  end

  local cli, expected = find_cli(mode and mode.script_dir)
  if not cli then
    say("Не найдена программа OpenKeyScan. Установите личную сборку OpenKeyScan, " ..
        "ожидаемый путь: " .. expected)
    return
  end

  local count = reaper.CountSelectedMediaItems(0)
  if count == 0 then
    say("Нет выбранных элементов. Выберите элемент со звуком и запустите действие снова.")
    return
  end

  local items, files, seen = {}, {}, {}
  for i = 0, count - 1 do
    local item = reaper.GetSelectedMediaItem(0, i)
    local file, name = item_file(item)
    if file then
      items[#items + 1] = { file = file, name = name, item = item, track = reaper.GetMediaItem_Track(item) }
      if not seen[file] then
        seen[file] = true
        files[#files + 1] = file
      end
    end
  end
  if #files == 0 then
    say("У выбранных элементов нет аудиофайлов: MIDI и пустые элементы не анализируются.")
    return
  end

  local dir = (os.getenv("TEMP") or reaper.GetResourcePath()) .. "\\OpenKeyScan-REAPER"
  reaper.RecursiveCreateDirectory(dir, 0)
  local stamp = tostring(os.time()) .. "_" .. tostring(math.random(100000, 999999))
  local list_path = dir .. "\\list_" .. stamp .. ".txt"
  local out_path = dir .. "\\out_" .. stamp .. ".tsv"

  local f = io.open(list_path, "w")
  if not f then
    say("Не удалось создать временный файл в папке " .. dir)
    return
  end
  f:write(table.concat(files, "\n"), "\n")
  f:close()

  -- -2: не ждать завершения и запустить окно консоли свёрнутым.
  reaper.ExecProcess('"' .. cli .. '" --list "' .. list_path .. '" --out "' .. out_path .. '"', -2)
  reaper.SetExtState(TITLE, "busy", tostring(os.time()), false)
  say(#files == 1 and "Анализирую." or ("Анализирую файлов: " .. #files .. "."))

  local started = reaper.time_precise()
  local timeout = TIMEOUT_BASE + TIMEOUT_PER_FILE * #files

  local function finish(text)
    reaper.DeleteExtState(TITLE, "busy", false)
    os.remove(list_path)
    os.remove(out_path)

    local results = {}
    for line in text:gmatch("[^\n]+") do
      local c = split_tabs(line)
      if c[1] == "ERROR" then
        say("Ошибка OpenKeyScan: " .. (c[6] or "неизвестная ошибка"))
        return
      end
      results[c[1]] = { bpm = c[2], camelot = c[3], key = c[4], openkey = c[5], err = c[6],
                        downbeat = tonumber(c[7] or "") }
    end

    local parts, first_bpm, first_item = {}, nil, nil
    for _, it in ipairs(items) do
      local r = results[it.file]
      local name = it.name ~= "" and it.name or it.file:match("[^\\/]+$")
      if not r then
        parts[#parts + 1] = name .. ": нет результата"
      elseif r.err and r.err ~= "" then
        parts[#parts + 1] = name .. ": ошибка, " .. r.err
      else
        local bpm = r.bpm ~= "" and (r.bpm .. " BPM") or "темп не определён"
        parts[#parts + 1] = name .. ": " .. bpm .. ", " .. r.camelot .. ", " .. r.key
        if not first_bpm and tonumber(r.bpm) then
          first_bpm, first_item = tonumber(r.bpm), { it = it, r = r }
        end
      end
    end
    local msg = table.concat(parts, ". ")

    if mode and mode.set_tempo then
      -- Темп и имена дорожек — одно действие отмены: Ctrl+Z откатывает всё сразу.
      reaper.Undo_BeginBlock()
      if not first_bpm then
        msg = msg .. ". Темп проекта не изменён: BPM не определён"
      elseif reaper.CountTempoTimeSigMarkers(0) > 0 then
        msg = msg .. ". Темп проекта не изменён: в проекте есть маркеры темпа"
      else
        reaper.SetCurrentBPM(0, first_bpm, false)
        msg = msg .. ". Темп проекта: " .. first_bpm
      end
      local renamed = tag_track_names(items, results)
      if renamed > 0 then
        msg = msg .. (renamed == 1 and ". Тональность записана в имя дорожки"
                      or (". Тональность записана в имена дорожек: " .. renamed))
      end
      if first_item then
        msg = msg .. ". " .. align_project_to_bar(first_item.it, first_item.r.downbeat, first_bpm)
      end
      msg = msg .. "."
      reaper.UpdateArrange()
      reaper.Undo_EndBlock("OpenKeyScan: темп, тональность и выравнивание по такту", -1)
    end
    say(msg)
  end

  local function poll()
    local res = io.open(out_path, "r")
    if res then
      local text = res:read("a")
      res:close()
      finish(text)
      return
    end
    if reaper.time_precise() - started > timeout then
      reaper.DeleteExtState(TITLE, "busy", false)
      os.remove(list_path)
      say("OpenKeyScan не ответила за " .. math.floor(timeout) .. " секунд. " ..
          "Проверьте, что программа запускается, и попробуйте ещё раз.")
      return
    end
    reaper.defer(poll)
  end
  reaper.defer(poll)
end

return M
