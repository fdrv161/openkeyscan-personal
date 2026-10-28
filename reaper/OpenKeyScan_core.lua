-- Общий код скриптов OpenKeyScan для REAPER. Это не отдельное действие:
-- его подключают файлы «OpenKeyScan - ….lua», загружать в список действий не нужно.
--
-- Что делает: берёт исходные файлы выбранных элементов, отдаёт их движку анализа
-- (engine\okscan-cli.exe рядом со скриптами) и сообщает результат.
-- Анализ идёт в отдельном процессе, REAPER при этом не замирает.

local M = {}

local TITLE = "OpenKeyScan"
-- Первый запуск грузит модели дольше, поэтому запас щедрый.
local TIMEOUT_BASE, TIMEOUT_PER_FILE = 90, 30

-- quiet = true — промежуточное сообщение («Анализирую»): без OSARA его не показываем,
-- чтобы не открывать окно на каждом шаге. Итоги и ошибки показываются всегда.
local function say(msg, quiet)
  if reaper.osara_outputMessage then
    reaper.osara_outputMessage(msg)
  elseif not quiet then
    reaper.MB(msg, TITLE, 0)
  end
end

-- Где искать okscan-cli.exe, по порядку:
-- 1) путь, сохранённый в ExtState (OpenKeyScan / cli_path);
-- 2) папка engine рядом со скриптом — так лежит распакованный архив;
-- 3) свежая сборка в папке исходников (reaper\..\dist\OpenKeyScan).
local function find_cli(script_dir)
  local default = (script_dir or "") .. "engine\\okscan-cli.exe"
  local candidates = { reaper.GetExtState(TITLE, "cli_path"), default }
  if script_dir then
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
  return file, reaper.GetTakeName(take), src
end

-- Диапазоны приведения BPM: нижняя граница (0 — не приводить) и название. Первый — по умолчанию.
local RANGES = { { 50, "50–100 BPM" }, { 70, "70–140 BPM" }, { 0, "без приведения, полный темп" } }
-- Ниже этой уверенности сети темп может быть ошибочным (половина, полтора раза) — предупреждаем.
-- Порог осторожный: верный темп 123.mp3 (88 BPM) имел уверенность 0,45.
local LOW_CONFIDENCE = 0.3

local function bpm_low()
  local v = tonumber(reaper.GetExtState(TITLE, "bpm_low"))
  for _, r in ipairs(RANGES) do
    if r[1] == v then return v end
  end
  return RANGES[1][1]
end

-- Как fold_bpm в движке: умножаем или делим на 2, пока темп не попадёт в [low, 2*low].
local function fold_bpm(bpm, low)
  if low == 0 then return bpm end
  while bpm < low do bpm = bpm * 2 end
  while bpm > 2 * low do bpm = bpm / 2 end
  return bpm
end

local function format_bpm(bpm)
  local text = string.format("%.1f", bpm)
  return (text:gsub("%.0$", ""))
end

-- Первое непустое значение тега из списка имён (REAPER 6.2 и новее) или nil.
local function file_tag(src, ids)
  if not reaper.GetMediaFileMetadata then return nil end
  for _, id in ipairs(ids) do
    local ok, value = reaper.GetMediaFileMetadata(src, id)
    if ok == 1 and value and value:match("%S") then return value:match("^%s*(.-)%s*$") end
  end
  return nil
end

-- BPM и тональность из тегов файла или nil, если чего-то нет.
local function tags_of(src)
  local text = (file_tag(src, { "ID3:TBPM", "XMP:dm/tempo", "VORBIS:BPM" }) or ""):gsub(",", ".")
  local bpm = tonumber(text)
  local key = file_tag(src, { "ID3:TKEY", "XMP:dm/key", "VORBIS:INITIALKEY", "VORBIS:KEY" })
  if bpm and bpm > 0 and key then return bpm, key end
  return nil
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

-- Порог «здесь начинается звук» (−50 дБFS) — тот же, что у движка (LEAD_LEVEL).
local LEAD_LEVEL = 10 ^ (-50 / 20)

-- Насколько шкала REAPER сдвинута относительно шкалы движка, в секундах исходного файла.
-- Декодеры могут по-разному срезать начало: у MP3 без заголовка LAME разница
-- доходит до десятков миллисекунд, и бит вставал бы мимо такта. Движок сообщает,
-- где в файле начинается звук (lead); здесь то же самое меряем через REAPER.
local function decoder_offset(take, lead)
  if not lead then return 0 end
  local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if lead < offs then return 0 end   -- начало звука обрезано в элементе, сверить нечем
  local src = reaper.GetMediaItemTake_Source(take)
  local sr = reaper.GetMediaSourceSampleRate(src)
  local nch = math.max(1, reaper.GetMediaSourceNumChannels(src))
  if sr <= 0 then return 0 end
  -- окно 0,4 с от «за 0,2 с до начала звука по движку» (секунды файла), но не раньше начала дубля
  local from = math.max(offs, lead - 0.2)
  local start = (from - offs) / rate
  local n = math.floor(0.4 * sr / rate)
  local buf = reaper.new_array(n * nch)
  local acc = reaper.CreateTakeAudioAccessor(take)
  local ok = reaper.GetAudioAccessorSamples(acc, sr, nch, start, n, buf)
  reaper.DestroyAudioAccessor(acc)
  if ok ~= 1 then return 0 end
  local data = buf.table()
  for i = 1, #data do
    if math.abs(data[i]) > LEAD_LEVEL then
      local idx = math.floor((i - 1) / nch)
      -- звук есть с первого же отсчёта, а окно начато не с начала файла:
      -- настоящего начала звука не видно, сверить нельзя
      if idx == 0 and from > 0 then return 0 end
      local found = from + idx / sr * rate
      local delta = found - lead
      -- больше 0,15 с — это уже не декодер (эффекты, другой файл), не трогаем
      return math.abs(delta) < 0.15 and delta or 0
    end
  end
  return 0
end

local function describe_shift(delta)
  local ms = math.floor(math.abs(delta) * 1000 + 0.5)
  local amount = ms < 1000 and (ms .. " мс") or (string.format("%.2f", math.abs(delta)) .. " с")
  return (delta > 0 and "вправо на " or "влево на ") .. amount
end

-- Ставит первую сильную долю бита (секунды от начала файла) на ближайшую линию
-- такта и сдвигает на столько же ВСЕ элементы проекта, чтобы ничего не разъехалось.
-- Возвращает фразу для итогового сообщения.
local function align_project_to_bar(it, downbeat, bpm, lead)
  if not downbeat then
    return "Первую сильную долю найти не удалось, элементы не сдвинуты"
  end
  local item = it.item
  if not reaper.ValidatePtr2(0, item, "MediaItem*") then
    return "Элемент пропал из проекта, выравнивание пропущено"
  end
  local take = reaper.GetActiveTake(item)
  if not take then
    return "У элемента нет активного дубля, выравнивание пропущено"
  end
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  local fix = decoder_offset(take, lead)
  downbeat = downbeat + fix
  -- О поправке говорим, только если она заметна: так видно, что сверка сработала.
  local note = math.abs(fix) >= 0.002
      and (", учтён сдвиг начала файла " .. math.floor(math.abs(fix) * 1000 + 0.5) .. " мс") or ""
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
    return "Бит уже стоит по сетке" .. note
  end

  local trimmed = false
  for i = 0, reaper.CountMediaItems(0) - 1 do
    if shift_item(reaper.GetMediaItem(0, i), delta) then trimmed = true end
  end
  return "Все элементы сдвинуты " .. describe_shift(delta) .. ", сильная доля бита на такте " ..
         (measure + (target == here and 1 or 2)) .. (trimmed and ", начало подрезано" or "") .. note
end

local function busy_since()
  local v = tonumber(reaper.GetExtState(TITLE, "busy"))
  -- Метка старше 15 минут — остаток от прерванного запуска, её не считаем.
  if v and os.time() - v < 900 then return v end
  return nil
end

-- mode.set_tempo = true — после анализа выставить темп проекта по первому элементу.
-- mode.script_dir — папка скрипта (со слешем на конце), чтобы найти программу рядом.
-- Действие «Переключить диапазон BPM»: по кругу 50–100, 70–140, без приведения.
-- Выбор запоминается между запусками REAPER.
function M.cycle_range()
  local cur, nxt = bpm_low(), RANGES[1]
  for i, r in ipairs(RANGES) do
    if r[1] == cur then nxt = RANGES[i % #RANGES + 1] end
  end
  reaper.SetExtState(TITLE, "bpm_low", tostring(nxt[1]), true)
  say("Диапазон BPM: " .. nxt[2])
end

function M.run(mode)
  if busy_since() then
    say("Анализ OpenKeyScan уже идёт, дождитесь результата.")
    return
  end

  local cli, expected = find_cli(mode and mode.script_dir)
  if not cli then
    say("Не найден движок анализа OpenKeyScan. Распакуйте архив целиком, чтобы папка engine " ..
        "лежала рядом со скриптами. Ожидаемый путь: " .. expected)
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
    local file, name, src = item_file(item)
    if file then
      items[#items + 1] = { file = file, name = name, src = src, item = item, track = reaper.GetMediaItem_Track(item) }
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

  local low = bpm_low()

  -- Только сообщить: если у всех файлов в тегах есть и BPM, и тональность, движок не нужен.
  -- Для «задать темп» теги не годятся — в них нет первой сильной доли.
  if not (mode and mode.set_tempo) then
    local parts = {}
    for _, it in ipairs(items) do
      local bpm, key = tags_of(it.src)
      if not bpm then
        parts = nil
        break
      end
      local name = it.name ~= "" and it.name or it.file:match("[^\\/]+$")
      parts[#parts + 1] = name .. ": " .. format_bpm(fold_bpm(bpm, low)) .. " BPM, " .. key
    end
    if parts then
      say(table.concat(parts, ". ") .. ". Из тегов файла.")
      return
    end
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
  reaper.ExecProcess('"' .. cli .. '" --list "' .. list_path .. '" --out "' .. out_path .. '" --range ' .. low, -2)
  reaper.SetExtState(TITLE, "busy", tostring(os.time()), false)
  say(#files == 1 and "Анализирую." or ("Анализирую файлов: " .. #files .. "."), true)

  local started = reaper.time_precise()
  local timeout = TIMEOUT_BASE + TIMEOUT_PER_FILE * #files

  local function cleanup()
    reaper.DeleteExtState(TITLE, "busy", false)
    os.remove(list_path)
    os.remove(out_path)
  end
  -- Если скрипт остановят до результата (или закроют REAPER), признак «занято» всё равно снимется.
  reaper.atexit(cleanup)

  local function finish(text)
    cleanup()

    local results = {}
    for line in text:gmatch("[^\n]+") do
      local c = split_tabs(line)
      if c[1] == "ERROR" then
        say("Ошибка OpenKeyScan: " .. (c[6] or "неизвестная ошибка"))
        return
      end
      results[c[1]] = { bpm = c[2], camelot = c[3], key = c[4], openkey = c[5], err = c[6],
                        downbeat = tonumber(c[7] or ""), lead = tonumber(c[8] or ""),
                        conf = tonumber(c[9] or "") }
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
        if r.conf and r.conf < LOW_CONFIDENCE and r.bpm ~= "" then
          bpm = bpm .. " (темп неуверенный, проверьте на слух)"
        end
        parts[#parts + 1] = name .. ": " .. bpm .. ", " .. r.camelot .. ", " .. r.key
        if not first_bpm and tonumber(r.bpm) then
          first_bpm, first_item = tonumber(r.bpm), { it = it, r = r }
        end
      end
    end
    local msg = table.concat(parts, ". ")

    if mode and mode.set_tempo then
      -- Темп, имена дорожек и сдвиг — одно действие отмены: Ctrl+Z откатывает всё сразу.
      -- PreventUIRefresh: экран не перерисовывается после каждого сдвинутого элемента.
      -- pcall: при сбое посередине блок отмены и перерисовка всё равно закрываются.
      reaper.PreventUIRefresh(1)
      reaper.Undo_BeginBlock()
      local ok, err = pcall(function()
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
          msg = msg .. ". " .. align_project_to_bar(first_item.it, first_item.r.downbeat, first_bpm, first_item.r.lead)
        end
      end)
      reaper.Undo_EndBlock("OpenKeyScan: темп, тональность и выравнивание по такту", -1)
      reaper.PreventUIRefresh(-1)
      reaper.UpdateArrange()
      if not ok then
        msg = msg .. ". Не удалось изменить проект: " .. tostring(err)
      end
      msg = msg .. "."
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
      cleanup()
      say("OpenKeyScan не ответила за " .. math.floor(timeout) .. " секунд. " ..
          "Проверьте, что программа запускается, и попробуйте ещё раз.")
      return
    end
    reaper.defer(poll)
  end
  reaper.defer(poll)
end

return M
