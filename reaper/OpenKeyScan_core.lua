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

local function find_cli()
  local custom = reaper.GetExtState(TITLE, "cli_path")
  if custom ~= "" and reaper.file_exists(custom) then return custom end
  local default = (os.getenv("LOCALAPPDATA") or "") .. "\\Programs\\OpenKeyScan-Personal\\okscan-cli.exe"
  if reaper.file_exists(default) then return default end
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

local function busy_since()
  local v = tonumber(reaper.GetExtState(TITLE, "busy"))
  -- Метка старше 15 минут — остаток от прерванного запуска, её не считаем.
  if v and os.time() - v < 900 then return v end
  return nil
end

-- mode.set_tempo = true — после анализа выставить темп проекта по первому элементу.
function M.run(mode)
  if busy_since() then
    say("Анализ OpenKeyScan уже идёт, дождитесь результата.")
    return
  end

  local cli, expected = find_cli()
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
      items[#items + 1] = { file = file, name = name }
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
      results[c[1]] = { bpm = c[2], camelot = c[3], key = c[4], openkey = c[5], err = c[6] }
    end

    local parts, first_bpm = {}, nil
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
        first_bpm = first_bpm or tonumber(r.bpm)
      end
    end
    local msg = table.concat(parts, ". ")

    if mode and mode.set_tempo then
      if not first_bpm then
        msg = msg .. ". Темп проекта не изменён: BPM не определён."
      elseif reaper.CountTempoTimeSigMarkers(0) > 0 then
        msg = msg .. ". Темп проекта не изменён: в проекте есть маркеры темпа."
      else
        reaper.SetCurrentBPM(0, first_bpm, true)
        msg = msg .. ". Темп проекта: " .. first_bpm .. "."
      end
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
