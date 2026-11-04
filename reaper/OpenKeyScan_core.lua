-- Shared code of the OpenKeyScan scripts for REAPER. This is not an action of its own:
-- the "OpenKeyScan - ....lua" files include it, there is no need to load it into the action list.
--
-- What it does: takes the source files of the selected items, passes them to the analysis
-- engine (engine\okscan-cli.exe next to the scripts) and reports the result.
-- The analysis runs in a separate process, so REAPER does not freeze.

local M = {}

local TITLE = "OpenKeyScan"
-- The first run loads the models for longer, so the margin is generous.
local TIMEOUT_BASE, TIMEOUT_PER_FILE = 90, 30

-- quiet = true: an interim message ("Analyzing"). Without OSARA it is not shown, so as
-- not to open a window at every step. Results and errors are always shown.
local function say(msg, quiet)
  if reaper.osara_outputMessage then
    reaper.osara_outputMessage(msg)
  elseif not quiet then
    reaper.MB(msg, TITLE, 0)
  end
end

-- Where to look for okscan-cli.exe, in order:
-- 1) the path saved in ExtState (OpenKeyScan / cli_path);
-- 2) the engine folder next to the script: that is how the unpacked archive looks;
-- 3) a fresh build in the source folder (reaper\..\dist\OpenKeyScan).
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

-- Source file of the active take. Trimmed and reversed items wrap the source in a
-- "section", so we climb to the topmost parent.
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

-- BPM folding ranges: lower bound (0 means no folding) and name. The first is the default.
local RANGES = { { 50, "50 to 100 BPM" }, { 70, "70 to 140 BPM" }, { 0, "no folding, full tempo" } }
-- Below this network confidence the tempo may be wrong (half, one and a half times): warn.
-- The threshold is cautious: a correct tempo of 88 BPM was seen with a confidence of 0.45.
local LOW_CONFIDENCE = 0.3

local function bpm_low()
  local v = tonumber(reaper.GetExtState(TITLE, "bpm_low"))
  for _, r in ipairs(RANGES) do
    if r[1] == v then return v end
  end
  return RANGES[1][1]
end

-- Like fold_bpm in the engine: multiply or divide by 2 until the tempo falls into [low, 2*low].
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

-- First non-empty tag value from a list of names (REAPER 6.2 and newer), or nil.
local function file_tag(src, ids)
  if not reaper.GetMediaFileMetadata then return nil end
  for _, id in ipairs(ids) do
    local ok, value = reaper.GetMediaFileMetadata(src, id)
    if ok == 1 and value and value:match("%S") then return value:match("^%s*(.-)%s*$") end
  end
  return nil
end

-- BPM and key from the file's tags, or nil if something is missing.
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

-- Appends the key to the end of the track name: "instr" -> "instr [D minor, 7A]".
-- An older mark of the same kind is removed first, so that brackets do not pile up on
-- repeated runs. A track with several items gets the key of the first one.
local function tag_track_names(items, results)
  local done, count = {}, 0
  for _, it in ipairs(items) do
    local r = results[it.file]
    local track = it.track
    if r and (r.err or "") == "" and r.camelot ~= "" and track and not done[track]
       and reaper.ValidatePtr2(0, track, "MediaTrack*") then
      done[track] = true
      local _, name = reaper.GetSetMediaTrackInfo_String(track, "P_NAME", "", false)
      -- "[anything, 7A]" at the very end of the name is our previous mark
      name = name:gsub("%s*%[[^%[%]]*%d+[AB]%]$", "")
      local tag = "[" .. r.key .. ", " .. r.camelot .. "]"
      reaper.GetSetMediaTrackInfo_String(track, "P_NAME", name ~= "" and (name .. " " .. tag) or tag, true)
      count = count + 1
    end
  end
  return count
end

-- Moves the whole item by delta seconds. What would go left of the project start is
-- trimmed (the start of the audio inside the item moves), not lost.
local function shift_item(item, delta)
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION") + delta
  if pos >= 0 then
    reaper.SetMediaItemInfo_Value(item, "D_POSITION", pos)
    return false
  end
  local cut = -pos
  local len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
  if cut >= len then
    -- the whole item would end up before the project start: leave it at zero as it is
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

-- The "sound starts here" threshold (-50 dBFS), the same as in the engine (LEAD_LEVEL).
local LEAD_LEVEL = 10 ^ (-50 / 20)

-- How far REAPER's timeline is shifted from the engine's, in seconds of the source file.
-- Decoders may trim the start differently: for MP3 without a LAME header the difference
-- reaches tens of milliseconds and the beat would land off the bar. The engine reports
-- where sound starts in the file (lead); here we measure the same through REAPER.
local function decoder_offset(take, lead)
  if not lead then return 0 end
  local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if lead < offs then return 0 end   -- the start of sound is trimmed in the item, nothing to compare
  local src = reaper.GetMediaItemTake_Source(take)
  local sr = reaper.GetMediaSourceSampleRate(src)
  local nch = math.max(1, reaper.GetMediaSourceNumChannels(src))
  if sr <= 0 then return 0 end
  -- a 0.4 s window from "0.2 s before the engine's start of sound" (file seconds), but not before the take start
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
      -- sound is there from the very first sample and the window does not start at the file start:
      -- the real start of sound is not visible, so no comparison is possible
      if idx == 0 and from > 0 then return 0 end
      local found = from + idx / sr * rate
      local delta = found - lead
      -- more than 0.15 s is not the decoder any more (effects, another file): leave it alone
      return math.abs(delta) < 0.15 and delta or 0
    end
  end
  return 0
end

local function describe_shift(delta)
  local ms = math.floor(math.abs(delta) * 1000 + 0.5)
  local amount = ms < 1000 and (ms .. " ms") or (string.format("%.2f", math.abs(delta)) .. " s")
  return (delta > 0 and "right by " or "left by ") .. amount
end

-- Puts the first downbeat of the beat (seconds from the file start) on the nearest bar
-- line and moves ALL project items by the same amount, so that nothing drifts apart.
-- Returns a phrase for the final message.
local function align_project_to_bar(it, downbeat, bpm, lead)
  if not downbeat then
    return "Could not find the first downbeat, items were not moved"
  end
  local item = it.item
  if not reaper.ValidatePtr2(0, item, "MediaItem*") then
    return "The item is gone from the project, alignment skipped"
  end
  local take = reaper.GetActiveTake(item)
  if not take then
    return "The item has no active take, alignment skipped"
  end
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local rate = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  local fix = decoder_offset(take, lead)
  downbeat = downbeat + fix
  -- The correction is mentioned only if it is noticeable: that shows the check worked.
  local note = math.abs(fix) >= 0.002
      and (", file start offset of " .. math.floor(math.abs(fix) * 1000 + 0.5) .. " ms applied") or ""
  -- If the item start is trimmed past the first downbeat, take the first bar inside the item.
  if downbeat < offs and bpm then
    local bar = 4 * 60 / bpm
    downbeat = downbeat + math.ceil((offs - downbeat) / bar) * bar
  end
  local beat_time = pos + (downbeat - offs) / rate

  -- nearest bar line on the current project grid (the tempo is already set above)
  local _, measure = reaper.TimeMap2_timeToBeats(0, beat_time)
  local here = reaper.TimeMap2_beatsToTime(0, 0, measure)
  local nextbar = reaper.TimeMap2_beatsToTime(0, 0, measure + 1)
  local target = (beat_time - here <= nextbar - beat_time) and here or nextbar
  local delta = target - beat_time
  if math.abs(delta) < 0.0005 then
    return "The beat is already on the grid" .. note
  end

  local trimmed = false
  for i = 0, reaper.CountMediaItems(0) - 1 do
    if shift_item(reaper.GetMediaItem(0, i), delta) then trimmed = true end
  end
  return "All items moved " .. describe_shift(delta) .. ", downbeat on bar " ..
         (measure + (target == here and 1 or 2)) .. (trimmed and ", start trimmed" or "") .. note
end

local function busy_since()
  local v = tonumber(reaper.GetExtState(TITLE, "busy"))
  -- A mark older than 15 minutes is left over from an interrupted run: ignore it.
  if v and os.time() - v < 900 then return v end
  return nil
end

-- mode.set_tempo = true: after the analysis, set the project tempo from the first item.
-- mode.script_dir: the script's folder (with a trailing slash), to find the engine next to it.
-- The "Cycle BPM range" action: 50 to 100, 70 to 140, no folding, in a circle.
-- The choice is kept between REAPER sessions.
function M.cycle_range()
  local cur, nxt = bpm_low(), RANGES[1]
  for i, r in ipairs(RANGES) do
    if r[1] == cur then nxt = RANGES[i % #RANGES + 1] end
  end
  reaper.SetExtState(TITLE, "bpm_low", tostring(nxt[1]), true)
  say("BPM range: " .. nxt[2])
end

function M.run(mode)
  if busy_since() then
    say("OpenKeyScan analysis is already running, wait for the result.")
    return
  end

  local cli, expected = find_cli(mode and mode.script_dir)
  if not cli then
    say("OpenKeyScan analysis engine not found. Unpack the whole archive so that the engine folder " ..
        "sits next to the scripts. Expected path: " .. expected)
    return
  end

  local count = reaper.CountSelectedMediaItems(0)
  if count == 0 then
    say("No items selected. Select an item with audio and run the action again.")
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
    say("The selected items have no audio files: MIDI and empty items are not analyzed.")
    return
  end

  local low = bpm_low()

  -- Report only: if all files have both BPM and key in their tags, the engine is not needed.
  -- Tags are no use for "set tempo": they have no first downbeat.
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
      say(table.concat(parts, ". ") .. ". From file tags.")
      return
    end
  end

  -- Not os.getenv("TEMP"): plain Lua gets it in the system code page, so a non-English
  -- user name would break the path. REAPER's own functions return UTF-8.
  local dir = reaper.GetResourcePath() .. "\\Data\\OpenKeyScan"
  reaper.RecursiveCreateDirectory(dir, 0)
  local stamp = tostring(os.time()) .. "_" .. tostring(math.random(100000, 999999))
  local list_path = dir .. "\\list_" .. stamp .. ".txt"
  local out_path = dir .. "\\out_" .. stamp .. ".tsv"

  local f = io.open(list_path, "w")
  if not f then
    say("Could not create a temporary file in the folder " .. dir)
    return
  end
  f:write(table.concat(files, "\n"), "\n")
  f:close()

  -- -2: do not wait for completion, and start the console window minimized.
  reaper.ExecProcess('"' .. cli .. '" --list "' .. list_path .. '" --out "' .. out_path .. '" --range ' .. low, -2)
  reaper.SetExtState(TITLE, "busy", tostring(os.time()), false)
  say(#files == 1 and "Analyzing." or ("Analyzing " .. #files .. " files."), true)

  local started = reaper.time_precise()
  local timeout = TIMEOUT_BASE + TIMEOUT_PER_FILE * #files

  local function cleanup()
    reaper.DeleteExtState(TITLE, "busy", false)
    os.remove(list_path)
    os.remove(out_path)
  end
  -- If the script is stopped before the result (or REAPER is closed), the busy mark is still cleared.
  reaper.atexit(cleanup)

  local function finish(text)
    cleanup()

    local results = {}
    for line in text:gmatch("[^\n]+") do
      local c = split_tabs(line)
      if c[1] == "ERROR" then
        say("OpenKeyScan error: " .. (c[6] or "unknown error"))
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
        parts[#parts + 1] = name .. ": no result"
      elseif r.err and r.err ~= "" then
        parts[#parts + 1] = name .. ": error, " .. r.err
      else
        local bpm = r.bpm ~= "" and (r.bpm .. " BPM") or "tempo not detected"
        if r.conf and r.conf < LOW_CONFIDENCE and r.bpm ~= "" then
          bpm = bpm .. " (tempo uncertain, check by ear)"
        end
        parts[#parts + 1] = name .. ": " .. bpm .. ", " .. r.camelot .. ", " .. r.key
        if not first_bpm and tonumber(r.bpm) then
          first_bpm, first_item = tonumber(r.bpm), { it = it, r = r }
        end
      end
    end
    local msg = table.concat(parts, ". ")

    if mode and mode.set_tempo then
      -- Tempo, track names and the shift are one undo step: Ctrl+Z reverts everything at once.
      -- PreventUIRefresh: the screen is not redrawn after each moved item.
      -- pcall: if something fails midway, the undo block and redraw are still closed.
      reaper.PreventUIRefresh(1)
      reaper.Undo_BeginBlock()
      local ok, err = pcall(function()
        if not first_bpm then
          msg = msg .. ". Project tempo not changed: BPM not detected"
        elseif reaper.CountTempoTimeSigMarkers(0) > 0 then
          msg = msg .. ". Project tempo not changed: the project has tempo markers"
        else
          reaper.SetCurrentBPM(0, first_bpm, false)
          msg = msg .. ". Project tempo: " .. first_bpm
        end
        local renamed = tag_track_names(items, results)
        if renamed > 0 then
          msg = msg .. (renamed == 1 and ". Key added to the track name"
                        or (". Key added to track names: " .. renamed))
        end
        if first_item then
          msg = msg .. ". " .. align_project_to_bar(first_item.it, first_item.r.downbeat, first_bpm, first_item.r.lead)
        end
      end)
      reaper.Undo_EndBlock("OpenKeyScan: tempo, key and bar alignment", -1)
      reaper.PreventUIRefresh(-1)
      reaper.UpdateArrange()
      if not ok then
        msg = msg .. ". Could not change the project: " .. tostring(err)
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
      say("OpenKeyScan did not respond in " .. math.floor(timeout) .. " seconds. " ..
          "Check that the engine starts and try again.")
      return
    end
    reaper.defer(poll)
  end
  reaper.defer(poll)
end

return M
