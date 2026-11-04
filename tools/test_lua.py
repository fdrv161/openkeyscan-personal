"""
Tests the logic of the REAPER scripts without REAPER: reaper/OpenKeyScan_core.lua runs
against a fake `reaper` table with a small project in memory, and the "engine" answers
with a prepared results file.
Needs the lupa package (embedded Lua): pip install lupa

Run: python tools/test_lua.py
Exit code 0: all tests passed, 1: there are failures (printed one per line).
"""
import os
import sys
import tempfile
from pathlib import Path

import lupa

ROOT = Path(__file__).resolve().parent.parent

MOCK = r'''
local T = {}          -- test state, reset by T.reset
reaper = {}

function T.reset(res_dir)
  T.res_dir, T.said, T.deferred, T.exec, T.ext = res_dir, {}, {}, nil, {}
  T.items, T.selected, T.tracks = {}, {}, {}
  T.answer, T.bpm, T.tempo_markers, T.now, T.tags = nil, 120, 0, 0, nil
  T.undo, T.samples, T.exit = 0, nil, nil
end

function T.track(name)
  local t = { name = name }
  T.tracks[#T.tracks + 1] = t
  return t
end

function T.item(pos, len, file, track, opts)
  opts = opts or {}
  local src = { file = file, midi = opts.midi }
  local take = { offs = opts.offs or 0, rate = opts.rate or 1, src = src, name = opts.name or "beat" }
  local it = { pos = pos, len = len, take = take, track = track or T.track("t") }
  T.items[#T.items + 1] = it
  if opts.selected ~= false then T.selected[#T.selected + 1] = it end
  return it
end

-- Runs deferred callbacks until the script stops asking for them.
function T.pump()
  local n = 0
  while #T.deferred > 0 and n < 1000 do
    n = n + 1
    table.remove(T.deferred, 1)()
  end
end

function T.last() return T.said[#T.said] or "" end

reaper.osara_outputMessage = function(m) T.said[#T.said + 1] = m end
reaper.MB = function(m) T.said[#T.said + 1] = m end
reaper.defer = function(f) T.deferred[#T.deferred + 1] = f end
reaper.atexit = function(f) T.exit = f end
reaper.time_precise = function() return T.now end
reaper.file_exists = function() return true end
reaper.GetResourcePath = function() return T.res_dir end
reaper.RecursiveCreateDirectory = function(path) py_makedirs(path) end
reaper.GetExtState = function(s, k) return T.ext[s .. "/" .. k] or "" end
reaper.SetExtState = function(s, k, v) T.ext[s .. "/" .. k] = v end
reaper.DeleteExtState = function(s, k) T.ext[s .. "/" .. k] = nil end
reaper.PreventUIRefresh = function() end
reaper.UpdateArrange = function() end
reaper.Undo_BeginBlock = function() end
reaper.Undo_EndBlock = function() T.undo = T.undo + 1 end
reaper.ValidatePtr2 = function() return true end

-- The "engine": writes the prepared answer where the command line says.
reaper.ExecProcess = function(cmd)
  T.exec = cmd
  local out = cmd:match('%-%-out "([^"]+)"')
  if T.answer then
    local f = assert(io.open(out, "w"))
    f:write(T.answer)
    f:close()
  end
end

reaper.CountSelectedMediaItems = function() return #T.selected end
reaper.GetSelectedMediaItem = function(_, i) return T.selected[i + 1] end
reaper.CountMediaItems = function() return #T.items end
reaper.GetMediaItem = function(_, i) return T.items[i + 1] end
reaper.GetMediaItem_Track = function(it) return it.track end
reaper.GetActiveTake = function(it) return it.take end
reaper.CountTakes = function() return 1 end
reaper.GetTake = function(it) return it.take end
reaper.TakeIsMIDI = function(t) return t.src.midi end
reaper.GetTakeName = function(t) return t.name end
reaper.GetMediaItemTake_Source = function(t) return t.src end
reaper.GetMediaSourceParent = function() return nil end
reaper.GetMediaSourceFileName = function(s) return s.file end
reaper.GetMediaSourceSampleRate = function() return 1000 end
reaper.GetMediaSourceNumChannels = function() return 1 end

local ITEM = { D_POSITION = "pos", D_LENGTH = "len" }
local TAKE = { D_STARTOFFS = "offs", D_PLAYRATE = "rate" }
reaper.GetMediaItemInfo_Value = function(it, k) return it[ITEM[k]] end
reaper.SetMediaItemInfo_Value = function(it, k, v) it[ITEM[k]] = v end
reaper.GetMediaItemTakeInfo_Value = function(t, k) return t[TAKE[k]] end
reaper.SetMediaItemTakeInfo_Value = function(t, k, v) t[TAKE[k]] = v end

reaper.GetSetMediaTrackInfo_String = function(tr, _, value, set)
  if set then tr.name = value end
  return true, tr.name
end

reaper.CountTempoTimeSigMarkers = function() return T.tempo_markers end
reaper.SetCurrentBPM = function(_, bpm) T.bpm = bpm end

-- 4/4 grid at the current tempo.
local function bar() return 4 * 60 / T.bpm end
reaper.TimeMap2_timeToBeats = function(_, t) return 0, math.floor(t / bar()) end
reaper.TimeMap2_beatsToTime = function(_, _, measure) return measure * bar() end

-- T.samples: the first sound of the file as REAPER decodes it, in file seconds.
reaper.new_array = function(n) return { n = n } end
reaper.CreateTakeAudioAccessor = function(t) return t end
reaper.DestroyAudioAccessor = function() end
reaper.GetAudioAccessorSamples = function(take, sr, _, start, n, buf)
  local data = {}
  for i = 1, n do
    local t = take.offs + (start + (i - 1) / sr) * take.rate
    data[i] = (T.samples and t >= T.samples) and 0.5 or 0
  end
  buf.table = function() return data end
  return 1
end

return T
'''

# Each test: name and Lua code. T is the state, M the core module, near() compares floats.
TESTS = [
    ("no selection asks to select an item", r'''
        M.run({ script_dir = "x\\" })
        check(T.last():find("No items selected"), T.last())
        check(not T.exec, "engine must not start")
    '''),
    ("MIDI items are skipped", r'''
        T.item(0, 10, "a.mid", nil, { midi = true })
        M.run({ script_dir = "x\\" })
        check(T.last():find("no audio files"), T.last())
    '''),
    ("report: result is spoken, project is not touched", r'''
        local it = T.item(3, 10, "a.mp3")
        T.answer = row("a.mp3", "88", "7A", "D minor", "12m", "", "0.3000", "0.1", "0.900")
        M.run({ set_tempo = false, script_dir = "x\\" })
        T.pump()
        check(T.last() == "beat: 88 BPM, 7A, D minor", T.last())
        check(it.pos == 3 and T.bpm == 120 and it.track.name == "t", "project changed")
    '''),
    ("temp files sit in the REAPER resource folder and are removed", r'''
        T.item(0, 10, "a.mp3")
        T.answer = row("a.mp3", "88", "7A", "D minor", "12m", "", "", "", "0.900")
        M.run({ script_dir = "x\\" })
        local list = T.exec:match('%-%-list "([^"]+)"')
        check(list:sub(1, #T.res_dir) == T.res_dir, list)
        local f = io.open(list)
        check(f and f:read("a") == "a.mp3\n", "list contents")
        if f then f:close() end
        T.pump()
        check(not io.open(list), "list not removed")
        check(reaper.GetExtState("OpenKeyScan", "busy") == "", "busy mark left")
    '''),
    ("low confidence adds a warning", r'''
        T.item(0, 10, "a.mp3")
        T.answer = row("a.mp3", "88", "7A", "D minor", "12m", "", "", "", "0.100")
        M.run({ script_dir = "x\\" })
        T.pump()
        check(T.last():find("tempo uncertain"), T.last())
    '''),
    ("one bad file does not lose the others", r'''
        T.item(0, 10, "a.mp3", nil, { name = "one" })
        T.item(0, 10, "b.mp3", nil, { name = "two" })
        T.answer = row("a.mp3", "", "", "", "", "file not found", "", "", "") ..
                   row("b.mp3", "90", "1A", "Ab minor", "6m", "", "", "", "0.900")
        M.run({ script_dir = "x\\" })
        T.pump()
        check(T.last() == "one: error, file not found. two: 90 BPM, 1A, Ab minor", T.last())
    '''),
    ("engine failure is reported", r'''
        T.item(0, 10, "a.mp3")
        T.answer = row("ERROR", "", "", "", "", "analyzer failure: boom")
        M.run({ script_dir = "x\\" })
        T.pump()
        check(T.last() == "OpenKeyScan error: analyzer failure: boom", T.last())
    '''),
    ("no answer from the engine ends with a timeout", r'''
        T.item(0, 10, "a.mp3")
        M.run({ script_dir = "x\\" })
        T.now = 100000
        T.pump()
        check(T.last():find("did not respond"), T.last())
        check(reaper.GetExtState("OpenKeyScan", "busy") == "", "busy mark left")
    '''),
    ("second run while busy is refused", r'''
        T.item(0, 10, "a.mp3")
        M.run({ script_dir = "x\\" })
        local first = T.exec
        T.exec = nil
        M.run({ script_dir = "x\\" })
        check(T.last():find("already running"), T.last())
        check(not T.exec, "second engine started")
    '''),
    ("tags answer without the engine", r'''
        T.item(0, 10, "a.mp3")
        reaper.GetMediaFileMetadata = function(_, id)
          if id == "ID3:TBPM" then return 1, "133" end
          if id == "ID3:TKEY" then return 1, "Dm" end
          return 0, ""
        end
        M.run({ set_tempo = false, script_dir = "x\\" })
        reaper.GetMediaFileMetadata = nil
        check(T.last() == "beat: 66.5 BPM, Dm. From file tags.", T.last())
        check(not T.exec, "engine started")
    '''),
    ("set tempo: moves everything left and trims the start", r'''
        local a = T.item(0, 10, "a.mp3", T.track("instr"))
        local b = T.item(5, 2, "b.wav", nil, { selected = false })
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "0.3000", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(T.bpm == 120, "tempo")
        check(near(a.pos, 0) and near(a.take.offs, 0.3) and near(a.len, 9.7), "item a trimmed")
        check(near(b.pos, 4.7), "item b moved with the rest")
        check(a.track.name == "instr [D minor, 7A]", a.track.name)
        check(T.last():find("left by 300 ms") and T.last():find("start trimmed"), T.last())
        check(T.undo == 1, "one undo step")
    '''),
    ("set tempo: moves right to the next bar", r'''
        local a = T.item(0, 10, "a.mp3")
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "1.9000", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(near(a.pos, 0.1), "item position " .. a.pos)
        check(T.last():find("right by 100 ms") and T.last():find("downbeat on bar 2"), T.last())
    '''),
    ("set tempo: repeated run replaces the key mark", r'''
        local a = T.item(0, 10, "a.mp3", T.track("instr [C major, 8B]"))
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "0.0000", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(a.track.name == "instr [D minor, 7A]", a.track.name)
        check(T.last():find("already on the grid"), T.last())
    '''),
    ("set tempo: tempo markers are respected", r'''
        T.tempo_markers = 1
        T.item(0, 10, "a.mp3")
        T.answer = row("a.mp3", "100", "7A", "D minor", "12m", "", "0.0000", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(T.bpm == 120, "tempo changed")
        check(T.last():find("has tempo markers"), T.last())
    '''),
    ("set tempo: downbeat in a trimmed part moves to the next bar", r'''
        local a = T.item(0, 10, "a.mp3", nil, { offs = 1 })
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "0.5000", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        -- first bar in the item is at 2.5 s of the file, 1.5 s of the item: the nearest line is 2.0
        check(near(a.pos, 0.5), "item position " .. a.pos)
    '''),
    ("set tempo: REAPER trimming the file start is corrected", r'''
        local a = T.item(0, 10, "a.mp3")
        T.samples = 0.075         -- REAPER hears the sound 25 ms before the engine
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "1.9000", "0.100000", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(near(a.pos, 0.125, 0.002), "item position " .. a.pos)
        check(T.last():find("offset of 25 ms"), T.last())
    '''),
    ("no downbeat: items stay", r'''
        local a = T.item(2, 10, "a.mp3")
        T.answer = row("a.mp3", "120", "7A", "D minor", "12m", "", "", "", "0.900")
        M.run({ set_tempo = true, script_dir = "x\\" })
        T.pump()
        check(a.pos == 2, "item moved")
        check(T.last():find("Could not find the first downbeat"), T.last())
    '''),
    ("BPM range cycles and is passed to the engine", r'''
        M.cycle_range()
        check(T.last() == "BPM range: 70 to 140 BPM", T.last())
        T.item(0, 10, "a.mp3")
        M.run({ script_dir = "x\\" })
        check(T.exec:find("%-%-range 70"), T.exec)
        T.exit()
        M.cycle_range()
        check(T.last():find("no folding"), T.last())
        M.cycle_range()
        check(T.last() == "BPM range: 50 to 100 BPM", T.last())
    '''),
]

HELPERS = r'''
function row(...) return table.concat({ ... }, "\t") .. "\n" end
function near(a, b, eps) return math.abs(a - b) < (eps or 1e-6) end
function check(ok, text)
  if not ok then error(tostring(text), 2) end
end
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, code in TESTS:
            lua = lupa.LuaRuntime(encoding='utf-8')
            lua.globals().py_makedirs = lambda p: os.makedirs(p, exist_ok=True)
            state = lua.execute(MOCK)
            lua.globals().T = state
            lua.execute(HELPERS)
            state.reset(tmp)
            source = (ROOT / 'reaper' / 'OpenKeyScan_core.lua').read_text(encoding='utf-8')
            lua.globals().M = lua.eval("function(src) return load(src, '=core')() end")(source)
            try:
                lua.execute(code)
                print(f"OK: {name}")
            except lupa.LuaError as e:
                failures += 1
                print(f"FAILED: {name}: {e}")
    print("All Lua tests passed." if not failures else f"Failed tests: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
