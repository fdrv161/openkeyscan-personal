-- OpenKeyScan: report the key and BPM, set the project tempo from the first selected item,
-- append the key to the track name ("instr" -> "instr [D minor, 7A]") and put the first
-- downbeat of the beat on a bar line. Each track gets the key of its own item.
-- Everything is undone with a single Ctrl+Z.
-- Needs OpenKeyScan_core.lua and the engine folder next to this script.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = true, script_dir = dir })
