-- OpenKeyScan: report the key and BPM of the selected items.
-- Needs OpenKeyScan_core.lua and the engine folder next to this script.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = false, script_dir = dir })
