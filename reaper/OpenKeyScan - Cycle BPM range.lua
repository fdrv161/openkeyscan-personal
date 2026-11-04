-- OpenKeyScan: cycle the BPM range: 50 to 100, 70 to 140, no folding.
-- The choice is kept and applies to both analysis actions.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").cycle_range()
