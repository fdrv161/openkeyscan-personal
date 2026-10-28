-- OpenKeyScan: переключить диапазон BPM по кругу: 50–100, 70–140, без приведения.
-- Выбор запоминается и действует на оба действия анализа.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").cycle_range()
