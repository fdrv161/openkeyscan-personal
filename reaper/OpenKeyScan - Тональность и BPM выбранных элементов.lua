-- OpenKeyScan: озвучить тональность и BPM выбранных элементов (через OSARA).
-- Нужны: личная сборка OpenKeyScan и файл OpenKeyScan_core.lua в той же папке.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = false })
