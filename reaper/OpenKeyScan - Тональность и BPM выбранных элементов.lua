-- OpenKeyScan: сообщить тональность и BPM выбранных элементов.
-- Нужны файл OpenKeyScan_core.lua и папка engine рядом с этим скриптом.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = false, script_dir = dir })
