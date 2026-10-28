-- OpenKeyScan: озвучить тональность и BPM выбранного элемента и выставить по нему темп проекта.
-- Если выбрано несколько элементов, темп берётся у первого. Отменяется обычным Ctrl+Z.
-- Нужны: личная сборка OpenKeyScan и файл OpenKeyScan_core.lua в той же папке.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = true })
