-- OpenKeyScan: сообщить тональность и BPM, выставить темп проекта по первому выбранному элементу,
-- дописать тональность в имя дорожки («instr» -> «instr [D minor, 7A]») и поставить
-- первую сильную долю бита на такт. Каждая дорожка получает тональность своего элемента.
-- Всё отменяется одним Ctrl+Z.
-- Нужны файл OpenKeyScan_core.lua и папка engine рядом с этим скриптом.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = true, script_dir = dir })
