-- OpenKeyScan: озвучить тональность и BPM, выставить темп проекта по первому выбранному элементу
-- и дописать тональность в имя его дорожки («instr» -> «instr [D minor, 7A]»).
-- Каждая дорожка получает тональность своего элемента. Всё отменяется одним Ctrl+Z.
-- Нужны: личная сборка OpenKeyScan и файл OpenKeyScan_core.lua в той же папке.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")
dofile(dir .. "OpenKeyScan_core.lua").run({ set_tempo = true, script_dir = dir })
