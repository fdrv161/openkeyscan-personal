"""
Проверка синтаксиса скриптов REAPER (reaper/*.lua) без запуска REAPER.
Нужен пакет lupa (встроенный Lua): pip install lupa

Запуск: python tools/check_lua.py
Код выхода 0 — все файлы компилируются, 1 — есть ошибки (выводятся построчно).
"""
import sys
from pathlib import Path

import lupa

ROOT = Path(__file__).resolve().parent.parent


def main():
    lua = lupa.LuaRuntime()
    # Имя куска — латиницей: Lua обрезает длинные имена посреди русской буквы.
    compile_ = lua.eval("function(src) local f, err = load(src, '=script') return err end")
    errors = 0
    for path in sorted((ROOT / "reaper").glob("*.lua")):
        err = compile_(path.read_text(encoding="utf-8"))
        if err:
            errors += 1
            print(f"ОШИБКА в {path.name}: {err}")
        else:
            print(f"OK: {path.name}")
    print("Синтаксис Lua в порядке." if not errors else f"Файлов с ошибками: {errors}.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
