"""
Checks the syntax of the REAPER scripts (reaper/*.lua) without running REAPER.
Needs the lupa package (embedded Lua): pip install lupa

Run: python tools/check_lua.py
Exit code 0: all files compile, 1: there are errors (printed one per line).
"""
import sys
from pathlib import Path

import lupa

ROOT = Path(__file__).resolve().parent.parent


def main():
    lua = lupa.LuaRuntime()
    # Fixed chunk name: Lua truncates long file names in its messages.
    compile_ = lua.eval("function(src) local f, err = load(src, '=script') return err end")
    errors = 0
    for path in sorted((ROOT / "reaper").glob("*.lua")):
        err = compile_(path.read_text(encoding="utf-8"))
        if err:
            errors += 1
            print(f"ERROR in {path.name}: {err}")
        else:
            print(f"OK: {path.name}")
    print("Lua syntax is fine." if not errors else f"Files with errors: {errors}.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
