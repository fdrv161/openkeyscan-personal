"""
Скачивает файлы, которые не хранятся в git, и проверяет их SHA-256:
веса модели BPM DeepRhythm 0.7 (AGPL-3.0, github.com/bleugreen/deeprhythm).

Запуск: python tools/fetch_assets.py
Код выхода 0 — всё скачано и проверено, 1 — ошибка (текст в stderr).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# Ссылка закреплена на коммит, чтобы веса не поменялись незаметно.
DEEPRHYTHM_URL = ("https://raw.githubusercontent.com/bleugreen/deeprhythm/"
                  "929dfc3eb8f7c2f0355719addc5a7c3216396ad6/deeprhythm-0.7.pth")
# Хэш опубликован автором в docs/benchmarks.md.
DEEPRHYTHM_SHA256 = "c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4"


def download(url):
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def check(data, expected, name):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name}: контрольная сумма не совпала ({actual}), файл не сохранён.")


def main():
    ASSETS.mkdir(exist_ok=True)

    weights = ASSETS / "deeprhythm-0.7.pth"
    if weights.exists() and hashlib.sha256(weights.read_bytes()).hexdigest() == DEEPRHYTHM_SHA256:
        print("Веса DeepRhythm уже на месте.")
    else:
        print("Скачиваю веса DeepRhythm...")
        data = download(DEEPRHYTHM_URL)
        check(data, DEEPRHYTHM_SHA256, "deeprhythm-0.7.pth")
        weights.write_bytes(data)
        print("Веса DeepRhythm сохранены.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
