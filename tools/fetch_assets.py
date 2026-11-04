"""
Downloads files that are not kept in git and checks their SHA-256:
the DeepRhythm 0.7 BPM model weights (AGPL-3.0, github.com/bleugreen/deeprhythm).
Not needed for the build: the engine uses the ready models/deeprhythm.onnx. The weights
are needed only to convert the network to ONNX again (tools/export_onnx.py).

Run: python tools/fetch_assets.py
Exit code 0: everything is downloaded and verified, 1: error (text in stderr).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# The link is pinned to a commit so that the weights cannot change unnoticed.
DEEPRHYTHM_URL = ("https://raw.githubusercontent.com/bleugreen/deeprhythm/"
                  "929dfc3eb8f7c2f0355719addc5a7c3216396ad6/deeprhythm-0.7.pth")
# The hash is published by the author in docs/benchmarks.md.
DEEPRHYTHM_SHA256 = "c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4"


def download(url):
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def check(data, expected, name):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name}: checksum mismatch ({actual}), the file was not saved.")


def main():
    ASSETS.mkdir(exist_ok=True)

    weights = ASSETS / "deeprhythm-0.7.pth"
    if weights.exists() and hashlib.sha256(weights.read_bytes()).hexdigest() == DEEPRHYTHM_SHA256:
        print("DeepRhythm weights are already in place.")
    else:
        print("Downloading DeepRhythm weights...")
        data = download(DEEPRHYTHM_URL)
        check(data, DEEPRHYTHM_SHA256, "deeprhythm-0.7.pth")
        weights.write_bytes(data)
        print("DeepRhythm weights saved.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
