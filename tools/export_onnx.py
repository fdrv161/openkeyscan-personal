"""
Converts both networks from torch format to ONNX, which the engine runs through onnxruntime.

For developers only, when weights change: torch does not go into the program.
Dependencies: pip install -r requirements-dev.txt; DeepRhythm weights: python tools/fetch_assets.py.
Run: python tools/export_onnx.py
Result: models/keynet.onnx and models/deeprhythm.onnx (kept in git).
Exit code 0: both files are written and give the same output as torch, 1: difference.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import KeyNet  # noqa: E402
from okscan.deeprhythm.model import DeepRhythmModel  # noqa: E402

MODELS = ROOT / "models"
OPSET = 17


def export(model, example, path, input_name, dynamic_axes):
    # The older exporter (dynamo=False): it needs no onnxscript, and for simple
    # convolutional networks the result is the same. Weights stay inside the file, no .data next to it.
    torch.onnx.export(model, (example,), str(path), dynamo=False, opset_version=OPSET,
                      input_names=[input_name], output_names=["logits"],
                      dynamic_axes=dynamic_axes)


def compare(model, path, inputs):
    """Compares torch and onnxruntime output on several inputs, returns the worst deviation."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    worst = 0.0
    with torch.inference_mode():
        for x in inputs:
            ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(None, {name: x})[0]
            if ref.argmax(axis=1).tolist() != got.argmax(axis=1).tolist():
                raise RuntimeError(f"{path.name}: different network answer on input {x.shape}")
            worst = max(worst, float(np.abs(ref - got).max()))
    return worst


def main():
    MODELS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    key = KeyNet(num_classes=24, in_channels=1, Nf=64, p=0.3)
    key.load_state_dict(torch.load(ROOT / "checkpoints" / "openkeyscan3.pt",
                                   map_location="cpu", weights_only=True))
    key.eval()
    key_path = MODELS / "keynet.onnx"
    # Input: log-CQT (1 track, 1 channel, 105 bins, frames of 0.2 s, any length).
    export(key, torch.rand(1, 1, 105, 900), key_path, "spec", {"spec": {3: "frames"}})
    key_err = compare(key, key_path, [rng.random((1, 1, 105, n), dtype=np.float32) * 3
                                      for n in (40, 900, 2500)])

    bpm = DeepRhythmModel()
    bpm.load_state_dict(torch.load(ROOT / "assets" / "deeprhythm-0.7.pth",
                                   map_location="cpu", weights_only=True))
    bpm.eval()
    bpm_path = MODELS / "deeprhythm.onnx"
    # Input: HCQM (8 s clips, 6 harmonics, 240 bins, 8 bands), any number of clips.
    export(bpm, torch.rand(4, 6, 240, 8), bpm_path, "hcqm", {"hcqm": {0: "clips"}})
    bpm_err = compare(bpm, bpm_path, [rng.random((n, 6, 240, 8), dtype=np.float32)
                                      for n in (1, 7, 64)])

    print(f"{key_path.name}: {key_path.stat().st_size / 1e6:.1f} MB, deviation from torch {key_err:.2e}")
    print(f"{bpm_path.name}: {bpm_path.stat().st_size / 1e6:.1f} MB, deviation from torch {bpm_err:.2e}")
    if max(key_err, bpm_err) > 1e-3:
        print("Deviation is too large, check the export.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
