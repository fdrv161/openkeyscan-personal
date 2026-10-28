"""
Переводит обе сети из формата torch в ONNX — их запускает движок через onnxruntime.

Нужно только разработчику, когда меняются веса: в программу torch не попадает.
Зависимости: pip install -r requirements-dev.txt; веса DeepRhythm — python tools/fetch_assets.py.
Запуск: python tools/export_onnx.py
Результат: models/keynet.onnx и models/deeprhythm.onnx (лежат в git).
Код выхода 0 — оба файла записаны и выдают то же, что torch; 1 — расхождение.
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
    # Старый экспортёр (dynamo=False): ему не нужен onnxscript, а для простых
    # свёрточных сетей результат тот же. Веса — внутри файла, без .data рядом.
    torch.onnx.export(model, (example,), str(path), dynamo=False, opset_version=OPSET,
                      input_names=[input_name], output_names=["logits"],
                      dynamic_axes=dynamic_axes)


def compare(model, path, inputs):
    """Сравнивает выход torch и onnxruntime на нескольких входах, возвращает худшее отклонение."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    worst = 0.0
    with torch.inference_mode():
        for x in inputs:
            ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(None, {name: x})[0]
            if ref.argmax(axis=1).tolist() != got.argmax(axis=1).tolist():
                raise RuntimeError(f"{path.name}: другой ответ сети на входе {x.shape}")
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
    # Вход: лог-CQT (1 трек, 1 канал, 105 бинов, кадры по 0,2 с — длина любая).
    export(key, torch.rand(1, 1, 105, 900), key_path, "spec", {"spec": {3: "frames"}})
    key_err = compare(key, key_path, [rng.random((1, 1, 105, n), dtype=np.float32) * 3
                                      for n in (40, 900, 2500)])

    bpm = DeepRhythmModel()
    bpm.load_state_dict(torch.load(ROOT / "assets" / "deeprhythm-0.7.pth",
                                   map_location="cpu", weights_only=True))
    bpm.eval()
    bpm_path = MODELS / "deeprhythm.onnx"
    # Вход: HCQM (куски по 8 с, 6 гармоник, 240 бинов, 8 полос), число кусков любое.
    export(bpm, torch.rand(4, 6, 240, 8), bpm_path, "hcqm", {"hcqm": {0: "clips"}})
    bpm_err = compare(bpm, bpm_path, [rng.random((n, 6, 240, 8), dtype=np.float32)
                                      for n in (1, 7, 64)])

    print(f"{key_path.name}: {key_path.stat().st_size / 1e6:.1f} МБ, отклонение от torch {key_err:.2e}")
    print(f"{bpm_path.name}: {bpm_path.stat().st_size / 1e6:.1f} МБ, отклонение от torch {bpm_err:.2e}")
    if max(key_err, bpm_err) > 1e-3:
        print("Отклонение слишком большое, проверьте экспорт.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
