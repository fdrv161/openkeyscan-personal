"""
Сверка обработки звука на numpy (okscan/dsp.py, deeprhythm/features.py) с оригиналами:
librosa 0.11 и nnAudio 0.3.3 на torch. Нужна после правок этих файлов.

Зависимости: pip install -r requirements-dev.txt.
Запуск: python tools/check_parity.py ФАЙЛ [ФАЙЛ ...]
  Для каждого файла печатает наибольшее относительное расхождение по каждому шагу.
  --ref ФАЙЛ.tsv — ещё сравнить итог (BPM, тональность, первая доля) с результатом
  старой сборки: okscan-cli --list files.txt --out ФАЙЛ.tsv --range 0 --no-cache
Код выхода 0 — всё совпало, 1 — есть расхождения.
"""
import argparse
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import torch  # noqa: E402

from okscan import analyzer as an  # noqa: E402
from okscan import dsp  # noqa: E402
from okscan.deeprhythm import features as feat  # noqa: E402
from okscan.library import format_bpm  # noqa: E402

# Последний коммит, где признаки DeepRhythm считались через nnAudio.
TORCH_COMMIT = "c65990b"
TOL = 1e-4   # допустимое относительное расхождение (шум float32)


def rel(a, b):
    """Наибольшее расхождение, отнесённое к размаху эталона."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float('inf')
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def old_features():
    """Модуль признаков DeepRhythm в том виде, в каком он был на torch и nnAudio."""
    src = subprocess.run(['git', 'show', f'{TORCH_COMMIT}:okscan/deeprhythm/features.py'],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8')
    mod = types.ModuleType('old_features')
    exec(compile(src, 'old_features.py', 'exec'), mod.__dict__)
    return mod


def check_file(path, old):
    y44, _ = an.decode_audio(path)
    y22 = dsp.resample(y44, an.KEY_SR, feat.SR)
    ref22 = librosa.resample(y44, orig_sr=an.KEY_SR, target_sr=feat.SR, res_type='soxr_hq')
    steps = {'передискретизация 44,1 -> 22,05 кГц': rel(y22, ref22)}

    cqt = dsp.cqt(y44, sr=an.KEY_SR, hop=8820, fmin=65, n_bins=105, bpo=24)
    ref = librosa.cqt(y44, sr=an.KEY_SR, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    steps['CQT для тональности'] = rel(np.abs(cqt), np.abs(ref))

    power = dsp.power_spectrum(ref22, an.ONSET_NFFT, an.ONSET_HOP)
    ref_power = np.abs(librosa.stft(ref22, n_fft=an.ONSET_NFFT, hop_length=an.ONSET_HOP)) ** 2
    steps['спектр мощности'] = rel(power, ref_power)
    for name, mel in (('огибающая атак', {}), ('огибающая баса', {'fmax': 250, 'n_mels': 16})):
        env = an._onset_env(ref_power, **mel)
        S = librosa.feature.melspectrogram(S=ref_power, sr=feat.SR, n_fft=an.ONSET_NFFT, **mel)
        ref_env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=feat.SR,
                                               hop_length=an.ONSET_HOP)
        steps[name] = rel(env, ref_env)
    env = ref_env - ref_env.mean()
    steps['автокорреляция'] = rel(dsp.autocorrelate(env, 3000),
                                  librosa.autocorrelate(env, max_size=3000))
    seg = ref22[feat.SR * 5:feat.SR * 5 + 2000]
    steps['RMS'] = rel(dsp.rms(seg, 128, 32),
                       librosa.feature.rms(y=seg, frame_length=128, hop_length=32, center=False)[0])

    clips = feat.split_clips(ref22, an.BPM_MAX_CLIPS)
    if clips is not None:
        hcqm = feat.compute_hcqm(clips, *feat.make_kernels())
        with torch.inference_mode():
            ref_hcqm = old.compute_hcqm(torch.from_numpy(clips), *old.make_kernels()).numpy()
        steps['HCQM для DeepRhythm'] = rel(hcqm, ref_hcqm)
    return steps


def check_results(paths, ref_tsv):
    """Итог нового движка против TSV старой сборки. Возвращает число расхождений."""
    ref = {}
    for line in Path(ref_tsv).read_text(encoding='utf-8').splitlines():
        cols = line.split('\t')
        ref[Path(cols[0]).name] = cols
    analyzer = an.Analyzer()
    bad = 0
    for path in paths:
        r = analyzer.analyze(path)
        old = ref.get(Path(path).name)
        if old is None:
            print(f"{Path(path).name}: нет в {ref_tsv}")
            continue
        new = [format_bpm(r.bpm) if r.bpm else '', r.camelot or '', r.error or '',
               f"{r.downbeat:.4f}" if r.downbeat is not None else '',
               f"{r.bpm_confidence}" if r.bpm_confidence is not None else '']
        was = [old[1], old[2], old[5], old[6], old[8] if len(old) > 8 else '']
        same = new == was
        bad += not same
        print(f"{'совпало' if same else 'РАЗНИЦА'}: {Path(path).name}: "
              f"было {', '.join(was)}; стало {', '.join(new)}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('files', nargs='+')
    p.add_argument('--ref', help="TSV старой сборки для сверки итога")
    args = p.parse_args()
    old = old_features()
    bad = 0
    for path in args.files:
        print(f"Файл: {Path(path).name}")
        try:
            y_len = len(an.decode_audio(path)[0])
        except Exception as e:
            print(f"  пропущен: {an.describe_error(e)}")
            continue
        if y_len < an.KEY_SR * 3:
            print("  пропущен: короче 3 секунд")
            continue
        for name, err in check_file(path, old).items():
            ok = err <= TOL
            bad += not ok
            print(f"  {name}: расхождение {err:.1e}{'' if ok else ' — БОЛЬШЕ ДОПУСКА'}")
    if args.ref:
        bad += check_results(args.files, args.ref)
    print("Всё совпало." if not bad else f"Расхождений: {bad}.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
