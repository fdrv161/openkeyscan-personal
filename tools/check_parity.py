"""
Compares the numpy audio processing (okscan/dsp.py, deeprhythm/features.py) with the originals:
librosa 0.11 and nnAudio 0.3.3 on torch. Run it after editing those files.

Dependencies: pip install -r requirements-dev.txt.
Run: python tools/check_parity.py FILE [FILE ...]
  Prints the largest relative difference at each step for each file.
  --ref FILE.tsv also compares the final result (BPM, key, first beat) with the result
  of an older build: okscan-cli --list files.txt --out FILE.tsv --range 0 --no-cache
Exit code 0: everything matches, 1: there are differences.
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

# The last commit where DeepRhythm features were computed through nnAudio.
TORCH_COMMIT = "c65990b"
TOL = 1e-4   # allowed relative difference (float32 noise)


def rel(a, b):
    """Largest difference relative to the reference's scale."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float('inf')
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def old_features():
    """The DeepRhythm features module as it was on torch and nnAudio."""
    src = subprocess.run(['git', 'show', f'{TORCH_COMMIT}:okscan/deeprhythm/features.py'],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8')
    mod = types.ModuleType('old_features')
    exec(compile(src, 'old_features.py', 'exec'), mod.__dict__)
    return mod


def check_file(path, old):
    y44, _ = an.decode_audio(path)
    y22 = dsp.resample(y44, an.KEY_SR, feat.SR)
    ref22 = librosa.resample(y44, orig_sr=an.KEY_SR, target_sr=feat.SR, res_type='soxr_hq')
    steps = {'resampling 44.1 -> 22.05 kHz': rel(y22, ref22)}

    cqt = dsp.cqt(y44, sr=an.KEY_SR, hop=8820, fmin=65, n_bins=105, bpo=24)
    ref = librosa.cqt(y44, sr=an.KEY_SR, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    steps['CQT for key'] = rel(np.abs(cqt), np.abs(ref))

    ref_power = np.abs(librosa.stft(ref22, n_fft=an.ONSET_NFFT, hop_length=an.ONSET_HOP)) ** 2
    for name, mel in (('onset envelope', {}), ('bass envelope', {'fmax': 250, 'n_mels': 16})):
        filters = dsp.mel_filters(feat.SR, an.ONSET_NFFT, **mel)
        new_S = dsp.mel_spectra(ref22, an.ONSET_NFFT, an.ONSET_HOP, [filters])[0]
        S = librosa.feature.melspectrogram(S=ref_power, sr=feat.SR, n_fft=an.ONSET_NFFT, **mel)
        steps[f'{name}: mel spectrum'] = rel(new_S, S)
        env = an._onset_from_mel(new_S)
        ref_env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=feat.SR,
                                               hop_length=an.ONSET_HOP)
        steps[name] = rel(env, ref_env)
    env = ref_env - ref_env.mean()
    steps['autocorrelation'] = rel(dsp.autocorrelate(env, 3000),
                                  librosa.autocorrelate(env, max_size=3000))
    seg = ref22[feat.SR * 5:feat.SR * 5 + 2000]
    steps['RMS'] = rel(dsp.rms(seg, 128, 32),
                       librosa.feature.rms(y=seg, frame_length=128, hop_length=32, center=False)[0])

    clips = feat.split_clips(ref22, an.BPM_MAX_CLIPS)
    if clips is not None:
        hcqm = feat.compute_hcqm(clips, *feat.make_kernels())
        with torch.inference_mode():
            ref_hcqm = old.compute_hcqm(torch.from_numpy(clips), *old.make_kernels()).numpy()
        steps['HCQM for DeepRhythm'] = rel(hcqm, ref_hcqm)
    return steps


def check_results(paths, ref_tsv):
    """New engine's result against the older build's TSV. Returns the number of differences."""
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
            print(f"{Path(path).name}: not in {ref_tsv}")
            continue
        new = [format_bpm(r.bpm) if r.bpm else '', r.camelot or '', r.error or '',
               f"{r.downbeat:.4f}" if r.downbeat is not None else '',
               f"{r.bpm_confidence}" if r.bpm_confidence is not None else '']
        was = [old[1], old[2], old[5], old[6], old[8] if len(old) > 8 else '']
        same = new == was
        bad += not same
        print(f"{'match' if same else 'DIFFERENT'}: {Path(path).name}: "
              f"was {', '.join(was)}; now {', '.join(new)}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('files', nargs='+')
    p.add_argument('--ref', help="TSV of an older build to compare the final result with")
    args = p.parse_args()
    old = old_features()
    bad = 0
    for path in args.files:
        print(f"File: {Path(path).name}")
        try:
            y_len = len(an.decode_audio(path)[0])
        except Exception as e:
            print(f"  skipped: {an.describe_error(e)}")
            continue
        if y_len < an.KEY_SR * 3:
            print("  skipped: shorter than 3 seconds")
            continue
        for name, err in check_file(path, old).items():
            ok = err <= TOL
            bad += not ok
            print(f"  {name}: difference {err:.1e}{'' if ok else ' - ABOVE TOLERANCE'}")
    if args.ref:
        bad += check_results(args.files, args.ref)
    print("Everything matches." if not bad else f"Differences: {bad}.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
