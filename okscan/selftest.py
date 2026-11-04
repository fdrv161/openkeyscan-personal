"""
Engine self-test on synthetic tracks with a known tempo and key.
Used by GitHub Actions and for a quick check after edits.

Run: python -m okscan.selftest  or  okscan-cli.exe --selftest
Exit code 0: all checks passed, 1: there are errors.
"""
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from okscan.analyzer import Analyzer
from okscan.library import format_bpm

SR = 44100


def synth_track(bpm, seconds=60, root_hz=261.63, minor=False):
    """Kick on every beat, hat on the offbeats and I-IV-V-I chords by bars."""
    n = int(seconds * SR)
    y = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    t_kick = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t_kick * 30)) * t_kick) * np.exp(-t_kick * 12)
    rng = np.random.default_rng(1)
    hat = rng.standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 120) * 0.2
    k = 0
    while True:
        pos = int(round(k * beat * SR))
        if pos >= n:
            break
        y[pos:pos + len(kick)] += kick[:n - pos]
        hpos = int(round((k + 0.5) * beat * SR))
        if hpos < n:
            y[hpos:hpos + len(hat)] += hat[:n - hpos]
        k += 1
    third = 3 if minor else 4
    # degrees I, IV, V, I in semitones
    for bar, degree in enumerate([0, 5, 7, 0] * int(seconds / (4 * beat) + 1)):
        start = int(bar * 4 * beat * SR)
        if start >= n:
            break
        end = min(n, int((bar + 1) * 4 * beat * SR))
        t = np.arange(end - start) / SR
        # all chords in one mode (natural minor/major): the key is unambiguous
        for semis in (0, third, 7, -12):
            f = root_hz * 2 ** ((degree + semis) / 12)
            y[start:end] += 0.08 * np.sin(2 * np.pi * f * t)
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def write_wav(path, y):
    """16-bit mono WAV with plain Python: no extra libraries in the build."""
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).round().astype('<i2').tobytes())


CASES = [
    # (BPM, key: tonic frequency, minor?, expected Camelot, silence before the first beat, s)
    (128.0, 261.63, False, '8B', 0.337),   # C major
    (174.0, 220.00, True, '8A', 0.0),      # A minor
    (122.5, 392.00, False, '9B', 0.0123),  # G major
    (93.0, 329.63, True, '9A', 1.234),     # E minor
]


def main():
    t0 = time.time()
    analyzer = Analyzer(workers=1)
    print(f"Models loaded in {time.time() - t0:.1f} s.")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for bpm, root, minor, camelot, lead in CASES:
            path = Path(tmp) / f"test_{bpm}.wav"
            silence = np.zeros(int(round(lead * SR)), dtype=np.float32)
            write_wav(path, np.concatenate([silence, synth_track(bpm, root_hz=root, minor=minor)]))
            t0 = time.time()
            # range 88-176: every test tempo stays as it is
            r = analyzer.analyze(path, bpm_low=88)
            took = time.time() - t0
            if r.error:
                print(f"ERROR: {path.name}: {r.error}")
                failures += 1
                continue
            bpm_ok = abs(r.bpm - bpm) <= 0.3
            key_ok = r.camelot == camelot
            # the first beat is needed for grid alignment in REAPER: tolerance of 2 ms
            beat_ok = r.downbeat is not None and abs(r.downbeat - lead) <= 0.002
            ok = bpm_ok and key_ok and beat_ok
            print(f"{'OK' if ok else 'FAILED'}: expected BPM {bpm}, {camelot}, first beat {lead} s; "
                  f"got BPM {format_bpm(r.bpm)} (raw {r.bpm_raw}, confidence {r.bpm_confidence}), "
                  f"{r.camelot} {r.key_name}, first beat {r.downbeat} s; {took:.1f} s.")
            failures += not ok
    print("Self-test passed." if not failures else f"Failed checks: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
