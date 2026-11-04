"""
Analysis engine: decodes a file once and detects its key and tempo from it.

Key: the OpenKeyScan model (openkeyscan3.pt), preprocessing as in
openkeyscan_analyzer_server.py, so that results match the original.
Tempo: DeepRhythm 0.7 (coarse estimate) + refinement by onset autocorrelation (fractional BPM).
Both networks are converted to ONNX (tools/export_onnx.py) and run through onnxruntime,
audio processing is numpy (okscan/dsp.py): without torch, librosa and numba startup is much faster.

Files are opened read-only: no tags, nothing is ever written to the audio.
"""
import os
import sys
from pathlib import Path

import av
import numpy as np
import onnxruntime as ort

from okscan import dsp
from okscan.deeprhythm import features as dr_features
from okscan.deeprhythm.classes import class_to_bpm
from okscan.library import Result, SUPPORTED_FORMATS, fold_bpm

KEY_SR = 44100

# Key network class index -> name (as in dataset.CAMELOT_MAPPING).
# 0-11 are minor (1A-12A), 12-23 are major (1B-12B).
KEY_NAMES = [
    'Ab minor', 'Eb minor', 'Bb minor', 'F minor', 'C minor', 'G minor',
    'D minor', 'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor',
    'B major', 'F# major', 'Db major', 'Ab major', 'Eb major', 'Bb major',
    'F major', 'C major', 'G major', 'D major', 'A major', 'E major',
]

# At most 64 clips of 8 seconds (~8.5 minutes) for the tempo estimate: long mixes
# would otherwise take gigabytes of memory. A normal track fits entirely.
BPM_MAX_CLIPS = 64
# 10 minutes from the middle of the track are enough to refine the BPM.
REFINE_MAX_SECONDS = 600

# The "sound starts here" threshold (-50 dBFS), the same as in the REAPER script (LEAD_LEVEL).
LEAD_LEVEL = 10 ** (-50 / 20)

# Onset envelopes: hop of 128 samples (~5.8 ms at 22,050 Hz), window of 2048.
ONSET_HOP = 128
ONSET_NFFT = 2048
# Silence before the track for beat_grid: exactly 86 frames (~0.5 s). Being a multiple of the
# hop matters: spectrum frames with and without silence then match with a shift of 86, so one
# spectrum serves both refine_bpm and beat_grid (three used to be computed, the slowest part).
PAD_FRAMES = 86


def resource_path(relative):
    """Path to a file next to the code or inside the PyInstaller build."""
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
    return base / relative


def camelot_of(class_id):
    return f"{class_id % 12 + 1}{'A' if class_id < 12 else 'B'}"


def openkey_of(camelot):
    num, mode = int(camelot[:-1]), camelot[-1]
    return f"{(num - 8) % 12 + 1}{'m' if mode == 'A' else 'd'}"


def _first_sound(frame):
    """Index of the first sample in the frame louder than LEAD_LEVEL on any channel, or None."""
    a = frame.to_ndarray()
    if not frame.format.is_planar:
        a = a.reshape(-1, len(frame.layout.channels)).T
    if a.dtype.kind == 'i':
        a = a / np.iinfo(a.dtype).max
    hit = np.nonzero((np.abs(a) > LEAD_LEVEL).any(axis=0))[0]
    return int(hit[0]) if len(hit) else None


def decode_audio(path):
    """Decodes any format through PyAV (FFmpeg inside) to mono float32, 44.1 kHz,
    peak 0.95: the same as the original server does for MP3 on Windows.

    Returns (audio, start of sound in seconds). The start is measured on the source
    samples before the mono downmix, as the REAPER script measures it, so the difference
    shows whether FFmpeg and REAPER trim the start of the file differently."""
    container = av.open(str(path))
    try:
        if not container.streams.audio:
            raise ValueError("the file has no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='flt', layout='mono', rate=KEY_SR)
        chunks = []
        pos, lead = 0, None
        for frame in container.decode(stream):
            if lead is None:
                i = _first_sound(frame)
                if i is not None:
                    lead = (pos + i) / frame.sample_rate
                pos += frame.samples
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray()[0])
    finally:
        container.close()
    if not chunks:
        raise ValueError("could not read the audio")
    y = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.95
    return y, lead


def key_spectrogram(y44):
    """Log-CQT with the original's parameters: 105 bins, 24 per octave, from 65 Hz, hop 0.2 s."""
    cqt = dsp.cqt(y44, sr=KEY_SR, hop=8820, fmin=65, n_bins=105, bpo=24)
    spec = np.log1p(np.abs(cqt))[:, :-2]   # as in the original: the last two frames are dropped
    return np.ascontiguousarray(spec, dtype=np.float32)[None, None]


def _onset_env(power, **mel):
    """Onset envelope from a ready power spectrum: the same as
    librosa.onset.onset_strength(y=...), but without a second STFT."""
    S = dsp.mel_filters(dr_features.SR, ONSET_NFFT, **mel) @ power
    return dsp.onset_strength(dsp.power_to_db(S), n_fft=ONSET_NFFT, hop=ONSET_HOP)


def onset_envelopes(y22):
    """Onset envelopes from one spectrum: (full without silence, full with leading silence,
    bass below 250 Hz with silence). beat_grid needs the leading silence, see PAD_FRAMES."""
    y = np.concatenate([np.zeros(PAD_FRAMES * ONSET_HOP, dtype=y22.dtype), y22])
    power = dsp.power_spectrum(y, ONSET_NFFT, ONSET_HOP)
    return (_onset_env(power[:, PAD_FRAMES:]), _onset_env(power),
            _onset_env(power, fmax=250, n_mels=16))


def refine_bpm(y22, coarse, env=None):
    """Refines the integer BPM from DeepRhythm to a fractional one.

    The network outputs a class 1 BPM wide (its lower bound), so we look for the best tempo
    in the window [coarse-1, coarse+2] with an autocorrelation "comb" of the onset envelope:
    the sum of autocorrelation at lags of 1..16 beats. More beats mean more precision:
    a 0.1% error over 16 beats already shifts the peak noticeably.
    env is a ready envelope of the whole track from onset_envelopes (optional).
    """
    sr = dr_features.SR
    hop = ONSET_HOP
    fps = sr / hop
    if len(y22) > REFINE_MAX_SECONDS * sr:
        start = (len(y22) - REFINE_MAX_SECONDS * sr) // 2
        y22 = y22[start:start + REFINE_MAX_SECONDS * sr]
        env = None   # the ready envelope covers the whole track, but the middle is needed
    if env is None:
        env = _onset_env(dsp.power_spectrum(y22, ONSET_NFFT, hop))
    env = env - env.mean()
    candidates = np.arange(max(coarse - 1.0, 20.0), coarse + 2.0, 0.01)
    beats = np.arange(1, 17)
    lags = (60.0 * fps / candidates)[:, None] * beats[None, :]
    max_lag = int(np.ceil(lags.max())) + 2
    if max_lag >= len(env):
        return coarse
    ac = dsp.autocorrelate(env, max_size=max_lag + 1)
    scores = np.interp(lags, np.arange(len(ac)), ac).sum(axis=1)
    return float(candidates[np.argmax(scores)])


def _attack_time(y, t, sr, reach_s=0.045):
    """Exact attack start (in samples) near moment t (in samples) and its sharpness.

    From the sharpest loudness rise in the ±reach_s window we walk back to the nearest
    dip, then look for the first sample above "floor + 10% of the rise". RMS frames of
    128 samples smear the attack, so the last step works on the signal itself.
    """
    fine = 32
    reach = int(reach_s * sr)
    base = max(0, t - reach - 64)
    seg_y = y[base:t + reach + 64]
    rms = dsp.rms(seg_y, frame_length=128, hop=fine)
    if len(rms) < 3:
        return None, 0.0
    diff = np.diff(rms)
    rise = int(np.argmax(diff))
    start = rise
    while start > 0 and rms[start - 1] < rms[start]:
        start -= 1
    s0 = base + start * fine
    a = np.abs(y[s0:base + rise * fine + 256])
    if len(a) == 0:
        return s0, float(diff[rise])
    # floor is the peak of the previous 10 ms: with a median, the crests of sustained
    # notes (bass, pads) falsely "crossed" the threshold before the hit
    floor = float(np.abs(y[max(0, s0 - 221):s0]).max()) if s0 > 0 else 0.0
    above = np.nonzero(a > floor + 0.1 * (a.max() - floor))[0]
    return s0 + (int(above[0]) if len(above) else 0), float(diff[rise])


def beat_grid(y22, bpm, env=None, low=None):
    """Beat grid of a track: (first downbeat in seconds, refined BPM) or (None, None).
    env, low are the track's envelopes with leading silence from onset_envelopes (optional).

    1) Coarse: the grid offset comes from attacks below 250 Hz (kick, 808), otherwise hats
       on weak beats pull the grid half a beat away. The music starts at the first beat
       whose attack is at least half the typical one.
    2) Fine: for each beat we find the attack start to the sample and fit the line
       "beat number -> time" that passes through the most hits (within 3 ms).
       Beats from a DAW sit on the grid perfectly, so the line gives both the offset
       and the tempo to thousandths. A lone non-kick sound on the first beat cannot shift it.
    If few hits lie on the line (live playing, complex rhythm), the attack start of the
    first beat is taken as is and the tempo is left alone.
    """
    sr, hop = dr_features.SR, ONSET_HOP
    if not bpm:
        return None, None
    # Leading silence: otherwise a hit in the very first millisecond gives no attack
    # (the envelope has nothing to compare with) and the first beat is missed. Subtracted at the end.
    pad = PAD_FRAMES * hop
    y = np.concatenate([np.zeros(pad, dtype=y22.dtype), y22])
    if env is None or low is None:
        _, env, low = onset_envelopes(y22)
    period = 60.0 * sr / hop / bpm
    n = int((len(env) - 1) // period)
    if n < 8:
        return None, None
    phases = np.arange(0, period, 0.25)
    grid = phases[:, None] + np.arange(n)[None, :] * period
    phase = phases[np.argmax(np.interp(grid, np.arange(len(low)), low).sum(axis=1))]
    points = phase + np.arange(n) * period
    # beat strength is the envelope maximum within ±3 frames (~17 ms) around the grid point
    strength = np.array([env[max(0, int(round(x)) - 3):int(round(x)) + 4].max() for x in points])
    typical = np.percentile(strength, 75)
    if typical <= 0:
        return None, None
    k0 = int(np.argmax(strength >= 0.5 * typical))
    t0 = points[k0] * hop                      # coarse start of the music, in samples
    step = period * hop                        # beat length, in samples

    ks, ts, ws = [], [], []
    for k in range(0, n - k0):
        t, w = _attack_time(y, int(round(t0 + k * step)), sr)
        if t is not None:
            ks.append(k), ts.append(t), ws.append(w)
    ks, ts, ws = np.array(ks), np.array(ts, dtype=float), np.array(ws)
    first, _ = _attack_time(y, int(round(t0)), sr, reach_s=0.03)
    fallback = (round(max(0.0, (first - pad) / sr), 4) if first is not None else None, None)
    keep = ws >= np.percentile(ws, 40)        # weak "attacks" are noise, not hits
    ks, ts = ks[keep], ts[keep]
    if len(ks) < 8:
        return fallback

    # Search tempo ±0.06 BPM and offset ±45 ms: how many hits fall on the line within 3 ms.
    tol = 0.003 * sr
    offsets = np.arange(-0.045, 0.045, 0.0005) * sr
    best_count, best = 0, None
    for b in np.arange(bpm - 0.06, bpm + 0.06, 0.0005):
        p = 60.0 * sr / b
        pred = t0 + offsets[:, None] + ks[None, :] * p
        counts = (np.abs(ts[None, :] - pred) < tol).sum(axis=1)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), (p, offsets[i])
    if best_count < max(8, 0.25 * len(ks)):
        return fallback
    p, off = best
    inl = np.abs(ts - (t0 + off + ks * p)) < tol
    c0, c1 = np.linalg.lstsq(np.vstack([np.ones(inl.sum()), ks[inl]]).T, ts[inl], rcond=None)[0]
    fitted = 60.0 * sr / c1
    # Tempo from a DAW is usually "round": 97.4999 -> 97.5. A rounding error of 0.01 BPM
    # adds up to less than a couple of milliseconds over a track.
    nice = round(fitted * 2) / 2
    if abs(fitted - nice) < 0.01:
        fitted = nice
    # Recompute the offset for the final tempo: with a "rounded" tempo the old line
    # offset is no longer exact. The median is robust to a few imprecise hits.
    c0 = float(np.median(ts[inl] - ks[inl] * (60.0 * sr / fitted)))
    return round(max(0.0, (c0 - pad) / sr), 4), round(float(fitted), 3)


class Analyzer:
    """Analyzer. onnxruntime sessions are thread-safe, so all worker threads share them."""

    def __init__(self, workers=1):
        paths = [resource_path('models/keynet.onnx'), resource_path('models/deeprhythm.onnx')]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
        opts = ort.SessionOptions()
        # Split the cores between worker threads so they do not get in each other's way.
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // max(1, workers))
        opts.log_severity_level = 3   # errors only: onnxruntime warnings clutter the output
        self.key_model, self.bpm_model = (
            ort.InferenceSession(str(p), opts, providers=['CPUExecutionProvider']) for p in paths)
        self.kernels = dr_features.make_kernels()

    def analyze(self, path, bpm_low=0):
        res = Result(path=str(path))
        try:
            if Path(path).suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError("unsupported format")
            y44, res.lead = decode_audio(path)
            if len(y44) < KEY_SR * 3:
                raise ValueError("track is shorter than 3 seconds")
            camelot, key_name, bpm_raw, conf, downbeat = self._analyze_waveform(y44)
            res.camelot, res.key_name = camelot, key_name
            res.openkey = openkey_of(camelot)
            res.bpm_raw, res.bpm_confidence = bpm_raw, conf
            res.downbeat = downbeat
            res.bpm = fold_bpm(bpm_raw, bpm_low)
        except Exception as e:
            res.error = describe_error(e)
        return res

    def _analyze_waveform(self, y44):
        logits = self.key_model.run(None, {'spec': key_spectrogram(y44)})[0]
        class_id = int(logits.argmax(axis=1)[0])

        y22 = dsp.resample(y44, KEY_SR, dr_features.SR)
        clips = dr_features.split_clips(y22, BPM_MAX_CLIPS)
        bpm = conf = downbeat = None
        if clips is not None:
            hcqm = dr_features.compute_hcqm(clips, *self.kernels)
            logits = self.bpm_model.run(None, {'hcqm': hcqm})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))   # softmax over classes
            probs = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
            cls = int(np.argmax(probs))
            env, env_pad, low_pad = onset_envelopes(y22)
            bpm = round(refine_bpm(y22, class_to_bpm(cls), env), 2)
            conf = round(float(probs[cls]), 3)
            downbeat, fitted = beat_grid(y22, bpm, env_pad, low_pad)
            if fitted:
                bpm = round(fitted, 2)   # tempo from the hits beats the network's estimate
        return camelot_of(class_id), KEY_NAMES[class_id], bpm, conf, downbeat


def describe_error(e):
    """Readable error text for a person, without a traceback."""
    if isinstance(e, FileNotFoundError):
        return "file not found"
    if isinstance(e, PermissionError):
        return "no access to the file"
    if isinstance(e, av.error.InvalidDataError):
        return "the file is damaged or is not audio"
    if isinstance(e, MemoryError):
        return "out of memory (the file is too long)"
    text = str(e).strip() or type(e).__name__
    return text
