"""
Движок анализа: один раз декодирует файл и определяет по нему тональность и темп.

Тональность — модель OpenKeyScan (openkeyscan3.pt), предобработка как в
openkeyscan_analyzer_server.py, чтобы результаты совпадали с оригиналом.
Темп — DeepRhythm 0.7 (грубая оценка) + уточнение по автокорреляции атак (дробный BPM).

Файлы открываются только на чтение — никаких тегов и записи в аудио.
"""
import os
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import librosa
import av

from model import KeyNet
from okscan.deeprhythm import features as dr_features
from okscan.deeprhythm.model import DeepRhythmModel, class_to_bpm

# Безвредное предупреждение torch про padding='same' в DeepRhythm — только засоряет вывод.
warnings.filterwarnings('ignore', message="Using padding='same'", category=UserWarning)

SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.flac', '.ogg', '.m4a', '.aac',
                     '.aiff', '.aif', '.au', '.opus', '.wma', '.alac'}

KEY_SR = 44100

# Индекс класса сети тональности -> название (как в dataset.CAMELOT_MAPPING).
# 0–11 — минор (1A–12A), 12–23 — мажор (1B–12B).
KEY_NAMES = [
    'Ab minor', 'Eb minor', 'Bb minor', 'F minor', 'C minor', 'G minor',
    'D minor', 'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor',
    'B major', 'F# major', 'Db major', 'Ab major', 'Eb major', 'Bb major',
    'F major', 'C major', 'G major', 'D major', 'A major', 'E major',
]

# Не больше 64 кусков по 8 секунд (~8,5 минут) на оценку темпа: на длинных миксах
# иначе уходят гигабайты памяти. Обычный трек укладывается целиком.
BPM_MAX_CLIPS = 64
# Для уточнения BPM хватает 10 минут из середины трека.
REFINE_MAX_SECONDS = 600


def resource_path(relative):
    """Путь к файлу рядом с кодом или внутри сборки PyInstaller."""
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
    return base / relative


@dataclass
class Result:
    path: str
    camelot: str = ''
    openkey: str = ''
    key_name: str = ''
    bpm: float | None = None        # после приведения к диапазону
    bpm_raw: float | None = None    # как определил алгоритм
    bpm_confidence: float | None = None
    error: str = ''


def camelot_of(class_id):
    return f"{class_id % 12 + 1}{'A' if class_id < 12 else 'B'}"


def openkey_of(camelot):
    num, mode = int(camelot[:-1]), camelot[-1]
    return f"{(num - 8) % 12 + 1}{'m' if mode == 'A' else 'd'}"


def fold_bpm(bpm, low):
    """Приводит темп к диапазону [low, 2*low] умножением/делением на 2.
    low = 0 — не менять. Лечит самую частую ошибку: темп «вдвое» больше или меньше.
    Обе границы входят: при low=50 темп 100 остаётся 100, а 133 становится 66,5."""
    if bpm is None or not low:
        return bpm
    while bpm < low:
        bpm *= 2
    while bpm > 2 * low:
        bpm /= 2
    return bpm


def format_bpm(bpm):
    """128.0 -> «128», 127.46 -> «127.5». Десятые — потолок разумной точности."""
    if bpm is None:
        return ''
    text = f"{bpm:.1f}"
    return text[:-2] if text.endswith('.0') else text


def decode_audio(path):
    """Декодирует любой формат через PyAV (FFmpeg внутри) в моно float32, 44,1 кГц,
    пик 0,95 — так же, как оригинальный сервер делает для MP3 на Windows."""
    container = av.open(str(path))
    try:
        if not container.streams.audio:
            raise ValueError("в файле нет звуковой дорожки")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='flt', layout='mono', rate=KEY_SR)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray()[0])
    finally:
        container.close()
    if not chunks:
        raise ValueError("не удалось прочитать звук")
    y = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.95
    return y


def key_spectrogram(y44):
    """Лог-CQT с параметрами оригинала: 105 бинов, 24 на октаву, от 65 Гц, шаг 0,2 с."""
    cqt = librosa.cqt(y44, sr=KEY_SR, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    spec = np.log1p(np.abs(cqt))[:, :-2]   # как в оригинале: два последних кадра отбрасываются
    return torch.from_numpy(np.ascontiguousarray(spec, dtype=np.float32))[None, None]


def refine_bpm(y22, coarse):
    """Уточняет целый BPM от DeepRhythm до дробного.

    Сеть выдаёт класс шириной 1 BPM (нижнюю границу), поэтому ищем лучший темп
    в окне [coarse-1, coarse+2] по «гребёнке» автокорреляции огибающей атак:
    сумма автокорреляции на задержках 1..16 долей. Чем больше долей, тем точнее —
    ошибка в 0,1% на 16 долях уже заметно сдвигает пик.
    """
    sr = dr_features.SR
    hop = 128
    fps = sr / hop
    if len(y22) > REFINE_MAX_SECONDS * sr:
        start = (len(y22) - REFINE_MAX_SECONDS * sr) // 2
        y22 = y22[start:start + REFINE_MAX_SECONDS * sr]
    env = librosa.onset.onset_strength(y=y22, sr=sr, hop_length=hop)
    env = env - env.mean()
    candidates = np.arange(max(coarse - 1.0, 20.0), coarse + 2.0, 0.01)
    beats = np.arange(1, 17)
    lags = (60.0 * fps / candidates)[:, None] * beats[None, :]
    max_lag = int(np.ceil(lags.max())) + 2
    if max_lag >= len(env):
        return coarse
    ac = librosa.autocorrelate(env, max_size=max_lag + 1)
    scores = np.interp(lags, np.arange(len(ac)), ac).sum(axis=1)
    return float(candidates[np.argmax(scores)])


class Analyzer:
    """Потокобезопасный анализатор: у каждого рабочего потока свои копии моделей."""

    def __init__(self, workers=1, device='cpu'):
        self.device = torch.device(device)
        self.key_model_path = resource_path('checkpoints/openkeyscan3.pt')
        self.bpm_model_path = resource_path('assets/deeprhythm-0.7.pth')
        for p in (self.key_model_path, self.bpm_model_path):
            if not p.exists():
                raise FileNotFoundError(f"Не найден файл модели: {p}")
        # Делим ядра между рабочими потоками, чтобы они не мешали друг другу.
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(1, workers)))
        self._local = threading.local()
        self._warmup()

    def _models(self):
        loc = self._local
        if not hasattr(loc, 'key_model'):
            km = KeyNet(num_classes=24, in_channels=1, Nf=64, p=0.3)
            km.load_state_dict(torch.load(self.key_model_path, map_location=self.device, weights_only=True))
            loc.key_model = km.to(self.device).eval()
            bm = DeepRhythmModel()
            bm.load_state_dict(torch.load(self.bpm_model_path, map_location=self.device, weights_only=True))
            loc.bpm_model = bm.to(self.device).eval()
            loc.kernels = dr_features.make_kernels(self.device)
        return loc.key_model, loc.bpm_model, loc.kernels

    def _warmup(self):
        # Первый вызов librosa.cqt/onset компилирует код numba. Если это случится
        # одновременно в нескольких потоках, оригинальный сервер зависал —
        # поэтому прогреваем всё один раз здесь, до запуска рабочих потоков.
        t = np.arange(KEY_SR * 10) / KEY_SR
        y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        self._analyze_waveform(y)

    def analyze(self, path, bpm_low=0):
        res = Result(path=str(path))
        try:
            if Path(path).suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError("формат не поддерживается")
            y44 = decode_audio(path)
            if len(y44) < KEY_SR * 3:
                raise ValueError("трек короче 3 секунд")
            camelot, key_name, bpm_raw, conf = self._analyze_waveform(y44)
            res.camelot, res.key_name = camelot, key_name
            res.openkey = openkey_of(camelot)
            res.bpm_raw, res.bpm_confidence = bpm_raw, conf
            res.bpm = fold_bpm(bpm_raw, bpm_low)
        except Exception as e:
            res.error = describe_error(e)
        return res

    def _analyze_waveform(self, y44):
        key_model, bpm_model, kernels = self._models()
        with torch.inference_mode():
            class_id = int(key_model(key_spectrogram(y44).to(self.device)).argmax(dim=1)[0])

            y22 = librosa.resample(y44, orig_sr=KEY_SR, target_sr=dr_features.SR, res_type='soxr_hq')
            clips = dr_features.split_clips(y22, BPM_MAX_CLIPS)
            bpm = conf = None
            if clips is not None:
                hcqm = dr_features.compute_hcqm(clips.to(self.device), *kernels)
                probs = torch.softmax(bpm_model(hcqm.to(self.device)), dim=1).mean(dim=0)
                p, cls = torch.max(probs, 0)
                bpm = round(refine_bpm(y22, class_to_bpm(int(cls))), 2)
                conf = round(float(p), 3)
        return camelot_of(class_id), KEY_NAMES[class_id], bpm, conf


def describe_error(e):
    """Понятный текст ошибки для человека, без трассировки."""
    if isinstance(e, FileNotFoundError):
        return "файл не найден"
    if isinstance(e, PermissionError):
        return "нет доступа к файлу"
    if isinstance(e, av.error.InvalidDataError):
        return "файл повреждён или это не аудио"
    if isinstance(e, MemoryError):
        return "не хватило памяти (файл слишком длинный)"
    text = str(e).strip() or type(e).__name__
    return text
