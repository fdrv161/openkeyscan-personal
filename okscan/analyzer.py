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
    downbeat: float | None = None   # первая сильная доля, секунды от начала файла
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


def _attack_time(y, t, sr, reach_s=0.045):
    """Точное начало атаки (в отсчётах) около момента t (в отсчётах) и её резкость.

    От самого резкого роста громкости в окне ±reach_s идём назад до ближайшей
    ямки, затем ищем первый отсчёт выше «фон + 10% прироста». RMS-кадры
    по 128 отсчётов размазывают атаку, поэтому последний шаг — по самому сигналу.
    """
    fine = 32
    reach = int(reach_s * sr)
    base = max(0, t - reach - 64)
    seg_y = y[base:t + reach + 64]
    rms = librosa.feature.rms(y=seg_y, frame_length=128, hop_length=fine, center=False)[0]
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
    # фон — пик предыдущих 10 мс: по медиане тянущиеся ноты (бас, пэд) своими
    # гребнями ложно «пересекали» порог раньше удара
    floor = float(np.abs(y[max(0, s0 - 221):s0]).max()) if s0 > 0 else 0.0
    above = np.nonzero(a > floor + 0.1 * (a.max() - floor))[0]
    return s0 + (int(above[0]) if len(above) else 0), float(diff[rise])


def beat_grid(y22, bpm):
    """Сетка долей трека: (первая сильная доля в секундах, уточнённый BPM) или (None, None).

    1) Грубо: сдвиг сетки ищем по атакам ниже 250 Гц (бочка, 808) — хэты на слабых
       долях иначе утягивают сетку на полдоли. Начало музыки — первая доля, где
       атака хотя бы вполовину от типичной.
    2) Точно: у каждой доли находим начало атаки с точностью до отсчёта и подбираем
       прямую «номер доли -> время», через которую проходит больше всего ударов
       (в пределах 3 мс). Биты из DAW стоят по сетке идеально, и прямая даёт и сдвиг,
       и темп до тысячных. Одиночный звук на первой доле (не кик) её не сбивает.
    Если ударов по прямой мало (живая игра, сложный ритм) — берём начало атаки
    первой доли как есть, а темп не трогаем.
    """
    sr, hop = dr_features.SR, 128
    if not bpm:
        return None, None
    # Тишина спереди: иначе удар в самой первой миллисекунде файла не даёт атаки
    # (огибающей не с чем сравнить) и первая доля пропускается. В конце её вычитаем.
    pad = int(0.5 * sr)
    y = np.concatenate([np.zeros(pad, dtype=y22.dtype), y22])
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=250, n_mels=16)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    period = 60.0 * sr / hop / bpm
    n = int((len(env) - 1) // period)
    if n < 8:
        return None, None
    phases = np.arange(0, period, 0.25)
    grid = phases[:, None] + np.arange(n)[None, :] * period
    phase = phases[np.argmax(np.interp(grid, np.arange(len(low)), low).sum(axis=1))]
    points = phase + np.arange(n) * period
    # сила доли — максимум огибающей в окне ±3 кадра (~17 мс) вокруг точки сетки
    strength = np.array([env[max(0, int(round(x)) - 3):int(round(x)) + 4].max() for x in points])
    typical = np.percentile(strength, 75)
    if typical <= 0:
        return None, None
    k0 = int(np.argmax(strength >= 0.5 * typical))
    t0 = points[k0] * hop                      # грубое начало музыки, в отсчётах
    step = period * hop                        # длина доли, в отсчётах

    ks, ts, ws = [], [], []
    for k in range(0, n - k0):
        t, w = _attack_time(y, int(round(t0 + k * step)), sr)
        if t is not None:
            ks.append(k), ts.append(t), ws.append(w)
    ks, ts, ws = np.array(ks), np.array(ts, dtype=float), np.array(ws)
    first, _ = _attack_time(y, int(round(t0)), sr, reach_s=0.03)
    fallback = (round(max(0.0, (first - pad) / sr), 4) if first is not None else None, None)
    keep = ws >= np.percentile(ws, 40)        # слабые «атаки» — это шум, не удары
    ks, ts = ks[keep], ts[keep]
    if len(ks) < 8:
        return fallback

    # Перебор темпа ±0,06 BPM и сдвига ±45 мс: сколько ударов ложится на прямую в пределах 3 мс.
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
    # Темп из DAW обычно «круглый»: 97,4999 -> 97,5. Ошибка округления в 0,01 BPM
    # за трек набегает меньше чем на пару миллисекунд.
    nice = round(fitted * 2) / 2
    if abs(fitted - nice) < 0.01:
        fitted = nice
    # Сдвиг пересчитываем под итоговый темп: при «округлённом» темпе старый сдвиг
    # прямой уже не точен. Медиана устойчива к паре неточных ударов.
    c0 = float(np.median(ts[inl] - ks[inl] * (60.0 * sr / fitted)))
    return round(max(0.0, (c0 - pad) / sr), 4), round(float(fitted), 3)


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
        key_model, bpm_model, kernels = self._models()
        with torch.inference_mode():
            class_id = int(key_model(key_spectrogram(y44).to(self.device)).argmax(dim=1)[0])

            y22 = librosa.resample(y44, orig_sr=KEY_SR, target_sr=dr_features.SR, res_type='soxr_hq')
            clips = dr_features.split_clips(y22, BPM_MAX_CLIPS)
            bpm = conf = downbeat = None
            if clips is not None:
                hcqm = dr_features.compute_hcqm(clips.to(self.device), *kernels)
                probs = torch.softmax(bpm_model(hcqm.to(self.device)), dim=1).mean(dim=0)
                p, cls = torch.max(probs, 0)
                bpm = round(refine_bpm(y22, class_to_bpm(int(cls))), 2)
                conf = round(float(p), 3)
                downbeat, fitted = beat_grid(y22, bpm)
                if fitted:
                    bpm = round(fitted, 2)   # темп по ударам точнее оценки сети
        return camelot_of(class_id), KEY_NAMES[class_id], bpm, conf, downbeat


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
