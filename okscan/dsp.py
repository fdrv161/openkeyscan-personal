"""
Обработка звука на numpy (+ soxr) вместо librosa 0.11 и nnAudio 0.3.3.

Зачем: librosa тянет numba, llvmlite, scipy и scikit-learn — это сотни мегабайт
в сборке и секунды на импорт при каждом запуске. Здесь только то, что нужно движку.

Формулы перенесены из исходников один в один, вплоть до типов данных: на CQT librosa
обучалась сеть тональности, на HCQM из nnAudio — DeepRhythm, и любое «улучшение»
сдвинуло бы ответы сетей. Сверка с оригиналами — tools/check_parity.py.
Лицензии оригиналов: librosa — ISC, nnAudio — MIT.
"""
from functools import lru_cache

import numpy as np
import soxr

# Кадров STFT за раз: целиком длинный трек с шагом 128 занял бы гигабайты.
STFT_BLOCK = 1024


def hann(n):
    """Периодическое окно Ханна — то же, что scipy.signal.get_window('hann', n)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def fix_length(y, size):
    """Обрезает или дополняет нулями до size отсчётов (librosa.util.fix_length)."""
    if len(y) > size:
        return y[:size]
    return np.pad(y, (0, size - len(y)))


def resample(y, orig_sr, target_sr, scale=False):
    """librosa.resample(y, res_type='soxr_hq'): длина ровно ceil(n · ratio)."""
    ratio = float(target_sr) / orig_sr
    y_hat = fix_length(soxr.resample(y, orig_sr, target_sr, quality='soxr_hq'),
                       int(np.ceil(len(y) * ratio)))
    if scale:
        y_hat = y_hat / np.sqrt(ratio)
    return np.asarray(y_hat, dtype=y.dtype)


def _frames(y, n_fft, hop):
    """Кадры как в librosa.stft(center=True, pad_mode='constant'): по n_fft//2 нулей
    с краёв, кадров 1 + len(y) // hop. Возвращает вид (кадры, n_fft) без копирования."""
    y = np.pad(y, n_fft // 2)
    return np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]


def _stft_blocks(y, n_fft, hop, window):
    # Окно float64, кадры float32 — librosa перемножает именно так и считает БПФ
    # в двойной точности, а хранит результат в complex64.
    frames = _frames(y, n_fft, hop)
    for s in range(0, len(frames), STFT_BLOCK):
        yield s, np.fft.rfft(window * frames[s:s + STFT_BLOCK], axis=-1).astype(np.complex64)


def stft(y, n_fft, hop, window):
    """librosa.stft: форма (n_fft//2 + 1, кадры), complex64."""
    n = 1 + len(y) // hop
    out = np.empty((n_fft // 2 + 1, n), dtype=np.complex64)
    for s, block in _stft_blocks(y, n_fft, hop, window):
        out[:, s:s + len(block)] = block.T
    return out


def power_spectrum(y, n_fft, hop):
    """np.abs(librosa.stft(y, n_fft, hop)) ** 2 с окном Ханна, float32 —
    по кускам, без огромного комплексного массива в памяти."""
    n = 1 + len(y) // hop
    out = np.empty((n_fft // 2 + 1, n), dtype=np.float32)
    for s, block in _stft_blocks(y, n_fft, hop, hann(n_fft)):
        out[:, s:s + len(block)] = (np.abs(block) ** 2).T
    return out


# --- Мел-шкала Слейни (librosa: htk=False, norm='slaney') ---
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(f):
    f = np.asarray(f, dtype=float)
    log = _MIN_LOG_MEL + np.log(np.maximum(f, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(f >= _MIN_LOG_HZ, log, f / _F_SP)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    log = _MIN_LOG_HZ * np.exp(_LOGSTEP * (np.maximum(m, _MIN_LOG_MEL) - _MIN_LOG_MEL))
    return np.where(m >= _MIN_LOG_MEL, log, _F_SP * m)


@lru_cache(maxsize=8)
def mel_filters(sr, n_fft, n_mels=128, fmax=None):
    """librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax): (n_mels, n_fft//2 + 1), float32."""
    if fmax is None:
        fmax = float(sr) / 2
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    weights.flags.writeable = False   # общий кэш: случайно испортить нельзя
    return weights


def power_to_db(S, amin=1e-10, top_db=80.0):
    """librosa.power_to_db(S, ref=1.0): порог top_db — от максимума всего спектра."""
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    return np.maximum(log_spec, log_spec.max() - top_db)


def onset_strength(S_db, n_fft=2048, hop=512, lag=1):
    """librosa.onset.onset_strength(S=S_db): средний по частотам рост громкости,
    сдвинутый так, чтобы кадры совпадали с кадрами спектра (center=True)."""
    env = np.maximum(0.0, S_db[:, lag:] - S_db[:, :-lag]).mean(axis=0)
    env = np.pad(env, (lag + n_fft // (2 * hop), 0))
    return env[:S_db.shape[-1]]


def _next_fast_len(n):
    """Наименьшее число >= n вида 2^a · 3^b · 5^c — удобная длина для БПФ
    (scipy.fft.next_fast_len(n, real=True))."""
    best = 1 << max(0, n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def autocorrelate(y, max_size):
    """librosa.autocorrelate: автокорреляция через БПФ, первые max_size задержек."""
    n_pad = _next_fast_len(2 * len(y) - 1)
    spec = np.fft.rfft(y, n=n_pad)
    return np.fft.irfft(spec.real ** 2 + spec.imag ** 2, n=n_pad)[:min(max_size, len(y))]


def rms(y, frame_length, hop):
    """librosa.feature.rms(y, center=False). Если звук короче кадра — пустой массив
    (librosa тут падала с ошибкой)."""
    if len(y) < frame_length:
        return np.zeros(0, dtype=np.float32)
    x = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop]
    return np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1))


# --- CQT librosa (vqt с gamma=0, окно Ханна, norm=1, sparsity=0.01, без подстройки строя) ---
_HANN_BANDWIDTH = 1.50018310546875   # librosa.filters.WINDOW_BANDWIDTHS['hann']


def _relative_bandwidth(freqs):
    bpo = np.empty_like(freqs)
    logf = np.log2(freqs)
    bpo[0] = 1 / (logf[1] - logf[0])
    bpo[-1] = 1 / (logf[-1] - logf[-2])
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])
    return (2.0 ** (2 / bpo) - 1) / (2.0 ** (2 / bpo) + 1)


def _wavelet_lengths(freqs, sr, alpha):
    """Длины фильтров в отсчётах и верхняя граничная частота всего набора."""
    Q = 1.0 / alpha
    cutoff = max(freqs * (1 + 0.5 * _HANN_BANDWIDTH / Q))
    return Q * sr / freqs, cutoff


def _filter_fft(sr, freqs, alpha):
    """Фильтры одной октавы в частотной области (librosa __vqt_filter_fft)."""
    lengths, _ = _wavelet_lengths(freqs, sr, alpha)
    filters = []
    for ilen, freq in zip(lengths, freqs):
        t = np.arange(-ilen // 2, ilen // 2, dtype=float) * 2 * np.pi * freq / sr
        sig = (np.cos(t) + 1j * np.sin(t)) * hann(len(t))
        filters.append(sig / np.sum(np.abs(sig)))
    n_fft = int(2.0 ** np.ceil(np.log2(max(lengths))))
    basis = np.zeros((len(filters), n_fft), dtype=np.complex64)
    for i, f in enumerate(filters):
        lpad = (n_fft - len(f)) // 2   # по центру, как util.pad_center
        basis[i, lpad:lpad + len(f)] = f
    basis *= lengths[:, np.newaxis] / float(n_fft)
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, :n_fft // 2 + 1]
    # util.sparsify_rows(quantile=0.01): обнуляем самые слабые коэффициенты,
    # в сумме дающие меньше 1% модуля строки. librosa хранит разреженную
    # матрицу, здесь — обычная с нулями: результат тот же.
    mags = np.abs(fft_basis)
    mag_sort = np.sort(mags, axis=1)
    cumulative = np.cumsum(mag_sort / np.sum(mags, axis=1, keepdims=True), axis=1)
    threshold = mag_sort[np.arange(len(mags)), np.argmin(cumulative < 0.01, axis=1)]
    return np.where(mags >= threshold[:, np.newaxis], fft_basis, 0).astype(np.complex64), n_fft


def _num_two_factors(x):
    n = 0
    while x > 0 and x % 2 == 0:
        n += 1
        x //= 2
    return n


@lru_cache(maxsize=4)
def _cqt_plan(sr, hop, fmin, n_bins, bpo):
    """Всё, что не зависит от звука: частоты, фильтры по октавам, ранний сдвиг частоты."""
    n_octaves = int(np.ceil(float(n_bins) / bpo))
    n_filters = min(bpo, n_bins)
    ratios = 2.0 ** (np.arange(0, bpo, dtype=float) / bpo)
    freqs = np.sort(np.multiply.outer(2.0 ** np.arange(n_octaves), ratios).flatten()[:n_bins]) * fmin
    alpha = _relative_bandwidth(freqs)
    lengths, cutoff = _wavelet_lengths(freqs, sr, alpha)
    # Раннее понижение частоты: если верхний фильтр намного ниже Найквиста.
    count1 = max(0, int(np.ceil(np.log2(sr / 2.0 / cutoff)) - 1) - 1)
    count2 = max(0, _num_two_factors(hop) - n_octaves + 1)
    early = 2 ** min(count1, count2)
    sr, hop = sr / early, hop // early   # дальше librosa считает от новой частоты
    octaves = []
    my_sr, my_hop = sr, hop
    for i in range(n_octaves):
        sl = slice(-n_filters, None) if i == 0 else slice(-n_filters * (i + 1), -n_filters * i)
        fft_basis, n_fft = _filter_fft(my_sr, freqs[sl], alpha[sl])
        fft_basis *= np.sqrt(sr / my_sr)
        # После октавы звук прореживается вдвое, пока шаг делится на 2; дальше
        # остальные октавы считаются на той же частоте (у нас шаг 8820 = 4 · 2205).
        down = my_hop % 2 == 0 and i + 1 < n_octaves
        octaves.append((fft_basis, n_fft, my_hop, down))
        if down:
            my_hop //= 2
            my_sr /= 2.0
    lengths, _ = _wavelet_lengths(freqs, sr, alpha)
    return early, octaves, np.sqrt(lengths)[:, np.newaxis]


def _trim_stack(resp, n_bins):
    """Склеивает октавы снизу вверх, обрезая по самой короткой (librosa __trim_stack)."""
    max_col = min(r.shape[-1] for r in resp)
    out = np.empty((n_bins, max_col), dtype=np.complex64)
    end = n_bins
    for r in resp:
        n_oct = r.shape[0]
        if end < n_oct:
            out[:end] = r[-end:, :max_col]
        else:
            out[end - n_oct:end] = r[:, :max_col]
        end -= n_oct
    return out


def cqt(y, sr, hop, fmin, n_bins, bpo):
    """librosa.cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave) с остальными
    параметрами по умолчанию: (n_bins, кадры), complex64."""
    early, octaves, norm = _cqt_plan(sr, hop, float(fmin), n_bins, bpo)
    if early > 1:
        y = resample(y, early, 1, scale=True)
    resp = []
    for fft_basis, n_fft, my_hop, down in octaves:
        resp.append(fft_basis @ stft(y, n_fft, my_hop, np.ones(n_fft)))
        if down:
            y = resample(y, 2, 1, scale=True)
    V = _trim_stack(resp, n_bins)
    V /= norm   # на месте, как в librosa: остаётся complex64
    return V
