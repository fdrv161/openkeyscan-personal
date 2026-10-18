# Признаки HCQM для DeepRhythm 0.7 (AGPL-3.0, см. __init__.py).
# Логика повторяет оригинал один в один: от неё зависят выученные веса.
import nnAudio.features as feat
import numpy as np
import torch
import torch.nn.functional as F

N_BINS = 240
N_BANDS = 8
SR = 22050
CLIP_SECONDS = 8
CLIP_SAMPLES = SR * CLIP_SECONDS


def make_kernels(device='cpu'):
    """Готовит STFT, полосовой фильтр и 6 CQT (по одному на гармонику). Делается один раз."""
    n_fft = 2048
    hop = 512
    n_fft_bins = n_fft // 2 + 1
    band_filter = _create_log_filter(n_fft_bins, N_BANDS, device)
    stft_spec = feat.stft.STFT(sr=SR, n_fft=n_fft, hop_length=hop,
                               output_format='Magnitude', verbose=False).to(device)
    cqt_specs = []
    for h in [1/2, 1, 2, 3, 4, 5]:
        # Частоты здесь — это темп в герцах (BPM / 60), CQT строится по огибающей атак.
        fmin = (32.7 * h) / 60
        sr_cqt = CLIP_SAMPLES // (hop * 8)
        fmax = sr_cqt / 2
        bins_per_octave = N_BINS / np.log2(fmax / fmin)
        cqt = feat.cqt.CQT(sr=sr_cqt, hop_length=CLIP_SAMPLES // hop, n_bins=N_BINS,
                           bins_per_octave=bins_per_octave, fmin=fmin,
                           output_format='Magnitude', verbose=False,
                           pad_mode='constant').to(device)
        cqt_specs.append(cqt)
    return stft_spec, band_filter, cqt_specs


def compute_hcqm(clips, stft_spec, band_filter, cqt_specs):
    """clips: (N, CLIP_SAMPLES) -> (N, 6, N_BINS, N_BANDS), готово для сети."""
    stft = stft_spec(clips)
    bands = torch.matmul(stft.transpose(1, 2), band_filter.T).transpose(1, 2)
    bands_flat = bands.reshape(stft.size(0) * N_BANDS, bands.size(2))
    osf = _onset_strength(bands_flat)
    hcqm = torch.zeros((stft.size(0) * N_BANDS, N_BINS, 6))
    for h, spec in enumerate(cqt_specs):
        hcqm[:, :, h] = spec(osf).mean(-1)
    hcqm = hcqm.reshape(stft.size(0), N_BINS, N_BANDS, 6)
    return hcqm.permute(0, 3, 1, 2)


def split_clips(audio_22k, max_clips):
    """Режет звук на 8-секундные куски без перекрытия (хвост отбрасывается).
    Если кусков больше max_clips, берёт равномерную выборку — для длинных миксов,
    чтобы не съесть гигабайты памяти. Возвращает None, если трек короче 8 секунд."""
    n = len(audio_22k) // CLIP_SAMPLES
    if n == 0:
        return None
    idx = range(n) if n <= max_clips else np.linspace(0, n - 1, max_clips).round().astype(int)
    clips = [audio_22k[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES] for i in idx]
    return torch.from_numpy(np.stack(clips).astype(np.float32))


def _create_log_filter(num_bins, num_bands, device):
    log_bins = np.logspace(0, np.log10(num_bins), num=num_bands + 1) - 1
    log_bins = np.unique(np.round(log_bins).astype(int))
    m = torch.zeros(num_bands, num_bins, device=device)
    for i in range(num_bands):
        start = log_bins[i]
        end = log_bins[i + 1] if i < num_bands - 1 else num_bins
        m[i, start:end] = 1 / (end - start)
    return m


def amplitude_to_db(x, top_db=80.0):
    """Замена torchaudio.transforms.AmplitudeToDB(top_db=80) с тем же результатом.

    Тонкость torchaudio: для входа формы (N, 1, T) порог top_db считается
    от ОДНОГО общего максимума по всему пакету, а не по каждому куску отдельно.
    Повторяем это, иначе признаки разойдутся с теми, на которых училась сеть.
    """
    x_db = 10.0 * torch.log10(torch.clamp(x, min=1e-10))
    return torch.maximum(x_db, x_db.amax() - top_db)


def _onset_strength(y, n_fft=2048, hop_length=512, lag=1):
    y = y.unsqueeze(1)
    s = amplitude_to_db(y)
    env = torch.clamp(s[..., lag:] - s[..., :-lag], min=0.0)
    env = env.mean(dim=-2)
    pad = lag + n_fft // (2 * hop_length)
    env = F.pad(env, (pad, 0), "constant", 0)
    return env[..., :s.shape[-1]]
