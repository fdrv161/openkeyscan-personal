# HCQM features for DeepRhythm 0.7 (AGPL-3.0, see __init__.py).
# The logic follows the original exactly: the trained weights depend on it.
# The original computes on torch through nnAudio; this is the same in numpy,
# with the STFT and CQT kernel formulas ported from nnAudio 0.3.3 (MIT).
import numpy as np

from okscan.dsp import hann

N_BINS = 240
N_BANDS = 8
SR = 22050
CLIP_SECONDS = 8
CLIP_SAMPLES = SR * CLIP_SECONDS
N_FFT = 2048
HOP = 512


def make_kernels():
    """Prepares the band filter and 6 CQTs (one per harmonic). Done once."""
    band_filter = _create_log_filter(N_FFT // 2 + 1, N_BANDS)
    cqt_specs = []
    for h in [1/2, 1, 2, 3, 4, 5]:
        # Frequencies here are tempo in hertz (BPM / 60): the CQT runs over the onset envelope.
        fmin = (32.7 * h) / 60
        sr_cqt = CLIP_SAMPLES // (HOP * 8)
        fmax = sr_cqt / 2
        bins_per_octave = N_BINS / np.log2(fmax / fmin)
        cqt_specs.append(_cqt_kernels(sr_cqt, fmin, N_BINS, bins_per_octave))
    return band_filter, cqt_specs


def compute_hcqm(clips, band_filter, cqt_specs):
    """clips: (N, CLIP_SAMPLES) -> (N, 6, N_BINS, N_BANDS), ready for the network."""
    n = len(clips)
    stft = np.stack([_stft_magnitude(c) for c in clips])          # (N, 1025, frames)
    bands = np.matmul(stft.transpose(0, 2, 1), band_filter.T).transpose(0, 2, 1)
    osf = _onset_strength(bands.reshape(n * N_BANDS, bands.shape[2]))
    hcqm = np.stack([_cqt_mean(osf, *spec) for spec in cqt_specs], axis=-1)
    # As in the original: reshape without swapping axes (bands and bins get mixed),
    # the network was trained on exactly this order.
    hcqm = hcqm.reshape(n, N_BINS, N_BANDS, 6)
    return np.ascontiguousarray(hcqm.transpose(0, 3, 1, 2))


def split_clips(audio_22k, max_clips):
    """Cuts audio into 8-second clips without overlap (the tail is dropped).
    With more than max_clips clips, takes an even sample: for long mixes, so as
    not to eat gigabytes of memory. Returns None if the track is shorter than 8 seconds."""
    n = len(audio_22k) // CLIP_SAMPLES
    if n == 0:
        return None
    idx = range(n) if n <= max_clips else np.linspace(0, n - 1, max_clips).round().astype(int)
    clips = [audio_22k[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES] for i in idx]
    return np.stack(clips).astype(np.float32)


def _stft_magnitude(x):
    """nnAudio STFT(n_fft=2048, hop=512, Hann window, center=True, pad_mode='reflect'),
    output_format='Magnitude': (1025, frames), float32."""
    x = np.pad(x, N_FFT // 2, mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    return np.abs(np.fft.rfft(hann(N_FFT) * frames, axis=-1)).T.astype(np.float32)


def _cqt_kernels(sr, fmin, n_bins, bins_per_octave):
    """nnAudio create_cqt_kernels (norm=1, Hann window, gamma=0) and filter lengths."""
    Q = 1.0 / (2 ** (1 / bins_per_octave) - 1)
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    lengths = np.ceil(Q * sr / freqs)
    fft_len = int(2 ** np.ceil(np.log2(int(max(lengths)))))
    kernels = np.zeros((n_bins, fft_len), dtype=np.complex64)
    for k in range(n_bins):
        l = lengths[k]
        # An odd-length filter is shifted one sample left, as in nnAudio.
        start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - (1 if l % 2 == 1 else 0)
        sig = hann(int(l)) * np.exp(np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freqs[k] / sr) / l
        kernels[k, start:start + int(l)] = sig / np.sum(np.abs(sig))
    return kernels, np.sqrt(lengths.astype(np.float32))


def _cqt_mean(x, kernels, scale):
    """nnAudio CQT1992v2 (pad_mode='constant', hop CLIP_SAMPLES // HOP), magnitude
    averaged over frames. x: (M, T) -> (M, N_BINS).

    The original convolves a kernel up to 16384 long with a 345-sample signal padded
    with zeros. Zeros add nothing to the sum, so for each frame we take only the
    part of the kernel that falls on the signal itself."""
    width = kernels.shape[1]
    hop = CLIP_SAMPLES // HOP
    n, pad = x.shape[1], width // 2
    n_frames = (n + 2 * pad - width) // hop + 1
    total = np.zeros((x.shape[0], kernels.shape[0]), dtype=np.float32)
    x64 = x.astype(np.float64)
    for f in range(n_frames):
        start = f * hop - pad                  # start of the kernel window, in signal samples
        i0, i1 = max(0, start), min(n, start + width)
        k = kernels[:, i0 - start:i1 - start]
        re = (x64[:, i0:i1] @ k.real.T.astype(np.float64)).astype(np.float32) * scale
        im = (x64[:, i0:i1] @ k.imag.T.astype(np.float64)).astype(np.float32) * scale
        total += np.sqrt(re ** 2 + im ** 2)
    return total / np.float32(n_frames)


def _create_log_filter(num_bins, num_bands):
    log_bins = np.logspace(0, np.log10(num_bins), num=num_bands + 1) - 1
    log_bins = np.unique(np.round(log_bins).astype(int))
    m = np.zeros((num_bands, num_bins), dtype=np.float32)
    for i in range(num_bands):
        start = log_bins[i]
        end = log_bins[i + 1] if i < num_bands - 1 else num_bins
        m[i, start:end] = 1 / (end - start)
    return m


def amplitude_to_db(x, top_db=80.0):
    """Replacement for torchaudio.transforms.AmplitudeToDB(top_db=80) with the same result.

    A torchaudio subtlety: for input of shape (N, 1, T) the top_db threshold is taken
    from ONE common maximum over the whole batch, not per clip.
    We repeat this, otherwise the features would differ from those the network learned on.
    """
    x_db = 10.0 * np.log10(np.maximum(x, np.float32(1e-10)))
    return np.maximum(x_db, x_db.max() - np.float32(top_db))


def _onset_strength(y, n_fft=N_FFT, hop_length=HOP, lag=1):
    s = amplitude_to_db(y)
    env = np.maximum(s[:, lag:] - s[:, :-lag], 0.0)
    pad = lag + n_fft // (2 * hop_length)
    env = np.pad(env, ((0, 0), (pad, 0)))
    return env[:, :s.shape[-1]]
