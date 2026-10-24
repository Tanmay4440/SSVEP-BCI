import numpy as np
from scipy.signal import welch

def compute_fft(signal, fs=250.0):
    """
    Computes Real Fast Fourier Transform (rfft) of a 1D EEG signal.
    Returns:
        freqs: Frequency bins (Hz)
        fft_magnitude: Magnitude spectrum (uV)
    """
    n_samples = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)
    fft_magnitude = np.abs(fft_vals) / n_samples
    return freqs, fft_magnitude

def compute_psd(signal, fs=250.0, nperseg=None):
    """
    Computes Power Spectral Density (PSD) using Welch's method.
    Returns:
        freqs: Frequency bins (Hz)
        psd: Power spectral density values (uV^2/Hz)
    """
    if nperseg is None:
        nperseg = min(len(signal), int(fs * 2.0))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def compute_snr(freqs, psd, target_freq, neighbor_bw=1.0):
    """
    Computes Signal-to-Noise Ratio (SNR) for a given target frequency.
    SNR = Power at target_freq / Average Power of neighboring frequency bins
    """
    target_idx = np.argmin(np.abs(freqs - target_freq))
    target_power = psd[target_idx]
    
    half_bw = neighbor_bw / 2.0
    neighbor_mask = (freqs >= (target_freq - half_bw)) & (freqs <= (target_freq + half_bw))
    neighbor_mask[target_idx] = False
    
    noise_power = np.mean(psd[neighbor_mask])
    if noise_power == 0:
        return 0.0
    return target_power / noise_power

def compute_itr(n_targets, accuracy, time_sec):
    """
    Computes Information Transfer Rate (ITR) in bits/minute using Wolpaw's formula.
    Args:
        n_targets: Number of stimulus targets (classes)
        accuracy: Classification accuracy as a fraction (0.0 to 1.0)
        time_sec: Decision time per selection in seconds
    Returns:
        ITR in bits/minute
    """
    if accuracy <= (1.0 / n_targets) or accuracy <= 0.0:
        return 0.0
    if accuracy >= 1.0:
        bits_per_trial = np.log2(n_targets)
    else:
        bits_per_trial = (
            np.log2(n_targets) +
            accuracy * np.log2(accuracy) +
            (1.0 - accuracy) * np.log2((1.0 - accuracy) / (n_targets - 1))
        )
    return bits_per_trial * (60.0 / time_sec)
