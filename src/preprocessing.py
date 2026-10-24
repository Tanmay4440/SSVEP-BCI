import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=6.0, highcut=80.0, fs=250.0, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter to EEG data.
    Input data shape: (n_channels, n_samples) or (n_samples,)
    Note: Lower bound 6.0 Hz preserves SSVEP fundamentals (lowest is 8.0 Hz);
    Upper bound 80.0 Hz preserves 2nd and 3rd harmonics without cutting SSVEP peaks.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return filtfilt(b, a, data, axis=-1)

def extract_flicker_epoch(data, start_sec=0.64, duration_sec=4.0, fs=250.0):
    """
    Extracts the flickering stimulation epoch accounting for visual pathway latency.
    By default:
      start_sec = 0.5s (stimulus onset) + 0.14s (retina-to-V1 latency) = 0.64s
      duration_sec = length of epoch to analyze (e.g., 1.0s to 4.0s)
    """
    start_sample = int(start_sec * fs)
    end_sample = start_sample + int(duration_sec * fs)
    
    if data.ndim == 1:
        return data[start_sample:end_sample]
    elif data.ndim == 2:
        return data[:, start_sample:end_sample]
    elif data.ndim == 4:
        return data[:, start_sample:end_sample, :, :]
    else:
        raise ValueError(f"Unsupported data shape {data.shape}")
