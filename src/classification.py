import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__))
from preprocessing import bandpass_filter, extract_flicker_epoch

class FrequencyPeakClassifier:
    """
    Frequency Peak / Power Spectrum Classifier for SSVEP BCI.
    
    Method:
      1. Extracts visual latency-adjusted stimulation epoch (0.64s onward).
      2. Computes RFFT across selected occipital channels.
      3. Measures spectral magnitude at each candidate target frequency (and 2nd harmonic).
      4. Predicts target candidate with highest power score.
    """
    def __init__(self, candidate_freqs, fs=250.0, include_harmonics=True, duration_sec=4.0):
        self.candidate_freqs = np.array(candidate_freqs)
        self.fs = fs
        self.include_harmonics = include_harmonics
        self.duration_sec = duration_sec

    def predict_trial(self, trial_data, channel_indices=None):
        if channel_indices is not None:
            eeg_channels = trial_data[channel_indices, :]
        else:
            eeg_channels = trial_data

        # 1. Preprocess: Extract epoch (0.64s to 0.64s + duration_sec)
        epoch = extract_flicker_epoch(eeg_channels, start_sec=0.64, duration_sec=self.duration_sec, fs=self.fs)
        filtered = bandpass_filter(epoch, lowcut=6.0, highcut=80.0, fs=self.fs)

        # 2. Compute Real FFT across time axis
        n_samples = filtered.shape[-1]
        fft_vals = np.abs(np.fft.rfft(filtered, axis=-1)) / n_samples
        freqs = np.fft.rfftfreq(n_samples, d=1.0/self.fs)
        
        # Average FFT magnitude across channels
        avg_fft = np.mean(fft_vals, axis=0)

        # 3. Evaluate power at candidate frequencies
        scores = np.zeros(len(self.candidate_freqs))
        for i, target_f in enumerate(self.candidate_freqs):
            idx_f = np.argmin(np.abs(freqs - target_f))
            score = avg_fft[idx_f]

            if self.include_harmonics and (2 * target_f <= 80.0):
                idx_2f = np.argmin(np.abs(freqs - 2 * target_f))
                score += 0.5 * avg_fft[idx_2f]

            scores[i] = score

        best_idx = np.argmax(scores)
        return best_idx, self.candidate_freqs[best_idx], scores
