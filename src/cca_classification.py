import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__))
from preprocessing import extract_flicker_epoch

class CCAClassifier:
    """
    Canonical Correlation Analysis (CCA) Classifier for SSVEP BCI.
    
    Intuition:
    For each candidate target frequency f_k:
      1. Generates synthetic reference sine/cosine signals Y_k at f_k and its harmonics (2f_k, 3f_k...).
      2. Finds spatial filter weight vectors w_x and w_y that maximize Pearson correlation
         between multi-channel occipital EEG X and reference matrix Y_k:
         rho_k = max corr(w_x^T X, w_y^T Y_k)
      3. Predicts target candidate with highest canonical correlation rho_k.
    """
    def __init__(self, candidate_freqs, fs=250.0, n_harmonics=3, duration_sec=4.0):
        self.candidate_freqs = np.array(candidate_freqs)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.duration_sec = duration_sec

    def _generate_reference_signals(self, target_f, n_samples):
        """Generates reference sine and cosine matrix for target_f and its harmonics."""
        t = np.arange(n_samples) / self.fs
        ref_list = []
        for h in range(1, self.n_harmonics + 1):
            ref_list.append(np.sin(2.0 * np.pi * h * target_f * t))
            ref_list.append(np.cos(2.0 * np.pi * h * target_f * t))
        return np.array(ref_list).T  # Shape: (n_samples, 2 * n_harmonics)

    def _fast_cca(self, X, Y):
        """Computes maximum canonical correlation using QR decomposition & SVD."""
        X_c = X - np.mean(X, axis=0)
        Y_c = Y - np.mean(Y, axis=0)
        
        Q_x, _ = np.linalg.qr(X_c)
        Q_y, _ = np.linalg.qr(Y_c)
        
        M = Q_x.T @ Q_y
        _, s, _ = np.linalg.svd(M)
        return s[0]  # Largest canonical correlation coefficient

    def predict_trial(self, trial_data, channel_indices=None):
        """
        Input:
            trial_data: ndarray of shape (n_channels, n_samples)
            channel_indices: list of occipital channel indices
        Returns:
            pred_idx: index of predicted target candidate
            pred_freq: predicted frequency (Hz)
            rhos: array of canonical correlations for all candidates
        """
        if channel_indices is not None:
            eeg = trial_data[channel_indices, :]
        else:
            eeg = trial_data

        # 1. Preprocess: Extract visual latency-adjusted epoch (0.64s to 0.64s + duration_sec)
        epoch = extract_flicker_epoch(eeg, start_sec=0.64, duration_sec=self.duration_sec, fs=self.fs)
        
        # Transpose so rows are time samples and columns are EEG channels
        X = epoch.T  # Shape: (n_samples, n_selected_channels)
        n_samples = X.shape[0]

        # 2. Compute canonical correlation for each candidate frequency
        rhos = np.zeros(len(self.candidate_freqs))
        for i, cand_f in enumerate(self.candidate_freqs):
            Y = self._generate_reference_signals(cand_f, n_samples)
            rhos[i] = self._fast_cca(X, Y)

        best_idx = np.argmax(rhos)
        return best_idx, self.candidate_freqs[best_idx], rhos
