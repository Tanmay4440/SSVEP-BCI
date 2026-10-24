import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')
from load_data import load_subject_data, get_channel_index
from preprocessing import bandpass_filter, extract_flicker_epoch
from analysis import compute_psd, compute_snr

def analyze_and_plot():
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Subject 1 data
    data, meta = load_subject_data()
    fs = meta['sampling_freq']
    target_freqs = meta['target_freqs']
    oz_idx = get_channel_index('OZ')
    
    # Select 2 distinct targets: Target 0 (8.0 Hz) and Target 4 (12.0 Hz)
    idx1, idx2 = 0, 4  # 8.0 Hz and 12.0 Hz in true Wang layout
    freq1, freq2 = target_freqs[idx1], target_freqs[idx2]
    
    # Extract trials for block 0
    raw1 = data[oz_idx, :, idx1, 0]
    raw2 = data[oz_idx, :, idx2, 0]
    
    # Preprocess: Epoch (0.64s to 4.64s) + Bandpass Filter (6-80 Hz)
    epoch1 = extract_flicker_epoch(raw1, start_sec=0.64, duration_sec=4.0, fs=fs)
    epoch2 = extract_flicker_epoch(raw2, start_sec=0.64, duration_sec=4.0, fs=fs)
    
    filt1 = bandpass_filter(epoch1, lowcut=6.0, highcut=80.0, fs=fs)
    filt2 = bandpass_filter(epoch2, lowcut=6.0, highcut=80.0, fs=fs)
    
    # Compute PSD using Welch's method
    freqs1, psd1 = compute_psd(filt1, fs=fs, nperseg=int(fs*2))
    freqs2, psd2 = compute_psd(filt2, fs=fs, nperseg=int(fs*2))
    
    # Compute SNR
    snr1 = compute_snr(freqs1, psd1, freq1)
    snr2 = compute_snr(freqs2, psd2, freq2)
    
    print(f"Target 1 ({freq1:.1f} Hz) - Peak PSD: {np.max(psd1):.2f} uV^2/Hz | SNR at {freq1:.1f}Hz: {snr1:.2f}")
    print(f"Target 2 ({freq2:.1f} Hz) - Peak PSD: {np.max(psd2):.2f} uV^2/Hz | SNR at {freq2:.1f}Hz: {snr2:.2f}")
    
    # Plot Spectral Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Plot 1: 8.0 Hz Target
    ax1.plot(freqs1, psd1, color='#1f77b4', lw=2.0, label=f'PSD (Target: {freq1:.1f} Hz)')
    ax1.axvline(freq1, color='red', linestyle='--', lw=1.5, label=f'Fundamental ({freq1:.1f} Hz)')
    if 2 * freq1 <= 50:
        ax1.axvline(2 * freq1, color='orange', linestyle=':', lw=1.5, label=f'2nd Harmonic ({2*freq1:.1f} Hz)')
    ax1.set_xlim(5, 35)
    ax1.set_ylabel('PSD (uV^2/Hz)', fontsize=11)
    ax1.set_title(f'Subject 1 - Channel Oz Spectral Response for Target {freq1:.1f} Hz (SNR = {snr1:.1f})', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Plot 2: 12.0 Hz Target
    ax2.plot(freqs2, psd2, color='#2ca02c', lw=2.0, label=f'PSD (Target: {freq2:.1f} Hz)')
    ax2.axvline(freq2, color='red', linestyle='--', lw=1.5, label=f'Fundamental ({freq2:.1f} Hz)')
    if 2 * freq2 <= 50:
        ax2.axvline(2 * freq2, color='orange', linestyle=':', lw=1.5, label=f'2nd Harmonic ({2*freq2:.1f} Hz)')
    ax2.set_xlim(5, 35)
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('PSD (uV^2/Hz)', fontsize=11)
    ax2.set_title(f'Subject 1 - Channel Oz Spectral Response for Target {freq2:.1f} Hz (SNR = {snr2:.1f})', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    output_img = os.path.join('results', 'stage3_psd_spectrum_comparison.png')
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"[SUCCESS] Saved spectral plot comparison to '{output_img}'.")

if __name__ == '__main__':
    analyze_and_plot()
