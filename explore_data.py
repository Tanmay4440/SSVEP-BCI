import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from load_data import load_subject_data, get_channel_index, get_occipital_channels

def explore_and_plot():
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Subject 1 data
    data, meta = load_subject_data()
    
    # 2. Extract metadata
    fs = meta['sampling_freq']
    n_samples = meta['n_samples']
    target_freqs = meta['target_freqs']
    time_vector = np.arange(n_samples) / fs  # 0.0 to 6.0 seconds
    
    # 3. Select channel Oz (Visual Cortex)
    oz_idx = get_channel_index('OZ')
    
    # 4. Pick Target 0 (8.0 Hz) and Block 0 (first trial)
    target_idx = 0
    block_idx = 0
    stim_freq = target_freqs[target_idx]
    
    # Signal shape: (1500,)
    signal_oz = data[oz_idx, :, target_idx, block_idx]
    
    # 5. Plot time-domain signal
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_vector, signal_oz, color='#1f77b4', lw=1.2, label=f'Electrode Oz (Target: {stim_freq:.1f} Hz)')
    
    # Highlight stimulation window (0.5s to 5.5s)
    ax.axvspan(0.5, 5.5, color='yellow', alpha=0.2, label='Flicker Stimulation Window (0.5s - 5.5s)')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7)
    ax.axvline(5.5, color='red', linestyle='--', alpha=0.7)
    
    ax.set_title(f'Stage 2: Raw Time-Domain EEG Signal (Subject 1, Channel Oz, Target 0: {stim_freq:.1f} Hz)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (uV)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    output_img = os.path.join('results', 'stage2_raw_eeg_waveform.png')
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"[SUCCESS] Saved raw EEG plot to '{output_img}'.")
    print(f"Trial Duration: {time_vector[-1]:.2f}s | Samples: {len(signal_oz)}")
    print(f"Signal Mean: {np.mean(signal_oz):.2f} uV | Std Dev: {np.std(signal_oz):.2f} uV")

if __name__ == '__main__':
    explore_and_plot()
