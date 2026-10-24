import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')
from load_data import load_subject_data, get_occipital_channels
from classification import FrequencyPeakClassifier
from cca_classification import CCAClassifier
from analysis import compute_itr

def evaluate_all():
    os.makedirs('results', exist_ok=True)
    
    # Load Subject 1 Data & true targets
    data, meta = load_subject_data()
    fs = meta['sampling_freq']
    target_freqs = meta['target_freqs']
    n_targets = meta['n_targets']
    n_blocks = meta['n_blocks']
    
    occ_indices, occ_names = get_occipital_channels()
    print(f"Using {len(occ_indices)} occipital channels: {occ_names}")
    
    # 4-Target Subset: 8.0, 10.0, 12.0, 14.0 Hz
    # In true Wang Freq_Phase.mat, these correspond to indices [0, 2, 4, 6]
    sub_indices = [0, 2, 4, 6]
    sub_freqs = target_freqs[sub_indices]
    
    print("\n" + "=" * 70)
    print("  STAGE 5: RIGOROUS BCI BENCHMARK ON SUBJECT 1 (4-TARGET & 40-TARGET)")
    print("=" * 70)
    
    # 1. Compare Peak Detector vs CCA on 4-Target Subset (4.0s window)
    peak_clf = FrequencyPeakClassifier(candidate_freqs=sub_freqs, fs=fs, duration_sec=4.0)
    cca_clf_4 = CCAClassifier(candidate_freqs=sub_freqs, fs=fs, n_harmonics=3, duration_sec=4.0)
    
    y_true_4, y_pred_peak, y_pred_cca = [], [], []
    for sub_i, orig_t in enumerate(sub_indices):
        for b in range(n_blocks):
            trial = data[:, :, orig_t, b]
            p_pk, _, _ = peak_clf.predict_trial(trial, channel_indices=occ_indices)
            p_cca, _, _ = cca_clf_4.predict_trial(trial, channel_indices=occ_indices)
            y_true_4.append(sub_i)
            y_pred_peak.append(p_pk)
            y_pred_cca.append(p_cca)
            
    y_true_4 = np.array(y_true_4)
    acc_peak_4 = np.mean(y_true_4 == np.array(y_pred_peak)) * 100.0
    acc_cca_4 = np.mean(y_true_4 == np.array(y_pred_cca)) * 100.0
    
    print(f"1. 4-Target Speller (4.0s Window, Chance = 25.0%):")
    print(f"   - FFT Peak Detector Accuracy:  {acc_peak_4:.2f}%")
    print(f"   - Canonical Correlation (CCA): {acc_cca_4:.2f}%")
    print(f"   - Performance Gain:            +{acc_cca_4 - acc_peak_4:.2f}%")
    
    # 2. Window Length Analysis on Full 40-Target Paradigm
    print(f"\n2. Full 40-Target Benchmark Across Epoch Lengths (Chance = 2.50%):")
    windows = [1.0, 2.0, 3.0, 4.0]
    acc_by_window = []
    itr_by_window = []
    
    for w in windows:
        clf_40_w = CCAClassifier(candidate_freqs=target_freqs, fs=fs, n_harmonics=3, duration_sec=w)
        correct = 0
        total = n_targets * n_blocks
        for t in range(n_targets):
            for b in range(n_blocks):
                trial = data[:, :, t, b]
                p_idx, _, _ = clf_40_w.predict_trial(trial, channel_indices=occ_indices)
                if p_idx == t:
                    correct += 1
        acc = (correct / total) * 100.0
        # Time includes 0.5s gaze-shift cue + window duration
        t_total = 0.5 + w
        itr = compute_itr(n_targets=40, accuracy=acc/100.0, time_sec=t_total)
        acc_by_window.append(acc)
        itr_by_window.append(itr)
        print(f"   - Window {w:.1f}s (Total {t_total:.1f}s): Accuracy = {acc:.2f}% | ITR = {itr:.2f} bits/min")
        
    print("=" * 70)
    
    # 3. Generate High-Quality Benchmark Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot A: 4-Target Accuracy Comparison
    methods = ['FFT Peak Detector', 'Canonical Correlation (CCA)']
    bar_colors = ['#d62728', '#2ca02c']
    bars = ax1.bar(methods, [acc_peak_4, acc_cca_4], color=bar_colors, width=0.45)
    ax1.axhline(25.0, color='gray', linestyle='--', label='Chance Level (25%)')
    ax1.set_ylim(0, 115)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('A. 4-Target Speller Accuracy Comparison (Subject 1)', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Plot B: 4-Target CCA Confusion Matrix
    conf_cca_4 = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true_4, y_pred_cca):
        conf_cca_4[t, p] += 1
    im = ax2.imshow(conf_cca_4, cmap='Greens')
    plt.colorbar(im, ax=ax2, label='Trials')
    labels = [f'{f:.0f}Hz' for f in sub_freqs]
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels)
    ax2.set_yticks(range(4)); ax2.set_yticklabels(labels)
    ax2.set_title(f'B. 4-Target CCA Confusion Matrix (Accuracy: {acc_cca_4:.1f}%)', fontweight='bold')
    ax2.set_xlabel('Predicted Target'); ax2.set_ylabel('True Target')
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, str(conf_cca_4[i, j]), ha='center', va='center',
                     color='white' if conf_cca_4[i, j] > 2 else 'black', fontweight='bold')

    # Plot C: 40-Target Accuracy vs Window Length
    ax3.plot(windows, acc_by_window, marker='o', lw=2.5, color='#1f77b4', markersize=8)
    ax3.axhline(2.5, color='gray', linestyle='--', label='40-Class Chance (2.5%)')
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('Epoch Window Length (seconds)', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('C. 40-Target CCA Accuracy vs Window Duration', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()
    for w, acc in zip(windows, acc_by_window):
        ax3.annotate(f'{acc:.1f}%', (w, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold')

    # Plot D: Information Transfer Rate (ITR) vs Window Length
    ax4.plot(windows, itr_by_window, marker='s', lw=2.5, color='#ff7f0e', markersize=8)
    ax4.set_xlabel('Epoch Window Length (seconds)', fontsize=11)
    ax4.set_ylabel('ITR (bits/minute)', fontsize=11)
    ax4.set_title('D. Information Transfer Rate (ITR) vs Duration', fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    for w, itr in zip(windows, itr_by_window):
        ax4.annotate(f'{itr:.1f}', (w, itr), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold')

    plt.tight_layout()
    output_img = os.path.join('results', 'stage5_cca_vs_peak_comparison.png')
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Generated comprehensive benchmark plot at '{output_img}'.")
    return acc_cca_4, acc_by_window, itr_by_window

if __name__ == '__main__':
    evaluate_all()
