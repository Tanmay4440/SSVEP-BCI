import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')
from load_data import load_subject_data, get_occipital_channels
from classification import FrequencyPeakClassifier

def evaluate():
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Subject 1 data
    data, meta = load_subject_data()
    fs = meta['sampling_freq']
    target_freqs = meta['target_freqs']
    n_targets = meta['n_targets']
    n_blocks = meta['n_blocks']
    
    occ_indices, occ_names = get_occipital_channels()
    
    print("\n" + "=" * 65)
    print("  STAGE 4: FREQUENCY PEAK DETECTOR CLASSIFICATION (SUBJECT 1)")
    print("=" * 65)
    
    # --- EVALUATION 1: Full 40 Targets (0.2 Hz spacing) ---
    clf_40 = FrequencyPeakClassifier(candidate_freqs=target_freqs, fs=fs, include_harmonics=True)
    y_true_40, y_pred_40 = [], []
    
    for t in range(n_targets):
        for b in range(n_blocks):
            trial = data[:, :, t, b]
            pred_idx, _, _ = clf_40.predict_trial(trial, channel_indices=occ_indices)
            y_true_40.append(t)
            y_pred_40.append(pred_idx)
            
    acc_40 = (np.sum(np.array(y_true_40) == np.array(y_pred_40)) / len(y_true_40)) * 100.0
    print(f"1. Full 40-Target Evaluation (0.2 Hz spacing):")
    print(f"   Accuracy: {acc_40:.2f}% (Chance level: {100.0/40:.2f}%)")
    
    # --- EVALUATION 2: 4-Target Subset (8.0, 10.0, 12.0, 14.0 Hz; 2.0 Hz spacing) ---
    subset_indices = [0, 10, 20, 30]  # 8.0 Hz, 10.0 Hz, 12.0 Hz, 14.0 Hz
    subset_freqs = target_freqs[subset_indices]
    clf_4 = FrequencyPeakClassifier(candidate_freqs=subset_freqs, fs=fs, include_harmonics=True)
    
    y_true_4, y_pred_4 = [], []
    for sub_t_idx, orig_t_idx in enumerate(subset_indices):
        for b in range(n_blocks):
            trial = data[:, :, orig_t_idx, b]
            pred_idx, _, _ = clf_4.predict_trial(trial, channel_indices=occ_indices)
            y_true_4.append(sub_t_idx)
            y_pred_4.append(pred_idx)
            
    acc_4 = (np.sum(np.array(y_true_4) == np.array(y_pred_4)) / len(y_true_4)) * 100.0
    print(f"\n2. 4-Target Subset Evaluation (2.0 Hz spacing: 8, 10, 12, 14 Hz):")
    print(f"   Accuracy: {acc_4:.2f}% (Chance level: 25.00%)")
    print("=" * 65)
    
    # Plot 4-Target Confusion Matrix
    conf_mat_4 = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true_4, y_pred_4):
        conf_mat_4[t, p] += 1
        
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat_4, cmap='Blues')
    plt.colorbar(im, ax=ax, label='Trial Count')
    labels = [f'{f:.1f}Hz' for f in subset_freqs]
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    ax.set_title(f'4-Target Peak Detector Confusion Matrix\n(Subject 1, Accuracy: {acc_4:.1f}%)', fontweight='bold')
    ax.set_xlabel('Predicted Target')
    ax.set_ylabel('True Target')
    
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(conf_mat_4[i, j]), ha='center', va='center',
                    color='white' if conf_mat_4[i, j] > 2 else 'black', fontweight='bold')
            
    output_img = os.path.join('results', 'stage4_peak_detector_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Saved 4-Target Confusion Matrix plot to '{output_img}'.")
    return acc_40, acc_4

if __name__ == '__main__':
    evaluate()
