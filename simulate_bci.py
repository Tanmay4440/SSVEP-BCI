import os, sys, time
import numpy as np

sys.path.append('src')
from load_data import load_subject_data, get_occipital_channels
from cca_classification import CCAClassifier

def run_simulation(n_trials_to_sim=8):
    print("=" * 75)
    print("  SIMULATED REAL-TIME SSVEP BCI SPELLER (PRERECORDED S1 EEG)")
    print("=" * 75)
    print("  [Offline BCI target selection streaming trials from Subject 1]")
    print("=" * 75)
    
    # Load Subject 1 Data & true targets
    data, meta = load_subject_data()
    fs = meta['sampling_freq']
    target_freqs = meta['target_freqs']
    
    # 4-Target Speller Subset: 8.0 Hz, 10.0 Hz, 12.0 Hz, 14.0 Hz
    target_indices = [0, 2, 4, 6]  # True indices in Freq_Phase.mat
    subset_freqs = target_freqs[target_indices]
    target_labels = ['[ TARGET A: 8.0 Hz ]', '[ TARGET B: 10.0 Hz ]', '[ TARGET C: 12.0 Hz ]', '[ TARGET D: 14.0 Hz ]']
    
    # Setup CCA Classifier with 2.0s window
    occ_indices, _ = get_occipital_channels()
    cca_clf = CCAClassifier(candidate_freqs=subset_freqs, fs=fs, n_harmonics=3, duration_sec=2.0)
    
    print("\n  ACTIVE TARGET BOXES ON VIRTUAL DISPLAY:")
    print("  +---------------------+---------------------+---------------------+---------------------+")
    print("  |  [A] 8.0 Hz FLICKER |  [B] 10.0 Hz FLICKER|  [C] 12.0 Hz FLICKER|  [D] 14.0 Hz FLICKER|")
    print("  +---------------------+---------------------+---------------------+---------------------+")
    print("\n  Starting simulated BCI selection stream (2.0s decision window)...\n")
    
    correct_count = 0
    rng = np.random.default_rng(42)
    sim_indices = rng.choice(len(target_indices) * 6, size=n_trials_to_sim, replace=False)
    
    for step, sim_idx in enumerate(sim_indices, 1):
        target_sub_idx = sim_idx // 6
        block_idx = sim_idx % 6
        orig_t_idx = target_indices[target_sub_idx]
        true_freq = subset_freqs[target_sub_idx]
        true_label = target_labels[target_sub_idx]
        
        trial_eeg = data[:, :, orig_t_idx, block_idx]
        
        print(f"  --- Trial {step}/{n_trials_to_sim} ---")
        print(f"  Subject Gaze Directed At: {true_label} (Stimulus: {true_freq:.1f} Hz)")
        print("  Buffering 2.0s occipital EEG...", end="", flush=True)
        time.sleep(0.3)
        
        # Run CCA Classifier
        pred_sub_idx, pred_freq, rhos = cca_clf.predict_trial(trial_eeg, channel_indices=occ_indices)
        pred_label = target_labels[pred_sub_idx]
        
        is_correct = (pred_sub_idx == target_sub_idx)
        if is_correct:
            correct_count += 1
            status = "SUCCESS [CORRECT]"
        else:
            status = "MISCLASSIFIED [ERROR]"
            
        print(f" Decoded!")
        print(f"  Max Canonical Correlation: rho = {np.max(rhos):.3f}")
        print(f"  BCI Decoded Decision:      {pred_label} --> {status}\n")
        time.sleep(0.2)
        
    acc = (correct_count / n_trials_to_sim) * 100.0
    print("=" * 75)
    print(f"  SIMULATION SUMMARY: {correct_count}/{n_trials_to_sim} Correct ({acc:.1f}% Accuracy)")
    print("=" * 75)

if __name__ == '__main__':
    run_simulation(n_trials_to_sim=8)
