import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from load_bids import load_bids_subject
from cca_classification import CCAClassifier
from analysis import compute_itr

# Official 40 stimulus target frequencies in Wang et al. 2016
OFFICIAL_40_FREQUENCIES = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
])

# Official 4-target speller subset (2.0 Hz spacing)
OFFICIAL_4_FREQUENCIES = np.array([8.0, 10.0, 12.0, 14.0])

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Subject & Multi-Window SSVEP BCI Benchmark")
    parser.add_argument('--subjects', type=str, default='all',
                        help="Comma-separated subject IDs (e.g. '1,2,3') or 'all' for all 34 subjects.")
    parser.add_argument('--windows', type=str, default='1.0,2.0,3.0,4.0',
                        help="Comma-separated window durations in seconds (default: '1.0,2.0,3.0,4.0').")
    parser.add_argument('--harmonics', type=int, default=3,
                        help="Number of harmonics for CCA reference signals (default: 3).")
    return parser.parse_args()

def run_multi_subject_multi_window():
    args = parse_args()
    
    # Parse subjects
    if args.subjects.lower() == 'all':
        subject_ids = list(range(1, 35))  # All 34 subjects
    else:
        subject_ids = [int(s.strip().replace('sub-', '')) for s in args.subjects.split(',') if s.strip()]
        
    # Parse windows
    windows = [float(w.strip()) for w in args.windows.split(',') if w.strip()]
    max_window = max(windows)
    fs = 250.0
    n_harmonics = args.harmonics
    
    os.makedirs('results', exist_ok=True)
    
    print("=" * 80)
    print(f"  POPULATION BENCHMARK: {len(subject_ids)} SUBJECTS x {len(windows)} WINDOWS")
    print(f"  Windows evaluated:    {windows} seconds")
    print(f"  Total evaluations:    {len(subject_ids) * len(windows)} experiments")
    print("=" * 80)
    
    # Pre-generate classifiers for each window length
    classifiers_40 = {w: CCAClassifier(OFFICIAL_40_FREQUENCIES, fs=fs, n_harmonics=n_harmonics, duration_sec=w) for w in windows}
    classifiers_4 = {w: CCAClassifier(OFFICIAL_4_FREQUENCIES, fs=fs, n_harmonics=n_harmonics, duration_sec=w) for w in windows}
    
    all_results = []
    start_total_time = time.time()
    
    for idx, sub_id in enumerate(subject_ids, 1):
        sub_name = f"sub-{sub_id}"
        t0 = time.time()
        print(f"\n[{idx:02d}/{len(subject_ids):02d}] Loading {sub_name}...", end="", flush=True)
        
        # Load the maximum window epoch once to optimize disk I/O
        epochs_max, true_labels, meta = load_bids_subject(
            sub_name, 
            channel_subset='occipital', 
            duration_sec=max_window
        )
        
        load_time = time.time() - t0
        print(f" Loaded ({load_time:.1f}s). Decoding windows: ", end="", flush=True)
        
        # 4-target mask
        mask_4 = np.isin(true_labels, OFFICIAL_4_FREQUENCIES)
        labels_4 = true_labels[mask_4]
        
        sub_window_summary = []
        
        for w in windows:
            w_samples = int(w * fs)
            total_decision_time = 0.5 + w  # 0.5s gaze-shift + window
            
            # --- 1. Evaluate 40-Target Paradigm (240 trials) ---
            correct_40 = 0
            for trial_idx in range(len(true_labels)):
                trial_data_w = epochs_max[trial_idx, :, :w_samples]
                pred_idx, pred_f, _ = classifiers_40[w].predict_trial(trial_data_w)
                if np.isclose(pred_f, true_labels[trial_idx]):
                    correct_40 += 1
                    
            acc_40 = (correct_40 / len(true_labels)) * 100.0
            itr_40 = compute_itr(n_targets=40, accuracy=acc_40 / 100.0, time_sec=total_decision_time)
            
            # --- 2. Evaluate 4-Target Paradigm (24 trials) ---
            correct_4 = 0
            epochs_4 = epochs_max[mask_4]
            for trial_idx in range(len(labels_4)):
                trial_data_w = epochs_4[trial_idx, :, :w_samples]
                pred_idx, pred_f, _ = classifiers_4[w].predict_trial(trial_data_w)
                if np.isclose(pred_f, labels_4[trial_idx]):
                    correct_4 += 1
                    
            acc_4 = (correct_4 / len(labels_4)) * 100.0
            itr_4 = compute_itr(n_targets=4, accuracy=acc_4 / 100.0, time_sec=total_decision_time)
            
            all_results.append({
                'Subject': sub_name,
                'Subject_Num': sub_id,
                'Window_Sec': w,
                'Decision_Time_Sec': total_decision_time,
                'Acc_40_Pct': acc_40,
                'ITR_40_bpm': itr_40,
                'Acc_4_Pct': acc_4,
                'ITR_4_bpm': itr_4
            })
            
            sub_window_summary.append(f"{w:.0f}s: {acc_40:.1f}%")
            
        print(" | ".join(sub_window_summary))
        
    total_elapsed = time.time() - start_total_time
    df = pd.DataFrame(all_results)
    
    # Save complete raw metrics table
    csv_raw_path = os.path.join('results', 'population_all_subjects_all_windows.csv')
    df.to_csv(csv_raw_path, index=False)
    
    # Compute population statistics by window
    summary_df = df.groupby('Window_Sec').agg({
        'Acc_40_Pct': ['mean', 'std', 'min', 'max'],
        'ITR_40_bpm': ['mean', 'std', 'min', 'max'],
        'Acc_4_Pct':  ['mean', 'std', 'min', 'max'],
        'ITR_4_bpm':  ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    csv_summary_path = os.path.join('results', 'population_summary_by_window.csv')
    summary_df.to_csv(csv_summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("  GRAND POPULATION BENCHMARK SUMMARY")
    print(f"  Total time elapsed: {total_elapsed/60.0:.2f} minutes")
    print("=" * 80)
    print("  WINDOW | 40-TARGET ACCURACY | 40-TARGET ITR (bpm) | 4-TARGET ACCURACY")
    print("  " + "-" * 72)
    for w in windows:
        sub_w = df[df['Window_Sec'] == w]
        acc_m, acc_s = sub_w['Acc_40_Pct'].mean(), sub_w['Acc_40_Pct'].std()
        itr_m, itr_s = sub_w['ITR_40_bpm'].mean(), sub_w['ITR_40_bpm'].std()
        acc4_m, acc4_s = sub_w['Acc_4_Pct'].mean(), sub_w['Acc_4_Pct'].std()
        print(f"   {w:4.1f}s | {acc_m:5.1f}% +/- {acc_s:4.1f}%  | {itr_m:5.1f} +/- {itr_s:4.1f} bpm | {acc4_m:5.1f}% +/- {acc4_s:4.1f}%")
    print("=" * 80)
    print(f"  Raw data saved:     '{csv_raw_path}'")
    print(f"  Summary data saved: '{csv_summary_path}'")
    
    # --- Generate Publication-Quality Visualizations ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. 40-Target Accuracy vs Window (Mean +/- Std Shaded Band)
    w_vals = windows
    acc40_means = [df[df['Window_Sec'] == w]['Acc_40_Pct'].mean() for w in w_vals]
    acc40_stds  = [df[df['Window_Sec'] == w]['Acc_40_Pct'].std()  for w in w_vals]
    
    ax1.plot(w_vals, acc40_means, color='#1f77b4', marker='o', lw=2.5, label='Cohort Mean Accuracy')
    ax1.fill_between(w_vals, 
                     np.array(acc40_means) - np.array(acc40_stds), 
                     np.array(acc40_means) + np.array(acc40_stds), 
                     color='#1f77b4', alpha=0.2, label='+/- 1 Std Dev')
    ax1.axhline(2.5, color='gray', linestyle='--', label='Chance Level (2.5%)')
    ax1.set_ylim(0, 105)
    ax1.set_xlabel('Window Length (seconds)', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('A. 40-Target Population Accuracy vs Window Duration', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    for w, m in zip(w_vals, acc40_means):
        ax1.annotate(f"{m:.1f}%", (w, m), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    # 2. 40-Target ITR vs Window (Speed Sweet Spot)
    itr40_means = [df[df['Window_Sec'] == w]['ITR_40_bpm'].mean() for w in w_vals]
    itr40_stds  = [df[df['Window_Sec'] == w]['ITR_40_bpm'].std()  for w in w_vals]
    
    ax2.plot(w_vals, itr40_means, color='#ff7f0e', marker='s', lw=2.5, label='Cohort Mean ITR')
    ax2.fill_between(w_vals, 
                     np.array(itr40_means) - np.array(itr40_stds), 
                     np.array(itr40_means) + np.array(itr40_stds), 
                     color='#ff7f0e', alpha=0.2, label='+/- 1 Std Dev')
    ax2.set_xlabel('Window Length (seconds)', fontsize=11)
    ax2.set_ylabel('ITR (bits/minute)', fontsize=11)
    ax2.set_title('B. Population Information Transfer Rate (ITR)', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    for w, m in zip(w_vals, itr40_means):
        ax2.annotate(f"{m:.1f} bpm", (w, m), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    # 3. 4-Target Speller Accuracy vs Window
    acc4_means = [df[df['Window_Sec'] == w]['Acc_4_Pct'].mean() for w in w_vals]
    acc4_stds  = [df[df['Window_Sec'] == w]['Acc_4_Pct'].std()  for w in w_vals]
    
    ax3.plot(w_vals, acc4_means, color='#2ca02c', marker='^', lw=2.5, label='4-Target Speller Mean')
    ax3.fill_between(w_vals, 
                     np.array(acc4_means) - np.array(acc4_stds), 
                     np.array(acc4_means) + np.array(acc4_stds), 
                     color='#2ca02c', alpha=0.2, label='+/- 1 Std Dev')
    ax3.axhline(25.0, color='gray', linestyle='--', label='Chance Level (25.0%)')
    ax3.set_ylim(40, 105)
    ax3.set_xlabel('Window Length (seconds)', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('C. 4-Target Speller Population Accuracy', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right')
    for w, m in zip(w_vals, acc4_means):
        ax3.annotate(f"{m:.1f}%", (w, m), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    # 4. Boxplot of Inter-Subject Variability (40-Target across windows)
    box_data = [df[df['Window_Sec'] == w]['Acc_40_Pct'].values for w in w_vals]
    bp = ax4.boxplot(box_data, labels=[f"{w:.0f}s" for w in w_vals], patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('#d6e4f0')
        box.set_edgecolor('#1f77b4')
    ax4.axhline(2.5, color='gray', linestyle=':', label='Chance (2.5%)')
    ax4.set_xlabel('Window Length (seconds)', fontsize=11)
    ax4.set_ylabel('40-Target Accuracy (%)', fontsize=11)
    ax4.set_title('D. Inter-Subject Accuracy Distribution (Boxplot)', fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right')

    plt.tight_layout()
    plot_path = os.path.join('results', 'population_benchmark_all_subjects.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"  Summary plots saved: '{plot_path}'\n")

if __name__ == '__main__':
    run_multi_subject_multi_window()
