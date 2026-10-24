import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_all_subjects_charts():
    csv_path = os.path.join('results', 'population_all_subjects_all_windows.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' not found.")
        
    df = pd.read_csv(csv_path)
    
    # Sort subjects 1 through 34
    df['Subject_Num'] = df['Subject'].apply(lambda x: int(x.replace('sub-', '')))
    subjects = sorted(df['Subject_Num'].unique())
    windows = sorted(df['Window_Sec'].unique())
    
    # Palette for the 4 window lengths
    palette = {
        1.0: '#1f77b4',  # Steel Blue
        2.0: '#2ca02c',  # Forest Green (Optimal ITR)
        3.0: '#ff7f0e',  # Amber Orange
        4.0: '#9467bd'   # Purple (Max Accuracy)
    }
    
    # =========================================================================
    # FIGURE 1: Comprehensive Grouped Bar Chart (Accuracy & ITR for All 34 Subjects)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    
    x = np.arange(len(subjects))  # 0 to 33
    bar_width = 0.20
    
    # Plot 1: 40-Target Accuracy (%)
    for i, w in enumerate(windows):
        sub_df = df[df['Window_Sec'] == w].sort_values('Subject_Num')
        accs = sub_df['Acc_40_Pct'].values
        offset = (i - 1.5) * bar_width
        ax1.bar(x + offset, accs, width=bar_width, color=palette[w], label=f'Window {w:.0f}s', alpha=0.9, edgecolor='black', lw=0.4)
        
    ax1.axhline(2.5, color='gray', linestyle='--', lw=1.2, label='Chance Level (2.5%)')
    ax1.set_ylabel('40-Target Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. 40-Target Classification Accuracy across All 34 Subjects (4 Window Lengths)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax1.legend(loc='upper right', ncol=5, fontsize=11, framealpha=0.9)
    
    # Plot 2: Information Transfer Rate (ITR in bits/min)
    for i, w in enumerate(windows):
        sub_df = df[df['Window_Sec'] == w].sort_values('Subject_Num')
        itrs = sub_df['ITR_40_bpm'].values
        offset = (i - 1.5) * bar_width
        ax2.bar(x + offset, itrs, width=bar_width, color=palette[w], label=f'Window {w:.0f}s', alpha=0.9, edgecolor='black', lw=0.4)
        
    ax2.set_ylabel('ITR (bits/minute)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Information Transfer Rate (ITR) across All 34 Subjects (4 Window Lengths)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Subject ID (sub-1 to sub-34)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{s}" for s in subjects], fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax2.legend(loc='upper right', ncol=4, fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    bar_chart_path = os.path.join('results', 'all_34_subjects_accuracy_and_itr_chart.png')
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved grouped bar chart to '{bar_chart_path}'.")
    
    # =========================================================================
    # FIGURE 2: Publication-Style Dual Heatmap (Accuracy & ITR Matrix)
    # =========================================================================
    # Pivot tables: rows = Window length, columns = Subject
    acc_pivot = df.pivot(index='Window_Sec', columns='Subject_Num', values='Acc_40_Pct')
    itr_pivot = df.pivot(index='Window_Sec', columns='Subject_Num', values='ITR_40_bpm')
    
    fig, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    
    # Heatmap 1: Accuracy
    im1 = ax_h1.imshow(acc_pivot.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    cbar1 = plt.colorbar(im1, ax=ax_h1, pad=0.01)
    cbar1.set_label('Accuracy (%)', fontsize=11, fontweight='bold')
    ax_h1.set_yticks(range(len(windows)))
    ax_h1.set_yticklabels([f"{w:.0f}s Window" for w in windows], fontsize=11, fontweight='bold')
    ax_h1.set_title('40-Target Accuracy (%) Heatmap across 34 Subjects', fontsize=13, fontweight='bold')
    
    # Annotate values on Heatmap 1
    for r in range(len(windows)):
        for c in range(len(subjects)):
            val = acc_pivot.values[r, c]
            color = "white" if val > 65 else "black"
            ax_h1.text(c, r, f"{val:.0f}", ha='center', va='center', color=color, fontsize=8, fontweight='bold')
            
    # Heatmap 2: ITR
    im2 = ax_h2.imshow(itr_pivot.values, cmap='Oranges', aspect='auto', vmin=0, vmax=120)
    cbar2 = plt.colorbar(im2, ax=ax_h2, pad=0.01)
    cbar2.set_label('ITR (bits/min)', fontsize=11, fontweight='bold')
    ax_h2.set_yticks(range(len(windows)))
    ax_h2.set_yticklabels([f"{w:.0f}s Window" for w in windows], fontsize=11, fontweight='bold')
    ax_h2.set_title('Information Transfer Rate (ITR in bits/min) Heatmap across 34 Subjects', fontsize=13, fontweight='bold')
    
    # Annotate values on Heatmap 2
    for r in range(len(windows)):
        for c in range(len(subjects)):
            val = itr_pivot.values[r, c]
            color = "white" if val > 70 else "black"
            ax_h2.text(c, r, f"{val:.0f}", ha='center', va='center', color=color, fontsize=8, fontweight='bold')
            
    ax_h2.set_xticks(range(len(subjects)))
    ax_h2.set_xticklabels([f"S{s}" for s in subjects], fontsize=10, fontweight='bold')
    ax_h2.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = os.path.join('results', 'all_34_subjects_heatmaps.png')
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved heatmaps to '{heatmap_path}'.")

if __name__ == '__main__':
    generate_all_subjects_charts()
