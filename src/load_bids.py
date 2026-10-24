import os
import mne
import pandas as pd
import numpy as np

# Official occipital and parieto-occipital channels for Wang et al. 2016
# covering the primary visual cortex (V1) and visual association areas:
OFFICIAL_OCCIPITAL_CHANNELS = ['PZ', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']

# Visual pathway latency delay (retina-to-V1 transmission time)
VISUAL_LATENCY_SEC = 0.14  # 140 ms

def get_bids_subject_path(subject_id, bids_root=None):
    """
    Returns paths to .set EEG file and events.tsv for a given BIDS subject.
    """
    if bids_root is None:
        bids_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nm000126_v1.0.3')
        
    sub_str = f"sub-{subject_id}" if isinstance(subject_id, int) or not str(subject_id).startswith("sub-") else str(subject_id)
    eeg_dir = os.path.join(bids_root, sub_str, 'ses-0', 'eeg')
    
    set_file = os.path.join(eeg_dir, f"{sub_str}_ses-0_task-ssvep_run-0_eeg.set")
    tsv_file = os.path.join(eeg_dir, f"{sub_str}_ses-0_task-ssvep_run-0_events.tsv")
    
    if not os.path.exists(set_file):
        raise FileNotFoundError(f"EEG file not found for {sub_str}: '{set_file}'")
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"Events TSV not found for {sub_str}: '{tsv_file}'")
        
    return set_file, tsv_file

def load_bids_subject(subject_id, bids_root=None, channel_subset='occipital', duration_sec=4.0):
    """
    Loads raw continuous EEG for a subject from the official BIDS Wang2016 dataset,
    parses events.tsv, and slices trials into latency-adjusted epochs.

    Parameters:
        subject_id: int (e.g. 1) or str (e.g. 'sub-1')
        bids_root: path to data/nm000126_v1.0.3
        channel_subset: 'occipital' (9 channels) or 'all' (64 channels)
        duration_sec: duration of visual flicker window to extract (e.g., 1.0, 2.0, 4.0)

    Returns:
        epochs: ndarray of shape (n_trials, n_channels, n_samples)
        labels: 1D ndarray of true stimulus frequencies for each trial (Hz)
        meta: dict containing sampling frequency, channel names, subject id, and event table
    """
    set_path, tsv_path = get_bids_subject_path(subject_id, bids_root)
    
    # 1. Read continuous EEGLAB dataset using MNE-Python
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    fs = float(raw.info['sfreq'])  # 250.0 Hz
    
    # 2. Read events table
    events_df = pd.read_csv(tsv_path, sep='\t')
    
    # 3. Match channel indices
    ch_names_upper = [ch.upper() for ch in raw.ch_names]
    if channel_subset == 'occipital':
        selected_channels = OFFICIAL_OCCIPITAL_CHANNELS
        ch_indices = [ch_names_upper.index(ch) for ch in selected_channels if ch in ch_names_upper]
        channel_names = [raw.ch_names[idx] for idx in ch_indices]
    else:
        ch_indices = list(range(len(raw.ch_names)))
        channel_names = raw.ch_names

    # 4. Extract continuous data matrix: (n_channels, total_samples)
    continuous_eeg = raw.get_data()[ch_indices, :]
    
    # 5. Extract trial epochs
    # In BIDS Wang2016:
    #   'onset' is 0.2s before cue;
    #   stimulus flicker begins at 0.5s;
    #   visual latency is 0.14s.
    # Relative offset from trial sample: 0.5s - 0.2s + 0.14s = 0.44s
    latency_sample_offset = int((0.5 - 0.2 + VISUAL_LATENCY_SEC) * fs)
    epoch_length_samples = int(duration_sec * fs)
    
    epochs_list = []
    labels_list = []
    
    for _, row in events_df.iterrows():
        sample_start = int(row['sample']) + latency_sample_offset
        sample_end = sample_start + epoch_length_samples
        
        trial_data = continuous_eeg[:, sample_start:sample_end]
        epochs_list.append(trial_data)
        labels_list.append(float(row['trial_type']))
        
    epochs = np.array(epochs_list)  # (240, n_channels, n_samples)
    labels = np.array(labels_list)  # (240,)
    
    meta = {
        'subject_id': f"sub-{subject_id}" if isinstance(subject_id, int) else subject_id,
        'sampling_freq': fs,
        'channel_names': channel_names,
        'channel_indices': ch_indices,
        'duration_sec': duration_sec,
        'n_trials': len(labels),
        'unique_frequencies': np.unique(labels)
    }
    
    return epochs, labels, meta
