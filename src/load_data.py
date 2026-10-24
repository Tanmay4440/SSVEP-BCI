import os
import scipy.io as sio
import numpy as np

# Official 64-channel 10-20 layout for Wang et al. (2016) SSVEP Benchmark dataset
# (Matches the sequence in '64-channels.loc' and MOABB 'Wang2016')
CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
    'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
    'O1', 'OZ', 'O2', 'CB2'
]

SAMPLING_FREQ = 250.0  # Hz
VISUAL_LATENCY = 0.14  # 140 ms visual pathway transmission delay

def load_subject_data(data_dir=None, subject_id=1):
    """
    Loads raw EEG data matrix and true target frequencies/phases for Subject 1.
    Returns:
        data: ndarray of shape (64, 1500, 40, 6)
        meta: dict containing true frequencies, phases, channel layout, and timing.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    file_path = os.path.join(data_dir, f'S{subject_id}.mat')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' does not exist.")

    mat = sio.loadmat(file_path)
    data = mat['data']  # (64, 1500, 40, 6)

    # Load official frequency and phase assignments
    freq_path = os.path.join(data_dir, 'Freq_Phase.mat')
    if os.path.exists(freq_path):
        fp = sio.loadmat(freq_path)
        target_freqs = fp['freqs'][0]    # Shape (40,)
        target_phases = fp['phases'][0]  # Shape (40,)
    else:
        # Fallback if Freq_Phase.mat not yet placed
        target_freqs = np.array([
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
            8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
            8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
            8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
        ])
        target_phases = np.zeros(40)

    meta = {
        'sampling_freq': SAMPLING_FREQ,
        'visual_latency': VISUAL_LATENCY,
        'channel_names': CHANNEL_NAMES,
        'target_freqs': target_freqs,
        'target_phases': target_phases,
        'n_channels': data.shape[0],
        'n_samples': data.shape[1],
        'n_targets': data.shape[2],
        'n_blocks': data.shape[3],
        'trial_duration': data.shape[1] / SAMPLING_FREQ
    }
    return data, meta

def get_channel_index(channel_name):
    """Returns 0-based index of a specific electrode channel name."""
    name_upper = channel_name.upper()
    if name_upper in CHANNEL_NAMES:
        return CHANNEL_NAMES.index(name_upper)
    raise ValueError(f"Channel '{channel_name}' not found in channel list.")

def get_occipital_channels():
    """
    Returns channel indices and names for posterior/occipital region (primary visual cortex).
    Key channels: PZ, PO7, PO3, POZ, PO4, PO8, O1, OZ, O2
    """
    target_names = ['PZ', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
    indices = [get_channel_index(name) for name in target_names if name in CHANNEL_NAMES]
    names = [CHANNEL_NAMES[idx] for idx in indices]
    return indices, names

if __name__ == '__main__':
    data, meta = load_subject_data()
    print("Loaded S1 data successfully!")
    print(f"Data shape: {data.shape}")
    occ_idx, occ_names = get_occipital_channels()
    print(f"Occipital channels: {occ_names} -> Indices: {occ_idx}")
    print(f"Target frequencies (first 8): {meta['target_freqs'][:8]}")
