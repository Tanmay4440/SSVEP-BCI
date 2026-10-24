import os
import scipy.io as sio
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
S1_PATH = os.path.join(DATA_DIR, 'S1.mat')
FREQ_PATH = os.path.join(DATA_DIR, 'Freq_Phase.mat')

def verify_s1():
    print("=" * 60)
    print("  VERIFYING SUBJECT 1 (S1.mat) DATASET LOAD")
    print("=" * 60)

    if not os.path.exists(S1_PATH):
        print(f"[ERROR] '{S1_PATH}' not found!")
        return False

    # Load S1.mat using scipy.io
    mat_contents = sio.loadmat(S1_PATH)
    print(f"\n1. Keys found inside S1.mat: {list(mat_contents.keys())}")

    data = mat_contents['data']
    print(f"\n2. Data Matrix Shape: {data.shape}")
    print("   Dimensions breakdown:")
    print(f"   - Channels (electrodes): {data.shape[0]}")
    print(f"   - Time Points per trial: {data.shape[1]} (at 250 Hz = {data.shape[1]/250:.1f} seconds)")
    print(f"   - Stimulus Targets:      {data.shape[2]}")
    print(f"   - Experimental Blocks:   {data.shape[3]}")
    
    print(f"\n3. Data Type: {data.dtype}")
    print(f"   Min value: {np.nanmin(data):.2f} uV | Max value: {np.nanmax(data):.2f} uV")

    # Load Freq_Phase.mat if available
    if os.path.exists(FREQ_PATH):
        freq_mat = sio.loadmat(FREQ_PATH)
        freqs = freq_mat['freqs'][0]
        print(f"\n4. Target Frequencies ({len(freqs)} targets total):")
        print(f"   Range: {freqs[0]} Hz to {freqs[-1]} Hz (Interval: {freqs[1]-freqs[0]:.1f} Hz)")
        print(f"   First 5 target frequencies: {freqs[:5]} Hz")

    print("\n[SUCCESS] S1.mat loaded and verified successfully!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    verify_s1()
