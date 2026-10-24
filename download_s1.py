import os
import urllib.request

# Directory for dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Direct URLs from public Zenodo mirror (Wang et al. 2016 SSVEP Benchmark dataset)
FILES = {
    "S1.mat": "https://zenodo.org/records/14865172/files/S1.mat",
    "Freq_Phase.mat": "https://zenodo.org/records/14865172/files/Freq_Phase.mat"
}

def download_file(filename, url):
    target_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(target_path):
        print(f"[EXISTS] '{filename}' is already present in {DATA_DIR}.")
        return target_path

    print(f"[DOWNLOADING] '{filename}' from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[SUCCESS] Downloaded '{filename}' ({size_mb:.2f} MB) to {DATA_DIR}.")
    except Exception as e:
        print(f"[ERROR] Failed to download '{filename}': {e}")
        print("Alternative: You can manually download S1.mat from the official Tsinghua repository or Zenodo and place it in the 'data/' folder.")

if __name__ == '__main__':
    for fname, furl in FILES.items():
        download_file(fname, furl)
