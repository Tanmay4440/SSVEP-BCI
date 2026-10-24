# SSVEP Brain-Computer Interface (BCI) Pipeline

An end-to-end, reproducible offline SSVEP (Steady-State Visual Evoked Potential) Brain-Computer Interface pipeline implemented in Python. 

Evaluated across all **34 human participants (8,160 trials total)** of the **Wang et al. (2016) SSVEP Benchmark Dataset** (BIDS v1.9.0 format).

---

## 🔬 Core Engineering & Signal Processing Pipeline

1. **EEG Data Ingestion & BIDS Parsing (`src/load_bids.py`)**:
   - Parses 64-channel EEG recordings ($250\text{ Hz}$ sampling rate) using `MNE-Python`.
   - Isolates the **9 primary visual cortex (V1) electrodes** over the occipital lobe: `Pz, PO7, PO3, POz, PO4, PO8, O1, Oz, O2`.
   - Incorporates the biological **$140\text{ ms}$ visual pathway latency delay** (retina-to-V1 transmission time).
2. **Signal Preprocessing (`src/preprocessing.py`)**:
   - 4th-order zero-phase Butterworth bandpass filter ($6\text{ Hz} - 80\text{ Hz}$) that removes slow baseline drifts while preserving fundamental stimulus frequencies and upper harmonics ($2f, 3f$).
3. **Spectral Feature Extraction (`src/analysis.py`)**:
   - Welch Power Spectral Density (PSD), Fast Fourier Transform (FFT), and Signal-to-Noise Ratio (SNR).
   - **Wolpaw's Information Transfer Rate (ITR)** calculation in $\text{bits/minute}$.
4. **Decoders (`src/classification.py` & `src/cca_classification.py`)**:
   - **Baseline**: Multi-harmonic FFT Peak Detector.
   - **Production Model**: Training-free **Canonical Correlation Analysis (CCA)** via QR decomposition and Singular Value Decomposition (SVD), matching multi-channel scalp potentials against harmonic sinusoidal reference templates.
5. **Real-Time Speller Simulation (`simulate_bci.py`)**:
   - Simulates streaming EEG epochs into the decoder to select targets on a 4-choice virtual speller interface.

---

## 📊 Benchmark Results

### 1. Grand Population Benchmark (All 34 Subjects, 40-Target Paradigm)
*Evaluation across $8.0\text{ Hz} - 15.8\text{ Hz}$ with $0.2\text{ Hz}$ spacing (Chance Level: $2.50\%$)*

| Window Length | Total Decision Time (incl. $0.5\text{s}$ cue) | 40-Target Accuracy (Mean $\pm$ Std) | Population ITR (Mean $\pm$ Std) | 4-Target Speller Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| **1.0 s** | $1.5\text{ s}$ | $17.9\% \pm 10.3\%$ | $14.1 \pm 14.4\text{ bpm}$ | $65.2\% \pm 17.3\%$ |
| **2.0 s** | $2.5\text{ s}$ | $71.3\% \pm 22.5\%$ | **$75.0 \pm 33.9\text{ bpm}$** *(Peak ITR)* | $92.2\% \pm 10.5\%$ |
| **3.0 s** | $3.5\text{ s}$ | $83.6\% \pm 18.7\%$ | $68.4 \pm 22.7\text{ bpm}$ | $96.1\% \pm 6.7\%$ |
| **4.0 s** | $4.5\text{ s}$ | **$89.2\% \pm 14.1\%$** | $58.6 \pm 14.4\text{ bpm}$ | **$97.5\% \pm 6.3\%$** |

* **Optimal Communication Speed**: Reached at a **$2.0\text{s}$ window**, delivering **$75.0\text{ bits/min}$** across the population.
* **Peak Accuracy**: At $4.0\text{s}$, the cohort average reached **$89.2\%$** on 40 targets (with 4 subjects scoring $100\%$, and 12 subjects scoring $\ge 99.1\%$).
* **4-Target Speller**: Reached **$97.5\% \pm 6.3\%$** across all 34 participants (26 subjects scored a perfect $100\%$).

---

## 📁 Repository Structure

```text
SSVEP-BCI/
│
├── src/
│   ├── load_bids.py            # MNE BIDS dataset loader (34 subjects)
│   ├── load_data.py            # MATLAB S1.mat array loader
│   ├── preprocessing.py        # Zero-phase Butterworth filtering & epoching
│   ├── analysis.py             # Welch PSD, SNR, & Wolpaw ITR calculator
│   ├── classification.py       # FFT Peak Detector Classifier
│   └── cca_classification.py   # Canonical Correlation Analysis (QR + SVD)
│
├── results/                    # Benchmark metrics and figures
│   ├── all_34_subjects_accuracy_and_itr_chart.png
│   ├── all_34_subjects_heatmaps.png
│   ├── population_benchmark_all_subjects.png
│   ├── population_all_subjects_all_windows.csv
│   └── population_summary_by_window.csv
│
├── evaluate_population.py      # Automated 34-subject multi-window benchmark
├── evaluate_cca.py             # Single-subject CCA evaluation
├── simulate_bci.py             # Simulated real-time streaming BCI speller
├── download_s1.py              # Automated Subject 1 downloader
├── requirements.txt            # Python dependencies
├── .gitignore                  # Data exclusion rules
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Tanmay4440/SSVEP-BCI.git
cd SSVEP-BCI
pip install -r requirements.txt
```

### 2. Download Sample Data (Subject 1)
```bash
python download_s1.py
```

### 3. Run Real-Time BCI Simulation
```bash
python simulate_bci.py
```

### 4. Run Population Benchmark
```bash
# Evaluate across subjects (e.g. first 5 subjects or all 34)
python evaluate_population.py --subjects 1,2,3,4,5 --windows 1.0,2.0,3.0,4.0
```

---

## 🧠 Key Theoretical Concepts

### 1. What is SSVEP?
Steady-State Visual Evoked Potential: When a user fixates on a visual light flickering at constant frequency $f$, neural populations in the primary visual cortex (V1) synchronize their post-synaptic firing to match $f$ and its integer harmonics ($2f, 3f$).

### 2. Why use Canonical Correlation Analysis (CCA)?
* **Zero Calibration (Training-Free)**: Unlike deep neural networks that require large labeled training sets and overfit on small trial numbers, standard CCA computes closed-form spatial filters on the fly.
* **Multi-Channel Spatial Noise Cancellation**: CCA calculates optimal spatial weights $w_x$ across 9 occipital electrodes to maximize correlation with reference sinusoids, suppressing background scalp noise.
* **Phase Invariance**: Reference templates incorporate both sine and cosine components across harmonics, rendering the detector robust to unknown visual transmission delays.

### 3. What is Information Transfer Rate (ITR)?
Measured in $\text{bits/minute}$ via Wolpaw's equation:
$$B = \log_2(N) + P\log_2(P) + (1-P)\log_2\left(\frac{1-P}{N-1}\right), \quad \text{ITR} = B \times \frac{60}{T}$$
In a 40-choice speller, each selection carries $\log_2(40) = 5.32\text{ bits}$. At a $2.0\text{s}$ window ($T = 2.5\text{s}$ total), the population ITR peaks at **$75.0\text{ bits/min}$**.
