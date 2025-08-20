import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.io import read_raw_snirf, RawArray
from scipy.signal import welch

# Paths
DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
RESULTS_CSV = os.path.join(DATA_DIR, "features.csv")
SUBJECTS = list(range(0, 16))

# Emotion labels per subject
subject_emotions = {
    0: 'Gloomy',
    1: 'Calm',
    2: 'Satisfied',
    3: 'Satisfied',
    4: 'Amused',
    5: 'Calm',
    6: 'Amused',
    7: 'Frustrated',
    8: 'Gloomy',
    9: 'Calm',
    10: 'Calm',
    11: 'Amused',
    12: 'Frustrated',
    13: 'Calm',
    14: 'Calm',
    15: 'Calm'
}

# Collected Features
all_features = []

# Sampling Rates
fs_EEG = 512
fs_EMG_ECG = 250
fs_fNIRS = 16

# Band ranges
alpha_band = (8,13)

def compute_bandpower(data, sf, band):
    freqs, psd = welch(data, sf, nperseg = 1024)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])

# Main Loop 

for sub in SUBJECTS:
    try:
        sub_str = f"{sub:03d}"
        print(f"Processing subject {sub_str}...")

        # ---- Load MARKERS ----
        marker_path = os.path.join(DATA_DIR, sub_str, f"{sub_str}_MARKERS.csv")
        if not os.path.exists(marker_path):
            print(f"Missing MARKERS for {sub_str}. Skipping...")
            continue
        markers = pd.read_csv(marker_path, header=None)
        eeg_start = int(markers.iloc[5, 1])
        eeg_end = int(markers.iloc[9, 1])
        ecg_start = int(markers.iloc[7, 1])
        ecg_end = int(markers.iloc[11, 1])
        fnirs_start = int(markers.iloc[6, 1])
        fnirs_end = int(markers.iloc[10, 1])

        # ---- EEG ----
        eeg_path = os.path.join(DATA_DIR, sub_str, f"{sub_str}_EEG.csv")
        if not os.path.exists(eeg_path):
            print(f"Missing EEG for {sub_str}. Skipping...")
            continue
        eeg_data = np.genfromtxt(eeg_path, delimiter=',', skip_header=1)
        eeg_segment = eeg_data[eeg_start:eeg_end]
        eeg_AF7 = eeg_segment[:, 0]
        eeg_AF8 = eeg_segment[:, 1]
        af7_alpha = compute_bandpower(eeg_AF7, fs_EEG, alpha_band)
        af8_alpha = compute_bandpower(eeg_AF8, fs_EEG, alpha_band)
        frontal_asym = np.log(af8_alpha + 1e-6) - np.log(af7_alpha + 1e-6)

        # ---- ECG ----
        ecg_path = os.path.join(DATA_DIR, sub_str, f"{sub_str}_ECG.csv")
        if not os.path.exists(ecg_path):
            print(f"Missing ECG for {sub_str}. Skipping...")
            continue
        ecg_data = pd.read_csv(ecg_path).values.flatten()
        ecg_segment = ecg_data[ecg_start:ecg_end]
        ecg_mean = np.mean(ecg_segment)
        ecg_std = np.std(ecg_segment)
        ecg_rms = np.sqrt(np.mean(ecg_segment**2))

        # ---- EMG ----
        emg_path = os.path.join(DATA_DIR, sub_str, f"{sub_str}_EMG.csv")
        if not os.path.exists(emg_path):
            print(f"Missing EMG for {sub_str}. Skipping...")
            continue
        emg_data = pd.read_csv(emg_path).values.flatten()
        emg_segment = emg_data[ecg_start:ecg_end]  # same markers as ECG
        emg_rms = np.sqrt(np.mean(emg_segment**2))
        emg_var = np.var(emg_segment)

        # ---- fNIRS ----
        snirf_path = os.path.join(DATA_DIR, sub_str, f"{sub_str}.snirf")
        if not os.path.exists(snirf_path):
            print(f"Missing fNIRS for {sub_str}. Skipping...")
            continue
        raw_intensity = read_raw_snirf(snirf_path, preload=True)
        raw_od = optical_density(raw_intensity)
        raw_hb = beer_lambert_law(raw_od)
        hbo = raw_hb.get_data(picks="hbo")[0]
        hbr = raw_hb.get_data(picks="hbr")[0]
        hbo_segment = hbo[fnirs_start:fnirs_end]
        hbr_segment = hbr[fnirs_start:fnirs_end]
        hbo_mean = np.mean(hbo_segment)
        hbr_mean = np.mean(hbr_segment)

        # ---- Assemble Features ----
        row = {
            "subject": sub_str,
            "emotion": subject_emotions[sub],
            "alpha_AF7": af7_alpha,
            "alpha_AF8": af8_alpha,
            "frontal_alpha_asymmetry": frontal_asym,
            "ecg_mean": ecg_mean,
            "ecg_std": ecg_std,
            "ecg_rms": ecg_rms,
            "emg_rms": emg_rms,
            "emg_var": emg_var,
            "hbo_mean": hbo_mean,
            "hbr_mean": hbr_mean
        }
        all_features.append(row)

    except Exception as e:
        print(f"Error in subject {sub_str}: {e}")

# ---- Save to CSV ----
df = pd.DataFrame(all_features)
df.to_csv(RESULTS_CSV, index=False)
print(f"\n Feature dataset saved to {RESULTS_CSV}")