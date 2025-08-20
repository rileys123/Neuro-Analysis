import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import mne
from mne import create_info
from mne.io import read_raw_snirf, RawArray
from mne.preprocessing.nirs import beer_lambert_law, optical_density
import numpy as np

###################################### USER INPUT
DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
RESULTS_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure output folder exists

subjects = [sub for sub in range(0, 16)]  # Subjects 0 to 15 inclusive

#################################################### fNIRS Settings
fs_NIRS = 16  # Sampling frequency for fNIRS

# Number of initial points to remove (if applicable)
remove_points = 0  # Adjust this based on your data (e.g., first 50 points are noisy)

# Loop through all subjects
for sub in subjects:
    try:
        print(f"Processing subject: {sub}")
        sub_str = f"{sub:03d}"
        # File paths (keeping the if-else structure unchanged)
        index_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}_MARKERS.csv')
        nirs_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}.snirf')

        # Ensure all files exist
        if not ( os.path.exists(index_file_path) and os.path.exists(nirs_file_path)):
            print(f"Missing files for subject {sub}. Skipping...")
            continue

        # Get the index from the MARKERS file
        index_data = pd.read_csv(index_file_path, header=None)
        index_2_fnirs = int(index_data.iloc[6, 1])
        index_1_fnirs = int(index_data.iloc[2, 1])
        index_3_fnirs = int(index_data.iloc[10, 1])

        # Create a directory for the current participant
        participant_dir = os.path.join(RESULTS_DIR, f"Participant_{sub_str}")
        os.makedirs(participant_dir, exist_ok=True)

        #################################################### fNIRS (HbO and HbR)
        raw_intensity = read_raw_snirf(nirs_file_path, preload=True)
        raw_od = optical_density(raw_intensity)
        raw_hb = beer_lambert_law(raw_od)

        data_s7d1_hbo = raw_hb._data[1]
        data_s7d1_hbr = raw_hb._data[25]

        data_s7d1_hbo_micro = data_s7d1_hbo / (10**-6)
        data_s7d1_hbr_micro = data_s7d1_hbr / (10**-6)

        # Plot and save fNIRS data
        plt.figure(figsize=(12, 6))
        plt.plot(data_s7d1_hbo_micro, label='HbO [µM]', color='blue')
        plt.plot(data_s7d1_hbr_micro, label='HbR [µM]', color='orange')
        plt.axvline(x=index_2_fnirs, color='black', linestyle='--', linewidth=1.5, label='Beg. Stim. Video')
        plt.title(f'HbO and HbR Concentrations - Subject {sub}')
        plt.xlabel('Time (Samples)')
        plt.ylabel('Concentration [µM]')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(participant_dir, f"HbO_HbR_{sub}.png"))  # Save to the participant-specific folder
        plt.close()

    except Exception as e:
        print(f"An error occurred for subject {sub}: {e}")