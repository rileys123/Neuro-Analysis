import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import mne
from mne import create_info
from mne.io import read_raw_snirf, RawArray
import numpy as np

###################################### USER INPUT
DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
RESULTS_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure output folder exists

time_window = 30  # Time window to consider before and after t1 [s]
subjects = [sub for sub in range(0, 16) ]  # updated to Subjects 0 to 15 inclusive

####################################################  ECG Settings
fs_ECG = 250 #ECG sampling rate
pts_window = time_window * fs_ECG

# Loop through all subjects
for sub in subjects:
    try:
        print(f"Processing subject: {sub}")
        sub_str = f"{sub:03d}"

        
        # File paths (keeping the if-else structure unchanged)
        ecg_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}_ECG.csv')
        index_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}_MARKERS.csv')

        # Ensure all files exist
        if not (os.path.exists(ecg_file_path) and os.path.exists(index_file_path)):
            print(f"Missing files for subject {sub}. Skipping...")
            continue
        


        # Get the index from the MARKERS file
        index_data = pd.read_csv(index_file_path, header=None)
        index_2_ecg = int(index_data.iloc[7, 1])
        index_1_ecg = int(index_data.iloc[3, 1])
        index_3_ecg = int(index_data.iloc[11, 1])

        #################################################### ECG
        # Import the ECG data
        participant_dir = os.path.join(RESULTS_DIR, f"Participant_{sub_str}")
        os.makedirs(participant_dir, exist_ok=True)
        ecg_data = pd.read_csv(ecg_file_path).values.T 

        # Create MNE RawArray object for filtering
        ch_names = ["ECG"]
        ch_types = ["misc"]  # Set as "misc" instead of "ecg" to avoid recognition issues
        info = create_info(ch_names=ch_names, sfreq=fs_ECG, ch_types=ch_types)
        raw_ecg = RawArray(ecg_data, info)

        # Apply filter
        raw_ecg.set_channel_types({'ECG': 'misc'})  # Ensure MNE treats it as a valid channel
        raw_ecg.filter(l_freq=1, h_freq=45, fir_design='firwin', picks="misc")

        # Get filtered ECG data
        ecg_filtered = raw_ecg.get_data().T  # Transpose back to original shape

        # # Plot and save filtered ECG data
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_filtered / 1000, label="ECG Filtered")  # Convert to mV
        plt.axvline(x=index_2_ecg, color='black', linestyle='--', linewidth=1.5, label='Beg. Stim. Video')
        plt.title(f'ECG Channel Data (Filtered 1-45) - Subject {sub}')
        plt.xlabel('Timepoint')
        plt.xlim([index_2_ecg - pts_window, index_2_ecg + pts_window])
        plt.ylim(-8,8)
        plt.ylabel('mV')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(participant_dir, f"ECG_{sub}.png"))  # Save filtered ECG
        plt.close()

    except Exception as e:
        print(f"An error occurred for subject {sub}: {e}")