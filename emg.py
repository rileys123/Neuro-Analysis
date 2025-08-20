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
subjects = [sub for sub in range(0, 16) ]  

#################################################### EMG, Settings
fs_EMG = 250
fs_NIRS = 16  # Sampling frequency for fNIRS
pts_window = time_window * fs_EMG

# Loop through all subjects
for sub in subjects:
    try:
        print(f"Processing subject: {sub}")
        sub_str = f"{sub:03d}"
        # File paths (keeping the if-else structure unchanged)
        emg_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}_EMG.csv')
        index_file_path = os.path.join(DATA_DIR, f'{sub_str}', f'{sub_str}_MARKERS.csv')

        # Ensure all files exist
        if not (os.path.exists(emg_file_path)  and os.path.exists(index_file_path) ):
            print(f"Missing files for subject {sub}. Skipping...")
            continue

        # Get the index from the MARKERS file
        index_data = pd.read_csv(index_file_path, header=None)
        index_2_emg = int(index_data.iloc[7, 1])
        index_1_emg = int(index_data.iloc[3, 1])
        index_3_emg = int(index_data.iloc[11, 1])

        #################################################### EMG
        emg_data = pd.read_csv(emg_file_path).values.T 
        emg_data_of_interest = emg_data[0,index_1_emg:index_3_emg]

        # Create MNE RawArray object for filtering
        ch_names = ["EMG"]
        ch_types = ["misc"]  # Set as "misc" instead of "emg" to avoid recognition issues
        info = create_info(ch_names=ch_names, sfreq=fs_EMG, ch_types=ch_types)
        raw_emg = RawArray(emg_data, info)

        # Apply filter
        raw_emg.set_channel_types({'EMG': 'misc'})  # Ensure MNE treats it as a valid channel
        raw_emg.filter(l_freq=1, h_freq=20, fir_design='firwin', picks="misc")

        # Get filtered ECG data
        emg_filtered = raw_emg.get_data().T  # Transpose back to original shape

        # Create a directory for the current participant
        participant_dir = os.path.join(RESULTS_DIR, f"Participant_{sub_str}")
        os.makedirs(participant_dir, exist_ok=True)

        # # Plot and save EMG data
        plt.figure(figsize=(12, 6))
        plt.plot(emg_filtered / 1000, label="EMG")  # Convert to mV
        plt.axvline(x=index_2_emg, color='black', linestyle='--', linewidth=1.5, label='Beg. Stim. Video')
        plt.title(f'EMG Channel Data - Subject {sub}')
        plt.xlabel('Timepoint')
        plt.xlim([index_2_emg - pts_window, index_2_emg + pts_window])
        plt.ylabel('mV')
        plt.legend()

        # Set y-axis limits to a range around the mean of the centered data
        ymin = np.min(emg_data_of_interest) / 1000  # Convert to mV
        ymax = np.max(emg_data_of_interest) / 1000  # Convert to mV
        plt.ylim(ymin, ymax)    

        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(participant_dir, f"EMG_{sub}.png"))  # Save to the participant-specific folder
        plt.close()

    except Exception as e:
        print(f"An error occurred for subject {sub}: {e}")