import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration Variables
CHANNEL_NAMES = ["AF7", "AF8", "F3", "F4", "PO7", "PO8", "PO3", "PO4"]
SFREQ = 512  # Sampling frequency in Hz

# File paths
DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
RESULTS_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"

# Filter settings
BANDPASS_FILTER = (5, 40)

# Time series visualization settings
WINDOW_BEFORE = 30  # Seconds before stimulus
WINDOW_AFTER = 30   # Seconds after stimulus

# Loop through data files
for file_id in range(16):
    file_id_str = f"{file_id:03d}"
    file_path = os.path.join(DATA_DIR, file_id_str, f"{file_id_str}_EEG.csv")
    file_path_mark = os.path.join(DATA_DIR, file_id_str, f"{file_id_str}_MARKERS.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load EEG data
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='float64', missing_values='', filling_values=np.nan)
    data = data[~np.isnan(data).any(axis=1)]

    # Load stimulus markers
    data_M = np.loadtxt(file_path_mark, delimiter=',', skiprows=5, dtype='str')
    Begin_stimulus_index = int(data_M[0, 1])

    # Loop through channels
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        eeg_data = data[:, channel_idx].reshape(-1, 1)
        ch_names = [f"EEG_{channel_name}"]

        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(eeg_data.T, info)

        # Apply Bandpass and Notch Filters
        raw_filtered = raw.copy().filter(l_freq=BANDPASS_FILTER[0], h_freq=BANDPASS_FILTER[1], fir_design='firwin')

        # Extract filtered data
        eeg_after = raw_filtered.get_data()[0]

        # Compute PSD
        psd_after = raw_filtered.compute_psd(fmax=60)
        psd_after_values = psd_after.get_data()
        freqs_after = psd_after.freqs

        # Convert stimulus index to time
        stimulus_time = Begin_stimulus_index / SFREQ
        window_start_time = stimulus_time - WINDOW_BEFORE
        window_end_time = stimulus_time + WINDOW_AFTER

        # # Plot EEG signals and PSD for the current channel
        fig, axs = plt.subplots(2, 1, figsize=(14, 8))

        axs[0].plot(np.arange(len(eeg_after)) / SFREQ, eeg_after, 'r', label=f"After Filtering ({BANDPASS_FILTER[0]}-{BANDPASS_FILTER[1]}Hz + Notch)")
        axs[0].axvline(stimulus_time, color='k', linestyle='--')
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("EEG Amplitude (µV)")
        axs[0].set_title(f"EEG After Filtering - {channel_name}")
        axs[0].legend()
        axs[0].grid()
        axs[0].set_xlim([window_start_time, window_end_time])
        axs[0].set_ylim(-400,400)

        axs[1].plot(freqs_after, psd_after_values.mean(axis=0), 'r', label=f"After Filtering ({BANDPASS_FILTER[0]}-{BANDPASS_FILTER[1]}Hz + Notch)")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Power (µV²/Hz)")
        axs[1].set_title("PSD After Filtering")
        axs[1].legend()
        axs[1].grid()
        axs[1].set_xlim([5, None])


        # Save figure for the current channel (separate file per channel)
        save_dir = os.path.join(RESULTS_DIR, f"Participant_{file_id_str}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_id_str}_EEG_{channel_name}.jpg")
        plt.tight_layout()
        plt.savefig(save_path, format='jpg')
        plt.close()

        print(f"Saved: {save_path}")


        stop=1