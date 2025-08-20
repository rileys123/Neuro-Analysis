# baseline_normalizer.py

import numpy as np

class BaselineNormalizer:
    """
    Normalizes physiological features based on a participant's baseline.
    Features are returned as ratio: post_video_feature / baseline_feature
    """

    def __init__(self, fs): # constructor
        self.fs = fs  # sampling rate for modalities like ECG, EEG, EMG, etc.

    def compute_feature_ratio(self, signal_before, signal_after, extractor_func):
        """
        Computes feature ratio for a given signal and feature extraction function.
        
        Params:
        - signal_before: np.array of baseline signal
        - signal_after: np.array of post-stimulus signal
        - extractor_func: function that extracts features from the signal
        
        Returns:
        - np.array of feature ratios (post / pre)
        """
        # Force features to NumPy arrays
        feat_before = np.array(extractor_func(signal_before, self.fs))
        feat_after = np.array(extractor_func(signal_after, self.fs))
        
        # Avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(feat_before != 0, feat_after / feat_before, 0)
        return ratio # return computed feature ratio as a numpy array
