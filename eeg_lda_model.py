import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from itertools import combinations
from sklearn.base import clone
from baseline_normalizer import BaselineNormalizer


# Configuration
CHANNEL_NAMES = ["AF7", "AF8", "F3", "F4", "PO7", "PO8", "PO3", "PO4"]
SFREQ = 512  # Sampling frequency in Hz
BANDPASS_FILTER = (5, 40)  # EEG bandpass filter range

def evaluate_model(model, features, labels, model_name=""):
    """Evaluates a given ML model using 5 fold cross validation and print results"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = cross_validate(
        model, features, labels, cv=cv,
        scoring='accuracy',
        return_train_score=True,
        return_estimator=True
    )
    
    print(f"\n{model_name} Performance:")
    print(f"Train Accuracy: {np.mean(metrics['train_score']):.2f} ± {np.std(metrics['train_score']):.2f}")
    print(f"Test Accuracy: {np.mean(metrics['test_score']):.2f} ± {np.std(metrics['test_score']):.2f}")
    
    # Plot confusion matrix from first fold
    best_model = metrics['estimator'][0] #use first trained model to plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(best_model, features, labels) #display confusion matrix
    plt.title(f"{model_name} Confusion Matrix (Fold 1)")
    plt.show()
    
    return metrics

def load_eeg_data(subjects, data_dir):
    """
    Load EEG data for all subjects and extract features
    """
    # quadrant mapping
    subject_quadrant_mapping = {
        0: 3,  # Gloomy
        1: 4,  # Calm
        2: 4,  # Satisfied
        3: 4,
        4: 1,  # Amused
        5: 4,
        6: 1,
        7: 2,  # Frustrated
        8: 3,
        9: 4,
        10: 4,
        11: 1,
        12: 2,
        13: 4,
        14: 4,
        15: 4
    }

    glob_list_baseline=[]
    glob_list_emotional=[]
    quadrants = []
    
    
    for sub in subjects:
        try:
            sub_str = f"{sub:03d}"
            eeg_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_EEG.csv')
            markers_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_MARKERS.csv')
            
            if not (os.path.exists(eeg_file) and os.path.exists(markers_file)):
                print(f"Missing files for subject {sub}")
                continue #skip subject if files missing
                
            # Load EEG data
            eeg_data = np.genfromtxt(eeg_file, delimiter=',', skip_header=1, dtype='float64', missing_values='', filling_values=np.nan)
            eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)] #removes NaN values
            
            # Load stimulus markers
            data_M = np.loadtxt(markers_file, delimiter=',', skiprows=5, dtype='str')
            t1 = int(data_M[0, 1])  # Start of emotional video
            
            # Extract the emotional segment (30s window)
            window_size = 30 * SFREQ  #num of samples in 30 seconds

            if t1<window_size:
                starting_index=0
            else:
                starting_index=t1-window_size

            if t1+window_size > len(eeg_data)-1:
                last_index = len(eeg_data)-1
            else:
                last_index = t1+window_size

            
            baseline_segment = eeg_data[starting_index:t1]
            emotional_segment = eeg_data[t1:last_index] #emotional video

            # if len(emotional_segment) < window_size or len(baseline_segment) < window_size:
            #     print(f"Subject {sub}: Not enough emotional or baseline data — skipping.")
            #     continue

            glob_list_baseline.append(baseline_segment)
            glob_list_emotional.append(emotional_segment)
            quadrants.append(subject_quadrant_mapping[sub])
            
            
            print(f"Processed Subject {sub}")
            
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
    
    return glob_list_baseline, glob_list_emotional, quadrants

def normalize_eeg_features(baseline_segment, emotional_segment):
    normalizer = BaselineNormalizer(SFREQ)
    features = []

    for baseline, emotional in zip(baseline_segment, emotional_segment):
            if len(baseline) == 0 or len(emotional) == 0:
                continue

            # Compute ratio: (after / before)
            feature_vector = normalizer.compute_feature_ratio(
                baseline,
                emotional,
                extractor_func = extract_eeg_features
            )
                

            features.append(feature_vector)

    return np.array(features)


def extract_eeg_features(eeg_signal, fs):
    """
    Extract emotion-relevant features from EEG signal for a single channel
    Returns: array of features for the channel
    """
    # Create MNE Raw object for filtering
    ch_names = ["EEG"]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg_signal.reshape(1, -1), info)
    
    # Apply bandpass filter (5-40 Hz as in your EEG.py)
    raw_filtered = raw.copy().filter(l_freq=BANDPASS_FILTER[0], h_freq=BANDPASS_FILTER[1], fir_design='firwin')
    eeg_filtered = raw_filtered.get_data()[0]
    
    # Calculate features
    features = []
    
    # 1. Power in different frequency bands
    freqs, psd = welch(eeg_filtered, fs=fs, nperseg=min(1024, len(eeg_filtered)//2)) #power spectral density
    
    # Delta (1-4 Hz) (DROPPED)
    delta = np.trapz(psd[(freqs >= 1) & (freqs < 4)])
    
    # Theta (4-8 Hz)
    theta = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
    
    # Alpha (8-13 Hz)
    alpha = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
    
    # Beta (13-30 Hz)
    beta = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
    
    # Gamma (30-40 Hz)
    gamma = np.trapz(psd[(freqs >= 30) & (freqs <= 40)])
    
    # 2. Power ratios
    theta_alpha = theta / alpha if alpha > 1e-6 else 0 #avoid dividng by near 0
    beta_alpha = beta / alpha if alpha > 1e-6 else 0
    
    # 3. Statistical features (DROPPED)
    mean_val = np.mean(eeg_filtered)
    std_val = np.std(eeg_filtered)
    skewness = pd.Series(eeg_filtered).skew()
    kurtosis = pd.Series(eeg_filtered).kurtosis()
    
    return [theta, alpha, beta, gamma, theta_alpha, beta_alpha]

def train_lda_model(features, quadrants, k_folds=5):
    """
    Train and evaluate LDA model using k-fold cross-validation
    """
    
    
    # Initialize LDA
    lda = LinearDiscriminantAnalysis()
    
    # Stratified k-fold (preserves class balance)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=22)
    
    # Cross-validation
    cv_scores = cross_val_score(lda, features, quadrants, cv=skf, scoring='accuracy')
    
    print(f"\n{k_folds}-Fold Cross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
    
    # Train final model on all data (for inspection)
    lda.fit(features, quadrants)
    
    # Generate confusion matrix from one fold
    _, test_idx = next(StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42).split(features, quadrants))
    y_pred = lda.predict(features[test_idx])
    
    # Plot confusion matrix
    quadrant_emotion_map = {
        1: "Amused",
        2: "Frustrated",
        3: "Gloomy", 
        4: "Calm/Satisfied"
    }
    
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(quadrants[test_idx], y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=[quadrant_emotion_map[i] for i in sorted(quadrant_emotion_map)],
                yticklabels=[quadrant_emotion_map[i] for i in sorted(quadrant_emotion_map)])
    plt.title(f'Confusion Matrix (One Fold Example)\nAccuracy: {cv_scores[0]:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return lda, np.mean(cv_scores)

def train_svm_model(features, quadrants, k_folds=5):
    """
    Train SVM with RBF kernel using k-fold CV
    """
    # Create pipeline (SVM)
    svm_pipe = make_pipeline(
        SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    )
    
    # Stratified k-fold CV
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm_pipe, features, quadrants, cv=skf)
    
    print(f"\nSVM {k_folds}-Fold CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
    
    # Train final model
    svm_pipe.fit(features, quadrants)
    return svm_pipe, np.mean(cv_scores)

def evaluate_quadrant_pairs_eeg(features, labels, model):
    """
    Evaluate EEG model performance on all quadrant pairs with channel-aware validation
    Returns: DataFrame with accuracy for each pair
    """
    results = [] #holds results for each quadrant pair, each result is a dictionary w/ info ab accuracy, std, method
    target_quadrant = 4
    neutral_quadrant = 5

    X_binary = []
    y_binary = []

    for i in range(len(labels)):
        original_label = labels[i]
        if original_label == target_quadrant:
            X_binary.append(features[i])
            y_binary.append(target_quadrant)
        elif original_label in [1,2,3]:
            X_binary.append(features[i])
            y_binary.append(neutral_quadrant)

    X_binary = np.array(X_binary)
    y_binary = np.array(y_binary)     
        
    n_splits = 3
    min_samples_per_class = min(np.sum(y_binary == target_quadrant), np.sum(y_binary == neutral_quadrant))
    if min_samples_per_class < n_splits:
        print(f"Warning: Not enough samples for {n_splits}-fold CV. Adjusting n_splits to {min_samples_per_class}.")
        n_splits = min_samples_per_class
        if n_splits < 3: # Cannot perform CV with less than 2 samples per class
            print(f"Skipping {target_quadrant} vs. Others: Not enough samples for CV (min {min_samples_per_class}).")
            return pd.DataFrame(columns=['Quadrant Pair', 'Accuracy', 'Std', 'n_samples', 'method'])

    unique_labels, counts = np.unique(y_binary, return_counts=True)
    print(f"Binary Label Counts (total): {dict(zip(unique_labels, counts))}")
    try:
        # Use StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_accuracies = []
        y_test_fold_first = None
        y_pred_fold_first = None

        for i, (train_index, test_index) in enumerate(skf.split(X_binary, y_binary)):
            X_train_fold, X_test_fold = X_binary[train_index], X_binary[test_index]
            y_train_fold, y_test_fold = y_binary[train_index], y_binary[test_index]

            # Fit the model for the current fold
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)

            cv_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))

            if i == 0: # Store results from the first fold for confusion matrix display
                y_test_fold_first = y_test_fold
                y_pred_fold_first = y_pred_fold
                
        mean_accuracy = np.mean(cv_accuracies)
        std_accuracy = np.std(cv_accuracies)

        # Store results
        results.append({
            'Quadrant Pair': f"{target_quadrant} vs {neutral_quadrant}",
            'Accuracy': mean_accuracy,
            'Std': std_accuracy,
            'n_samples': len(y_binary),
            'method': f'{n_splits}-fold CV'
        })

        # Plot 2x2 confusion matrix from the first fold
        if y_test_fold_first is not None and y_pred_fold_first is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test_fold_first, y_pred_fold_first),
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=[neutral_quadrant, target_quadrant],
                        yticklabels=[neutral_quadrant, target_quadrant])
            plt.title(f'Confusion Matrix: {target_quadrant} vs. Others (First CV Fold)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            st=1

    except Exception as e:
        print(f"Failed {target_quadrant} vs. Others: {str(e)}")
        
    
    # Return empty DataFrame with correct columns if no results
    if not results:
        return pd.DataFrame(columns=['Quadrant Pair', 'Accuracy', 'Std', 'n_samples', 'method', 'n_features'])
    return pd.DataFrame(results)

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
    subjects = range(16)  # Subjects 0-15
    
    # 1. Load data and extract features
    print("Loading EEG data and extracting features...")
    baseline_data, emotional_data, quadrants = load_eeg_data(subjects, DATA_DIR)
    features = normalize_eeg_features(baseline_data,emotional_data)
    
    # Print feature information
    print(f"\nFeature matrix shape: {features.shape}") #rows = subjects, columns = EEG features
    print(f"Number of channels: {len(CHANNEL_NAMES)}") #EEG channels used
    print(f"Features per channel: {len(extract_eeg_features(np.random.rand(100), SFREQ))}") #how many features per EEG channel
    
    # 2. Set up models
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    }
    
    # 3. Evaluate each model
    #for model_name, model in models.items():
    model_name = "LDA"
    model = models[model_name]

    print(f"\n=== {model_name} Evaluation ===")
    
    # Full dataset performance
    # print("\nOverall Performance:")
    # metrics = evaluate_model(model, features, quadrants, model_name)
    
    # Quadrant-pair performance
    print("\nQuadrant-Pair Performance:")
    pair_results = evaluate_quadrant_pairs_eeg(features, quadrants, model)
    print(pair_results) # raw results table
    
    # Visualize pair performance if we have results
    if not pair_results.empty:
        plt.figure(figsize=(10,4))
        sns.barplot(data=pair_results, x='Quadrant Pair', y='Accuracy')  # Match the column name
        plt.title(f"{model_name}: Accuracy by Quadrant Pair")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)  # Rotate labels if needed
        plt.tight_layout()  # Adjust layout
        plt.show()
    else:
        print("No valid quadrant pairs could be evaluated")