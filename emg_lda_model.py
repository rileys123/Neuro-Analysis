import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import mne
from mne import create_info
from mne.io import RawArray
from itertools import combinations
from sklearn.base import clone
from baseline_normalizer import BaselineNormalizer


def evaluate_model(model, features, labels, model_name=""):
    """Enhanced evaluation with train/test metrics"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #5 folds
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
    best_model = metrics['estimator'][0]
    ConfusionMatrixDisplay.from_estimator(best_model, features, labels)
    plt.title(f"{model_name} Confusion Matrix (Fold 1)")
    plt.show()
    
    return metrics

def load_emg_data(subjects, data_dir):
    """
    Load EMG data for all subjects and extract features
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
    
    fs_EMG = 250  # Sampling frequency
    glob_list_baseline=[]
    glob_list_emotional=[]
    quadrants = []

    
    for sub in subjects:
        try:
            sub_str = f"{sub:03d}"
            emg_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_EMG.csv')
            markers_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_MARKERS.csv')
            
            if not (os.path.exists(emg_file) and os.path.exists(markers_file)):
                print(f"Missing files for subject {sub}")
                continue
                
            # Load EMG data
            emg_data = pd.read_csv(emg_file).values.T[0]  # Get first channel
            
            # Load markers to find emotional video segment
            markers = pd.read_csv(markers_file, header=None)
            t1 = int(markers.iloc[7, 1])  # Start of emotional video
            
            # Extract the emotional segment (30s window)
            window_size = 30 * fs_EMG  # 30 seconds worth of samples
            baseline_segment = emg_data[t1-window_size:t1]
            emotional_segment = emg_data[t1:t1+window_size]

            # Ensure both segments have correct length
            
            if len(baseline_segment) < window_size:
                print(f"Skipping subject {sub}: insufficient baseline or emotional segment length.")
                continue
            

            glob_list_baseline.append(baseline_segment)
            glob_list_emotional.append(emotional_segment)
            quadrants.append(subject_quadrant_mapping[sub])
            
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
    
    return glob_list_baseline, glob_list_emotional, quadrants

def normalize_emg_features(baseline_segment, emotional_segment):
    fs = 250
    normalizer = BaselineNormalizer(fs)
    features = []


    for baseline, emotional in zip(baseline_segment, emotional_segment):
        if len(baseline) == 0 or len(emotional) == 0:
            continue

        # Compute ratio: (after / before)
        feature_vector = normalizer.compute_feature_ratio(
            baseline,
            emotional,
            extractor_func = extract_emg_features
        )
            

        features.append(feature_vector)

    return np.array(features)
            

def extract_emg_features(emg_signal, fs): #raw EMG data to 4 features RMS (root mean square), MAV (mean abs val), ZCR (zero crossing rate), power ratio
    """
    Extract emotion-relevant features from EMG signal
    Returns: array([rms, mean_amplitude, zero_crossings, power_ratio])
    """
    # 1. Apply bandpass filter (1-20 Hz as in your EMG.py)
    b, a = butter(4, [1, 20], btype='bandpass', fs=fs)
    emg_filtered = filtfilt(b, a, emg_signal)

    # remove DC offset and detrend
    emg_filtered = emg_filtered - np.mean(emg_filtered)
    
    # 2. Calculate features
    # A. Root Mean Square (RMS) - muscle activation level, signal power
    rms = np.sqrt(np.mean(np.square(emg_filtered)))
    
    # B. Mean Absolute Value (MAV) - average amplitude
    mav = np.mean(np.abs(emg_filtered))
    
    # C. Zero Crossing Rate (ZCR) - frequency content
    zero_crossings = len(np.where(np.diff(np.sign(emg_filtered)))[0])
    
    # D. Power ratio (high vs low frequency)
    freqs, psd = welch(
        emg_filtered,
        fs,
        nperseg=min(512, len(emg_filtered)//2),
        noverlap=min(256, len(emg_filtered)//4)
    )
    low_power = np.trapz(psd[(freqs >= 1) & (freqs < 10)])
    high_power = np.trapz(psd[(freqs >= 10) & (freqs < 20)])
    power_ratio = high_power / low_power if low_power > 1e-6 else 0
    
    return np.array([rms, mav, zero_crossings, power_ratio])

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

def evaluate_quadrant_pairs_emg(features, labels, model):
    """
    Evaluate EMG model performance on all quadrant pairs with muscle-specific validation
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
        return pd.DataFrame(columns=['Quadrant Pair', 'Accuracy', 'Std', 'n_samples', 'method', 'features'])
    return pd.DataFrame(results)


# Main execution
if __name__ == "__main__":
    DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
    subjects = range(16)  # Subjects 0-15
    
   # 1. Load data and extract features
    print("Loading EMG data and extracting features...")
    baseline_data, emotional_data, quadrants = load_emg_data(subjects, DATA_DIR)
    features = normalize_emg_features(baseline_data,emotional_data)
    
    

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
    
    '''
    # Full dataset performance
    print("\nOverall Performance:")
    try:
        metrics = evaluate_model(model, features, quadrants, model_name)
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
    '''

    # Quadrant-pair performance
    print("\nQuadrant-Pair Performance:")
    pair_results = evaluate_quadrant_pairs_emg(features, quadrants, model)
    print(pair_results)
    
    # Visualize pair performance if we have results
    if not pair_results.empty:
        plt.figure(figsize=(10,4))
        sns.barplot(data=pair_results, x='Quadrant Pair', y='Accuracy')  # Match the column name
        plt.title(f"{model_name}: Accuracy by Quadrant Pair")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)  # Rotate labels if needed
        plt.tight_layout()  # Adjust layout
        plt.show()
        
        # Print interpretation based on EMG characteristics
        print("\nInterpretation Guide:")
        print("- High accuracy pairs may reflect muscle tension differences")
        print("- Q1 (Amused) vs neutral_quadrant (Frustrated): Compare facial muscle activity")
        print("- Q3 (Gloomy) vs Q4 (Calm): Compare overall muscle tension")
    else:
        print("No valid quadrant pairs could be evaluated")