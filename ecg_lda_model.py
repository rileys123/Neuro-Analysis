import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, iirnotch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #imports LDA model, dimensionality reduction and classification
from sklearn.model_selection import train_test_split #splits data into training and testing sets
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay #evaluate model's performance
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns #graphs
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from itertools import combinations
import sys
from sklearn.base import clone
from baseline_normalizer import BaselineNormalizer

def evaluate_model(model, features, labels, model_name=""):
    """Enhanced evaluation with train/test metrics"""
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
    best_model = metrics['estimator'][0]
    ConfusionMatrixDisplay.from_estimator(best_model, features, labels)
    plt.title(f"{model_name} Confusion Matrix (Fold 1)")
    plt.show()
    
    return metrics

# 1. Load and Prepare Data
def load_ecg_data(subjects, data_dir):
    """
    Load ECG data for all subjects and extract features
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
    
    fs = 250 #ECG sampling rate

    glob_list_baseline=[]
    glob_list_emotional=[]
    quadrants = []

    for sub in subjects: #loop thru each subject
        try:
            sub_str = f"{sub:03d}" #3 digit string format like 003
            ecg_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_ECG.csv')
            markers_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_MARKERS.csv') #build full file paths for ECG and marker CSVs
            
            if not (os.path.exists(ecg_file) and os.path.exists(markers_file)):
                print(f"Missing files for subject {sub}")
                continue #skip subject if required file missing
                
            # Load ECG data
            ecg_data = pd.read_csv(ecg_file).values.flatten() #read ECG data as 1D numpy array
            
            # Load markers to find emotional video segment
            markers = pd.read_csv(markers_file, header=None)
            t1 = int(markers.iloc[7, 1])  # Start of emotional video
            
            # Extract the emotional segment (30s window)
            window_size = 30 * fs  # 30 seconds worth of samples
            baseline_segment = ecg_data[t1-window_size:t1]

            emotional_segment = ecg_data[t1:t1+window_size]
            

            if len(baseline_segment) < window_size:
                print(f"Subject {sub}: baseline too short even from t0 — skipping.")
                continue

            glob_list_baseline.append(baseline_segment)
            glob_list_emotional.append(emotional_segment)
            quadrants.append(subject_quadrant_mapping[sub])

        except Exception as e:
            print(f"Error processing subject {sub}: {e}") #catches and prints errors
    
    return glob_list_baseline, glob_list_emotional, quadrants 

# 2. Feature Extraction 
def normalize_ecg_features(baseline_segment, emotional_segment):
    """
    Normalize ECG features and return array of ratio (emotional to baseline)
    """

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
            extractor_func = extract_ecg_features
        )
            

        features.append(feature_vector)

    return np.array(features)


def extract_ecg_features(sub_data, fs=250):
        
    """
    Extract emotion-relevant features with 0.5-30 Hz bandpass filtering
    Returns: array([heart_rate, lf_hf_ratio, rmssd])
    """
    # 1. Apply bandpass filter (0.5-30 Hz)
    b, a = butter(4, [0.03, 30], btype='bandpass', fs=fs)
    ecg_filtered = filtfilt(b, a, sub_data)
    
    # remove DC offset and detrend
    ecg_filtered = ecg_filtered - np.mean(ecg_filtered)

    # 2. Heart Rate Detection (optimized threshold)
    peaks, _ = find_peaks(
        ecg_filtered,
        height=np.mean(ecg_filtered) + 2*np.std(ecg_filtered),
        distance=int(fs*0.6),  # Min 0.6s between beats
        prominence=np.std(ecg_filtered)
    )
    
    # 3. Calculate Features
    # A. Heart Rate (bpm)
    hr = len(peaks) / (len(ecg_filtered)/fs/60) if len(peaks) > 1 else 0
    
    # B. LF/HF Ratio (Welch's PSD with 50% overlap) !!! can't do this, LF doesn't work unless you have ~2 min of sub_data, choose to do only HF?
    nperseg = min(4096, len(ecg_filtered)) #increased nperseg (number of samples per second) from 1024
    freqs, psd = welch(
        ecg_filtered,
        fs,
        nperseg=nperseg,
        noverlap=nperseg//2,
        scaling='density',
        detrend='constant'
    )
    
    # Calculate power in bands using proper frequency integration
    lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], 
                        freqs[(freqs >= 0.04) & (freqs < 0.15)])
    hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], 
                        freqs[(freqs >= 0.15) & (freqs < 0.4)])
    
    lf_hf = lf_power / hf_power if hf_power > 1e-6 else 0




    # C. RMSSD (with physiologically valid RR intervals)
    rr = np.diff(peaks)/fs*1000 if len(peaks)>1 else np.array([0])
    valid_rr = rr[(rr > 300) & (rr < 1500)]  # Normal RR range (300-1500ms)
    rmssd = np.sqrt(np.mean(np.square(np.diff(valid_rr)))) if len(valid_rr)>1 else 0

    return np.array([hr, lf_hf, rmssd])


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
    
    '''
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(quadrants[test_idx], y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=[quadrant_emotion_map[i] for i in sorted(quadrant_emotion_map)],
                yticklabels=[quadrant_emotion_map[i] for i in sorted(quadrant_emotion_map)])
    plt.title(f'Confusion Matrix (One Fold Example)\nAccuracy: {cv_scores[0]:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''

    return lda, np.mean(cv_scores)

def train_svm_model(features, quadrants, k_folds=5):
    """
    Train SVM with RBF kernel using k-fold CV
    """
    # Create pipeline 
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


def evaluate_quadrant_pairs(features, labels, model): 
    #takes 3 extracted features, quadrant labels, ML classifier (LDA or SVM), quadrant IDs

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
        return pd.DataFrame(columns=['Quadrant Pair', 'Accuracy', 'Std', 'n_samples', 'method'])
    return pd.DataFrame(results)
    
# Main execution
if __name__ == "__main__":
    fs=250
    DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
    subjects = range(16)  # Subjects 0-15
    
    # 1. Load data
    print("Loading data...")
    baseline_data, emotional_data, quadrants = load_ecg_data(subjects, DATA_DIR)
    features = normalize_ecg_features(baseline_data,emotional_data)


    if len(features) == 0:
        print("No data loaded - exiting")
        sys.exit(1)
    
    # 2. Set up models
    models = {
        "LDA": LinearDiscriminantAnalysis(),
       
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    }
    
    # 3. Evaluate each model
    #for model_name, model in models.items():

    model_name = "SVM"
    model = models[model_name]
    print(f"\n=== {model_name} Evaluation ===")
    
    '''
    commented out to plot 2x2 matrix instead of 4x4
    # Full dataset performance
    print("\nOverall Performance:")
    metrics = evaluate_model(model, features, quadrants, model_name)
    '''

    # Quadrant-pair performance
    print("\nQuadrant-Pair Performance:")
    pair_results = evaluate_quadrant_pairs(features, quadrants, models[model_name])
    print(pair_results)
    
    # Visualize pair performance
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