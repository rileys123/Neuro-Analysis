import os
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_snirf
from mne.preprocessing.nirs import beer_lambert_law, optical_density
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
from sklearn.decomposition import PCA
from baseline_normalizer import BaselineNormalizer


# Configuration
FS_NIRS = 16  # Sampling frequency for fNIRS

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

def load_fnirs_data(subjects, data_dir):
    """
    Load fNIRS data for all subjects and extract features
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
            nirs_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}.snirf')
            markers_file = os.path.join(data_dir, f'{sub_str}', f'{sub_str}_MARKERS.csv')
            
            if not (os.path.exists(nirs_file) and os.path.exists(markers_file)):
                print(f"Missing files for subject {sub}")
                continue
                
            # Load fNIRS data and convert to hemoglobin concentrations
            raw_intensity = read_raw_snirf(nirs_file, preload=True)
            raw_od = optical_density(raw_intensity)
            raw_hb = beer_lambert_law(raw_od)
            
            # remove DC offset
            raw_hb.filter(l_freq=0.01, h_freq=0.2, h_trans_bandwidth=0.05, verbose=False)

            # Get HbO and HbR data
            hbo_data_temp = raw_hb._data[1] / (10**-6)  # Channel 1 for HbO (µM)
            hbo_data = hbo_data_temp - np.mean(hbo_data_temp)

            hbr_data_temp = raw_hb._data[25] / (10**-6)  # Channel 25 for HbR (µM)
            hbr_data = hbr_data_temp - np.mean(hbr_data_temp)

            
            # Load markers to find emotional video segment
            markers = pd.read_csv(markers_file, header=None)
            t1 = int(markers.iloc[6, 1])  # Start of emotional video (index_1_fnirs)
            
            # Extract the emotional segment (30s window)
            window_size = 30 * FS_NIRS  # 30 seconds worth of samples

            if t1<window_size:
                starting_index=0
            else:
                starting_index=t1-window_size

            hbo_baseline = hbo_data[starting_index:t1]
            hbr_baseline = hbr_data[starting_index:t1]

            hbo_emotion = hbo_data[t1:t1+window_size]
            hbr_emotion = hbr_data[t1:t1+window_size]
            
            # Ensure segment lengths are valid
            # if len(hbo_baseline) < window_size or len(hbo_emotion) < window_size:
            #     print(f"Skipping subject {sub}: insufficient baseline or emotional segment.")
            #     continue
            # Final segment length validation
            # if len(hbo_baseline) < window_size or len(hbr_baseline) < window_size:
            #     print(f"Skipping subject {sub}: Baseline segment too short after extraction.")
            #     continue
            # if len(hbo_emotion) < window_size or len(hbr_emotion) < window_size:
            #     print(f"Skipping subject {sub}: Emotional segment too short after extraction.")
            #     continue

            glob_list_baseline.append((hbo_baseline, hbr_baseline))
            glob_list_emotional.append((hbo_emotion, hbr_emotion))
            quadrants.append(subject_quadrant_mapping[sub])
            
            
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
    
    return glob_list_baseline, glob_list_emotional, quadrants 

def normalize_fnirs_features(baseline_segment, emotional_segment):
    normalizer = BaselineNormalizer(FS_NIRS)
    features = []

    for baseline, emotional in zip(baseline_segment, emotional_segment):
        if len(baseline) == 0 or len(emotional) == 0:
            continue

        # Compute ratio: (after / before)
        feature_vector = normalizer.compute_feature_ratio(
            baseline,
            emotional,
            extractor_func = lambda signal_tuple, sample_rate: extract_fnirs_features(signal_tuple[0], signal_tuple[1], sample_rate)
        )
            

        features.append(feature_vector)

    return np.array(features)


def extract_fnirs_features(hbo_segment, hbr_segment, fs):
    """
    Extract emotion-relevant features from fNIRS signals
    Returns: array of features combining HbO and HbR information
    """
    features = []
    
    
    # Time-domain features for HbO
    features.append(np.mean(hbo_segment))
    features.append(np.mean(hbr_segment))
    features.append(np.std(hbo_segment))
    features.append(np.std(hbr_segment))
    
    '''
    # Time-domain features for HbR
    features.append(np.mean(hbr_signal))  # Mean HbR concentration
    features.append(np.std(hbr_signal))   # HbR variability
    features.append(np.max(hbr_signal) - np.min(hbr_signal))  # HbR range
    '''
    # HbO-HbR correlation features
    features.append(np.corrcoef(hbo_segment, hbr_segment)[0, 1])  # Correlation b/t HbO and HbR
    
    # Frequency-domain features (using Welch's method)
    for signal in [hbo_segment, hbr_segment]:
        freqs, psd = welch(signal, fs=fs, nperseg=min(64, len(signal)//2))
        
        # Very low frequency (0.01-0.1 Hz)
        vlf = np.trapz(psd[(freqs >= 0.01) & (freqs < 0.1)])
        
        # Low frequency (0.1-0.2 Hz)
        lf = np.trapz(psd[(freqs >= 0.1) & (freqs < 0.2)])
        
        # High frequency (0.2-0.5 Hz)
        #hf = np.trapz(psd[(freqs >= 0.2) & (freqs < 0.5)])
        
        #features.extend([vlf, lf, hf, lf/hf if hf > 1e-6 else 0])
        features.extend([vlf, lf])
    
    return np.array(features)

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

def train_svm_model(features, labels, k_folds=5):
    """
    Train SVM with RBF kernel using k-fold CV
    """
    # Create pipeline (SVM)
    svm_pipe = make_pipeline(
        PCA(n_components=9),
        SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    )
    
    # Stratified k-fold CV
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=22)
    cv_scores = cross_val_score(svm_pipe, features, labels, cv=skf)
    
    print(f"\nSVM {k_folds}-Fold CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
    
    # Train final model
    svm_pipe.fit(features, labels)
    return svm_pipe, np.mean(cv_scores)

def evaluate_quadrant_pairs_fnirs(features, labels, model):
    """
    Evaluate fNIRS model performance on all quadrant pairs with hemodynamic-aware validation
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

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\rsril\OneDrive\Desktop\neuro_analysis\Subjects"
    subjects = range(16)  # Subjects 0-15
    
    # 1. Load data and extract features
    print("Loading fNIRS data and extracting features...")
    baseline_data, emotional_data, quadrants = load_fnirs_data(subjects, DATA_DIR)
    features = normalize_fnirs_features(baseline_data,emotional_data)

    # Print feature information
    print(f"\nFeature matrix shape: {features.shape}")
    print("Feature types: HbO/HbR time-domain + frequency-domain features")
    
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
    
    # Full dataset performance
    # print("\nOverall Performance:")
    # metrics = evaluate_model(model, features, quadrants, model_name)
    
    # Quadrant-pair performance
    print("\nQuadrant-Pair Performance:")
    pair_results = evaluate_quadrant_pairs_fnirs(features, quadrants, model)
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
        
        # Print neurophysiological interpretation
        print("\nNeurophysiological Interpretation:")
        print("- High accuracy pairs may reflect distinct cortical activation patterns")
        print("- target_quadrant (Amused) vs neutral_quadrant (Frustrated): Compare prefrontal cortex activity")
        print("- Q3 (Gloomy) vs Q4 (Calm): Compare default mode network involvement")
        print("- HbO-HbR decoupling often indicates strong neural activation")
    else:
        print("No valid quadrant pairs could be evaluated")