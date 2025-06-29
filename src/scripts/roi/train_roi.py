import pandas as pd
import numpy as np
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft

# Constants
WINDOW_SIZE = 10  # 10 samples, 12s
STEP_SIZE = 5  # 5 samples, 6s
data_path = "data/raw/brain/train/*_with_timestamps.csv"
subjects = [5, 7, 8, 11, 12, 13, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32]

roi_columns = [
    "comp_vs_prod_combined",
    "dmn_medial_tpj",
    "prod_vs_comp_combined",
]

# Hyperparameter grid for XGBoost
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 6],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
}

# Load mapping
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()

mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map(
        {
            "low": "low",
            "normal": "normal",
            "high": "high",
        }
    )
)


def extract_features_from_window(window):
    """
    Enhanced feature extraction from a time-windowed ROI signal.
    Applies statistical, frequency, and complexity features across ROIs.
    """
    features = []
    num_rois = window.shape[0]

    for roi_signal in window:
        # Basic statistics
        mean_val = np.mean(roi_signal)
        std_val = np.std(roi_signal)
        min_val = np.min(roi_signal)
        max_val = np.max(roi_signal)
        range_val = np.ptp(roi_signal)
        skew_val = skew(roi_signal)
        kurt_val = kurtosis(roi_signal)

        # Derivatives
        first_diff = np.diff(roi_signal)
        second_diff = np.diff(first_diff)
        slope, intercept = np.polyfit(range(len(roi_signal)), roi_signal, 1)

        # Signal energy & entropy
        energy = np.sum(roi_signal**2)
        hist, _ = np.histogram(roi_signal, bins=10)
        ent = entropy(hist + 1)  # +1 to avoid log(0)

        # Zero crossing rate
        zcr = ((roi_signal[:-1] * roi_signal[1:]) < 0).sum()

        # Autocorrelation with lag 1
        autocorr = np.corrcoef(roi_signal[:-1], roi_signal[1:])[0, 1]

        # FFT features
        fft_coeffs = np.abs(fft(roi_signal))
        fft_mean = np.mean(fft_coeffs)
        fft_std = np.std(fft_coeffs)
        fft_power = np.sum(fft_coeffs**2)
        peak_freq = np.argmax(fft_coeffs[1:]) + 1  # Ignore DC component

        # Hjorth parameters
        var_1 = np.var(first_diff)
        var_2 = np.var(second_diff)
        var_0 = np.var(roi_signal)
        mobility = np.sqrt(var_1 / var_0) if var_0 != 0 else 0
        complexity = (
            np.sqrt(var_2 / var_1) / mobility if var_1 != 0 and mobility != 0 else 0
        )

        # Slope change count
        slope_changes = np.sum(np.diff(np.sign(first_diff)) != 0)

        # Append all features for this ROI
        features.extend(
            [
                mean_val,
                std_val,
                min_val,
                max_val,
                range_val,
                skew_val,
                kurt_val,
                np.mean(first_diff),
                np.mean(second_diff),
                slope,
                intercept,
                energy,
                ent,
                zcr,
                autocorr,
                fft_mean,
                fft_std,
                fft_power,
                peak_freq / len(fft_coeffs),
                mobility,
                complexity,
                slope_changes,
            ]
        )

    # Cross-ROI correlations
    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            corr = np.corrcoef(window[i], window[j])[0, 1]
            features.append(corr)

    # Lagged cross-ROI correlations
    for lag in [1, 2, 3]:
        for i in range(num_rois):
            for j in range(i + 1, num_rois):
                try:
                    lagged_corr = np.corrcoef(window[i][:-lag], window[j][lag:])[0, 1]
                except:
                    lagged_corr = 0
                features.append(lagged_corr)

    return features


def load_roi_data(window_size=10, step_size=4):
    X, y, groups = [], [], []
    metadata = []
    file_paths = glob.glob(data_path)

    for file in file_paths:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        filename = os.path.basename(file)

        # Extract subject ID from filename: "sub-05_roi_data_with_timestamps.csv"
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
        except (IndexError, ValueError):
            print(f"Could not extract subject_id from filename: {filename}")
            continue

        if "run_index" not in df.columns:
            print(f"No 'run_index' column in {filename}")
            continue

        # Normalize ROI features per subject (enhanced more accuracy)
        df[roi_columns] = (df[roi_columns] - df[roi_columns].mean()) / df[
            roi_columns
        ].std()

        # Group by run_index
        for run_id_str, group in df.groupby("run_index"):
            try:
                run_id = int(run_id_str.strip().split("-")[1])
            except (IndexError, ValueError):
                print(f"Could not parse run_id from value '{run_id_str}' in {filename}")
                continue

            # Lookup engagement level from mapping
            label_row = mapping_df[
                (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
            ]
            if label_row.empty:
                print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
                continue

            label = label_row["EngagementLevel"].values[0]

            # Extract ROI features
            features = group[roi_columns].to_numpy().T  # shape: (3, n_samples)

            timestamps = group["timestamp"].to_numpy()

            for i in range(0, features.shape[1] - window_size + 1, step_size):
                start_time = timestamps[i]
                end_time = timestamps[i + window_size - 1]
                window = features[:, i : i + window_size]
                feature_set = extract_features_from_window(window)
                X.append(feature_set)
                y.append(label)
                groups.append(subject_id)

                metadata.append(
                    {
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": label,
                    }
                )

    if not X:
        raise ValueError("No valid data was loaded. Check input files and mappings.")

    X = np.stack(X)  # shape: (n_samples, features)
    y = np.array(y)
    groups = np.array(groups)
    metadata_df = pd.DataFrame(metadata)
    return X, y, groups, metadata_df


X, y, groups, metadata_df = load_roi_data()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Encode string labels to integers
y_encoded = le.fit_transform(y)

pipeline = Pipeline(
    steps=[
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=len(le.classes_),
                eval_metric="mlogloss",
                subsample=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ],
    memory=joblib.Memory(location="cache_dir", verbose=0),
)

# Leave-One-Group-Out Cross Validation
logo = LeaveOneGroupOut()

# Save metadata
metadata_df.to_csv("window_metadata.csv", index=False)
print(f"\nSaved metadata for {len(metadata_df)} windows to 'window_metadata.csv'")

accuracies = []
all_y_true = []
all_y_pred = []

print("\nRunning Leave-One-Group-Out Cross-Validation:\n")

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(
        f"Fold {fold + 1} | Test Group: {np.unique(groups[test_idx])} | Accuracy: {acc:.4f}"
    )
    roi_stats = [
    "mean", "std", "min", "max", "range", "skew", "kurtosis",
    "diff_mean", "diff2_mean", "slope", "intercept",
    "energy", "entropy", "zcr", "autocorr",
    "fft_mean", "fft_std", "fft_power", "peak_freq",
    "mobility", "complexity", "slope_changes"
    ]
    roi_labels = roi_columns
    feature_names = []

    # ROI-specific features
    for roi in roi_labels:
        for stat in roi_stats:
            feature_names.append(f"{roi}_{stat}")

    # Cross-ROI correlations
    for i in range(len(roi_labels)):
        for j in range(i + 1, len(roi_labels)):
            feature_names.append(f"corr_{roi_labels[i]}_vs_{roi_labels[j]}")

    # Lagged cross-ROI correlations
    for lag in [1, 2, 3]:
        for i in range(len(roi_labels)):
            for j in range(i + 1, len(roi_labels)):
                feature_names.append(f"lag{lag}_corr_{roi_labels[i]}_vs_{roi_labels[j]}")

    # Sanity check
    assert len(feature_names) == X.shape[1], "Feature name count mismatch"

    # === Feature Importance ===

    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    # Save to CSV
    feature_importance_df.to_csv(
        f"roi_feature_importance_sub-{np.unique(groups[test_idx])}.csv", index=False
    )


# Overall results
print("\n=== LOGO CV Summary ===")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Std Accuracy: {np.std(accuracies):.4f}")
print("\nClassification Report (Aggregated):")
print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

print("\nConfusion Matrix (Aggregated):")
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
print(
    pd.DataFrame(
        conf_matrix,
        index=[f"Actual {label}" for label in le.classes_],
        columns=[f"Predicted {label}" for label in le.classes_],
    )
)
