import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder
from sktime.transformations.panel.rocket import MiniRocket
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

# Constants
WINDOW_SIZE = 10  # samples
STEP_SIZE = 5  # samples

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

feature_data_path = "data/raw/eyetracking/feature"
gaze_data_path = "data/raw/eyetracking/feature/mutual_gaze"


def load_and_prepare_data_for_minirocket(window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X_list, y_list, group_list = [], [], []
    file_paths = glob.glob(f"{feature_data_path}/*_tracking_resampled.csv")
    files = glob.glob(f"{gaze_data_path}/*_tracking_resampled.csv")
    files_dict = {os.path.basename(f): f for f in files}

    for file in file_paths:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        filename = os.path.basename(file)

        if filename not in files_dict:
            print(f"Gaze file for {filename} not found, skipping.")
            continue

        df_gaze = pd.read_csv(files_dict[filename])
        df_gaze.columns = df_gaze.columns.str.strip()

        if "mutual_gaze_ratio" not in df_gaze.columns:
            print(f"'mutual_gaze_ratio' not found in {filename}, skipping.")
            continue
        if len(df) != len(df_gaze):
            print(
                f"Row count mismatch for {filename}: main={len(df)}, gaze={len(df_gaze)}"
            )
            continue

        subject_id = int(filename.split("_")[0].split("-")[1])
        run_id = int(filename.split("_")[1].replace(".csv", "").split("-")[1])

        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
            continue

        label = label_row["EngagementLevel"].values[0]

        df_feat = df.drop(columns=["Timestamp"])

        for i in range(0, len(df_feat) - window_size + 1, step_size):
            window = df_feat.iloc[i : i + window_size].reset_index(drop=True)
            X_list.append(window)
            y_list.append(label)
            group_list.append(subject_id)

    return X_list, y_list, group_list


# Load and prepare data
X_raw, y, groups = load_and_prepare_data_for_minirocket()
y = np.array(y)
groups = np.array(groups)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Initialize MiniRocket and classifier
logo = LeaveOneGroupOut()
accuracies = []
all_y_true, all_y_pred = [], []

print("Running Leave-One-Group-Out CV using MiniRocket...")

for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y_encoded, groups), 1):
    X_train_raw = [X_raw[i] for i in train_idx]
    X_test_raw = [X_raw[i] for i in test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # Convert to 3D numpy arrays (n_instances, n_channels, n_timepoints)
    X_train_3d = np.stack([x.to_numpy().T for x in X_train_raw])
    X_test_3d = np.stack([x.to_numpy().T for x in X_test_raw])

    # Convert to sktime nested format
    X_train_nested = from_3d_numpy_to_nested(X_train_3d)
    X_test_nested = from_3d_numpy_to_nested(X_test_3d)

    # Apply MiniRocket
    transformer = MiniRocket()
    transformer.fit(X_train_nested)
    X_train_transformed = transformer.transform(X_train_nested)
    X_test_transformed = transformer.transform(X_test_nested)

    # Train classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transformed, y_train)
    y_pred = classifier.predict(X_test_transformed)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    print(f"Fold {fold} | Test Subject: {groups[test_idx[0]]} | Accuracy: {acc:.4f}")

# Report
print("\nLeave-One-Group-Out CV Results:")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print("Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))
