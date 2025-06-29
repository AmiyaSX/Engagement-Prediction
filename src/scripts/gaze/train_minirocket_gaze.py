import pandas as pd
import numpy as np
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


# Constants
WINDOW_SIZE = 10  # samples, 12s
STEP_SIZE = 5  # samples, 6s

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
logo = LeaveOneGroupOut()
accuracies = []
all_y_true, all_y_pred = [], []
X_all, y_all, groups_all = [], [], []

def load_and_prepare_data(window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X, y, groups = [], [], []
    file_paths = glob.glob(f"{feature_data_path}/*_tracking_resampled.csv")
    files = glob.glob(f"{gaze_data_path}/*_tracking_resampled.csv")
    files_dict = {os.path.basename(f): f for f in files}

    for file in file_paths:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        filename = os.path.basename(file)

        df_gaze = pd.read_csv(files_dict[filename])
        df_gaze.columns = df_gaze.columns.str.strip()
        if "mutual_gaze_ratio" not in df_gaze.columns:
            print(f"'mutual_gaze_ratio' not found in {filename}, skipping.")
            continue
        if len(df) != len(df_gaze):
            print(f"Row count mismatch for {filename}: main={len(df)}, gaze={len(df_gaze)}")
            continue

        # Add/merge column by index
        # df["mutual_gaze_ratio"] = df_gaze["mutual_gaze_ratio"]

        subject_id = int(filename.split("_")[0].split("-")[1])  # '1'
        run_id = int(filename.split("_")[1].replace(".csv", "").split("-")[1])  # '1'

        # Get engagement label from mapping CSV
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
            continue

        label = label_row["EngagementLevel"].values[0]

        # Drop timestamp and apply sliding window
        df_feat = df.drop(columns=["Timestamp"])
        # df_feat = df.drop(columns=["mutual_gaze_ratio"])
        # df_feat = df.drop(columns=["Timestamp"])
        # df_feat = df.drop(columns=["Timestamp"])
        for i in range(0, len(df_feat) - WINDOW_SIZE + 1, STEP_SIZE):
            window = df_feat.iloc[i : i + WINDOW_SIZE]
            feature_vec = pd.concat(
                [
                    window.mean().add_suffix("_mean"),
                    window.std().add_suffix("_std"),
                    window.max().add_suffix("_max"),
                    window.min().add_suffix("_min"),
                ]
            )
            X_all.append(feature_vec.to_numpy())
            y_all.append(label)
            groups_all.append(subject_id)

        # Final arrays
    X = np.vstack(X_all)
    y = np.array(y_all)
    groups = np.array(groups_all)

    return X, y, groups

# Load data
X, y, groups = load_and_prepare_data()
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    pipeline = Pipeline(
        steps=[
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    objective="multi:softmax",
                    num_class=len(le.classes_),
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        memory=joblib.Memory(location="cache_dir", verbose=0),
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    print(f"\nFold {fold} | Test Subject: {groups[test_idx[0]]} | Accuracy: {acc:.4f}")

print("Leave-One-Group-Out CV Results:")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print("Classification Report:")
print(
classification_report(
        all_y_true, all_y_pred, target_names=le.classes_, zero_division=0
    )
)
