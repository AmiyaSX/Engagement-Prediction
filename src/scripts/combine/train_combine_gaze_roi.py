import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sktime.transformations.panel.rocket import MiniRocket
import google.generativeai as genai
from google.api_core import retry

# === CONFIGURATION ===
WINDOW_SIZE = 12
STEP_SIZE = 6
EMBEDDING_DIM = 3072


mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)

def extract_id(file_name):
    parts = file_name.split("_")
    subject = parts[0]
    run = parts[2].split("-")[1].split(".")[0]
    return f"{subject}_run-{run}"


def load_brain_data(window_size=11, step_size=5):
    X, y, groups, metadata = [], [], [], []
    roi_cols = ["comp_vs_prod_combined", "dmn_medial_tpj", "prod_vs_comp_combined"]
    for file in glob.glob("data/raw/brain/train/*_with_timestamps.csv"):
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        subject_id = int(filename.split("_")[0].split("-")[1])
        df[roi_cols] = (df[roi_cols] - df[roi_cols].mean()) / df[roi_cols].std()

        for run_id_str, group in df.groupby("run_index"):
            run_id = int(run_id_str.split("-")[1])
            label_row = mapping_df[
                (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
            ]
            if label_row.empty:
                continue
            label = label_row["EngagementLevel"].values[0]
            features = group[roi_cols].to_numpy().T
            timestamps = group["timestamp"].to_numpy()
            for i in range(0, features.shape[1] - window_size + 1, step_size):
                X.append(features[:, i : i + window_size].T)  # Transposed here
                y.append(label)
                groups.append(subject_id)
                metadata.append(
                    {
                        "key": f"sub-{subject_id}_run-{run_id}",
                        "win_start": timestamps[i],
                        "win_end": timestamps[i + window_size - 1],
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "label": label,
                    }
                )
    return np.stack(X), np.array(y), np.array(groups), pd.DataFrame(metadata)


def load_gaze_data(window_size=11, step_size=5):
    X_gaze, metadata = [], []
    for file in glob.glob("data/raw/eyetracking/feature/*.csv"):
        df = pd.read_csv(file)
        if "Timestamp" not in df.columns:
            continue
        df = df.sort_values("Timestamp").dropna()
        subject_id = int(file.split("_")[0].split("-")[1])
        run_id = int(file.split("_")[1].split("-")[1])
        timestamps = df["Timestamp"].to_numpy()
        for i in range(0, len(df) - window_size + 1, step_size):
            win = df.iloc[i : i + window_size]
            start_t, end_t = timestamps[i], timestamps[i + window_size - 1]
            X_gaze.append(win.to_numpy())  # Already (window_size, features)
            metadata.append(
                {
                    "key": f"sub-{subject_id:02d}_run-{run_id:02d}",
                    "win_start": int(start_t * 10),
                    "win_end": int(end_t * 10),
                    "subject_id": subject_id,
                    "run_id": run_id,
                }
            )
    return np.stack(X_gaze), pd.DataFrame(metadata)


def main():
    print("Loading brain data...")
    X_brain, y_brain, groups_brain, meta_brain = load_brain_data()

    print("Loading gaze data...")
    X_gaze_raw, meta_gaze = load_gaze_data()
    gaze_lookup = {
        (row["key"], row["win_start"], row["win_end"]): idx
        for idx, row in meta_gaze.iterrows()
    }

    print("Combining modalities...")
    X_all, y_all, groups_all = [], [], []
    for i, row in meta_brain.iterrows():
        win_key = (
            f"sub-{row['subject_id']:02d}_run-{row['run_id']:02d}",
            int(row["win_start"] * 10),
            int(row["win_end"] * 10),
        )
        gaze_idx = gaze_lookup.get(win_key)
        if gaze_idx is None:
            continue

        # Concatenate along feature axis, then flatten
        brain_win = X_brain[i]
        gaze_win = X_gaze_raw[gaze_idx]

        if brain_win.shape[0] != gaze_win.shape[0]:
            continue  # skip mismatched windows

        combined = np.concatenate(
            [brain_win, gaze_win], axis=1
        )  # shape: (time, features)
        X_all.append(combined.flatten())  # flatten to 1D for MiniRocket
        y_all.append(y_brain[i])
        groups_all.append(row["subject_id"])

    print(f"Number of combined samples: {len(X_all)}")
    X_all, y_all, groups_all = np.stack(X_all), np.array(y_all), np.array(groups_all)
    X_all = StandardScaler().fit_transform(X_all)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    pipeline = Pipeline(
        [
            ("transform", MiniRocket(random_state=42)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softmax",
                    num_class=len(le.classes_),
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )

    print("Running Leave-One-Group-Out CV...")
    logo = LeaveOneGroupOut()
    accs, all_y_true, all_y_pred = [], [], []
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_all, y_enc, groups_all), 1
    ):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"Fold {fold} | Subject {groups_all[test_idx[0]]} | Acc: {acc:.4f}")

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accs):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
