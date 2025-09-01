# Early Fusion of Gaze and Pause Features for Engagement Prediction

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

# === CONFIGURATION ===
WINDOW_SIZE = 12
STEP_SIZE = 5

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


def load_pause_features(file_path, window_size=12, step_size=5):
    pause_df = pd.read_csv(file_path)
    pause_df["subject"] = pause_df["filename"].apply(
        lambda x: int(x.split("_")[0].split("-")[1])
    )
    pause_df["run"] = pause_df["filename"].apply(
        lambda x: int(x.split("_")[1].split("-")[1])
    )

    pause_features = {}
    event_types = ["pause_p", "pause_c", "gap_p2c", "gap_c2p"]
    grouped = pause_df.groupby(["subject", "run"])

    for (subject, run), group in grouped:
        max_time = group["start_time"].max() + group["duration"].max()
        for start in np.arange(0, max_time - window_size + 1, step_size):
            end = start + window_size
            window = group[(group["start_time"] >= start) & (group["start_time"] < end)]
            if window.empty:
                continue
            feat = {}
            for ev in event_types:
                ev_data = window[window["event_type"] == ev]
                durations = ev_data["duration"].values
                feat[f"count_{ev}"] = len(durations)
                feat[f"duration_{ev}"] = durations.sum()
                feat[f"rate_{ev}"] = len(durations) / window_size
                feat[f"proportion_time_{ev}"] = durations.sum() / window_size
                feat[f"mean_duration_{ev}"] = durations.mean() if len(durations) else 0
                feat[f"std_duration_{ev}"] = durations.std() if len(durations) else 0
            key = (
                f"sub-{subject:02d}_run-{run:02d}",
                round(start, 1),
                round(end, 1),
            )
            pause_features[key] = np.array(list(feat.values()))
    return pause_features


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
            X_gaze.append(win.to_numpy())
            metadata.append(
                {
                    "key": f"sub-{subject_id:02d}_run-{run_id:02d}",
                    "win_start": round(start_t, 1),
                    "win_end": round(end_t, 1),
                    "subject_id": subject_id,
                    "run_id": run_id,
                }
            )
    return np.stack(X_gaze), pd.DataFrame(metadata)


def main():
    print("Loading gaze data...")
    X_gaze_raw, meta_gaze = load_gaze_data()

    print("Loading pause features...")
    pause_feature_dict = load_pause_features("data/raw/pause/prod_comp_gaps_pauses.csv")

    print("Combining gaze and pause features...")
    X_all, y_all, groups_all = [], [], []
    for i, row in meta_gaze.iterrows():
        win_key = (
            row["key"],
            round(row["win_start"], 1),
            round(row["win_end"], 1),
        )
        pause_feat = pause_feature_dict.get(win_key)
        if pause_feat is None:
            continue

        subject_id = row["subject_id"]
        run_id = row["run_id"]
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            continue
        label = label_row["EngagementLevel"].values[0]

        gaze_win = X_gaze_raw[i]
        fused = np.concatenate([gaze_win.flatten(), pause_feat])
        X_all.append(fused)
        y_all.append(label)
        groups_all.append(subject_id)

    print(f"Number of combined samples: {len(X_all)}")
    X_all = StandardScaler().fit_transform(np.stack(X_all))
    y_all = np.array(y_all)
    groups_all = np.array(groups_all)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    pipeline = Pipeline(
        [
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
            )
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
        pipeline.fit(
            X_train,
            y_train,
            clf__sample_weight=compute_sample_weight("balanced", y_train),
        )
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"Fold {fold} | Subject {groups_all[test_idx[0]]} | Accuracy: {acc:.4f}")

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accs):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
