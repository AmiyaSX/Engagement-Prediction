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

# === CONFIGURATION ===
WINDOW_SIZE = 12
STEP_SIZE = 5
EMBEDDING_DIM = 3072
BUFFER = 30

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


def load_transcripts(dir_path):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception as e:
            continue
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            continue
        label = label_row["EngagementLevel"].values[0]
        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            continue
        df["subject_id"] = subject_id
        df["start_sec"] = df["start_time"]
        df["end_sec"] = df["end_time"]
        df["file"] = filename
        df["eng_level"] = label
        cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                embeddings = json.load(f)
            if len(embeddings) != len(df):
                continue
            df["embedding"] = embeddings
        else:
            continue
        all_data.append(df)
    if not all_data:
        raise ValueError("No valid transcript data was loaded.")
    return pd.concat(all_data, ignore_index=True)


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
                    "win_start": int(start_t * 10),
                    "win_end": int(end_t * 10),
                    "subject_id": subject_id,
                    "run_id": run_id,
                }
            )
    return np.stack(X_gaze), pd.DataFrame(metadata)


def main():
    print("Loading transcript embeddings...")
    participant_df = load_transcripts("data/raw/transcripts/participant")
    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    matched_keys = set(participant_df["match_key"])

    print("Loading gaze data...")
    X_gaze_raw, meta_gaze = load_gaze_data()
    gaze_lookup = {
        (row["key"], row["win_start"], row["win_end"]): idx
        for idx, row in meta_gaze.iterrows()
    }

    print("Processing text windows...")
    X_text, y_text, text_metadata = [], [], {}
    for key in tqdm(matched_keys):
        part = participant_df[participant_df["match_key"] == key]
        max_time = part["end_sec"].max()
        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            buffer_start = max(0, win_start - BUFFER)
            part_win = part[
                (part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)
            ]
            if part_win.empty:
                continue
            try:
                part_emb = np.vstack(part_win["embedding"].values)
                part_avg = np.mean(part_emb, axis=0)
            except:
                part_avg = np.zeros(EMBEDDING_DIM)
            X_text.append(part_avg)
            y_text.append(part_win["eng_level"].iloc[0])
            win_key = (key, int(win_start * 10), int(win_end * 10))
            text_metadata[win_key] = len(X_text) - 1

    print("Combining gaze and embeddings...")
    X_all, y_all, groups_all = [], [], []
    for i, row in meta_gaze.iterrows():
        win_key = (row["key"], row["win_start"], row["win_end"])
        text_idx = text_metadata.get(win_key)
        if text_idx is None:
            continue
        gaze_win = X_gaze_raw[i]
        text_feat = X_text[text_idx]
        X_all.append(np.concatenate([gaze_win.flatten(), text_feat]))
        y_all.append(y_text[text_idx])
        groups_all.append(row["subject_id"])

    X_all = StandardScaler().fit_transform(np.stack(X_all))
    y_all = np.array(y_all)
    groups_all = np.array(groups_all)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

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
        logo.split(X_all, y_encoded, groups_all), 1
    ):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
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
