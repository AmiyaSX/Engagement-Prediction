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
import google.generativeai as genai
from google.api_core import retry

# === CONFIGURATION ===
WINDOW_SIZE = 12
STEP_SIZE = 5
EMBEDDING_DIM = 3072
BUFFER = 30

# Set your Gemini API key
genai.configure(api_key="My key")

mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)


# === HELPER FUNCTIONS ===
@retry.Retry(timeout=300.0)
def get_embedding(text):
    result = genai.embed_content(
        model="gemini-embedding-exp-03-07", content=text, task_type="classification"
    )
    return result["embedding"]


def extract_id(file_name):
    parts = file_name.split("_")
    subject = parts[0]  # sub-01
    run = parts[2].split("-")[1].split(".")[0]  # 01
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


def load_transcripts(dir_path):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception as e:
            print(f"[ERROR] Skipping file {filename}: {e}")
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
        df["mid_time"] = (df["start_sec"] + df["end_sec"]) / 2
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
            df["embedding"] = df["transcription"].apply(get_embedding)
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# === MAIN PROCESS ===
def main():
    participant_df = load_transcripts("data/raw/transcripts/participant")
    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    matched_keys = set(participant_df["match_key"])

    pause_feature_dict = load_pause_features("data/raw/pause/prod_comp_gaps_pauses.csv")

    X_fused, y_fused, groups = [], [], []
    text_metadata = {}

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

            pause_key = (key, round(win_start, 1), round(win_end, 1))
            pause_feat = pause_feature_dict.get(pause_key, None)
            if pause_feat is None:
                continue

            fused_feat = np.concatenate([part_avg, pause_feat])
            X_fused.append(fused_feat)
            y_fused.append(part_win["eng_level"].iloc[0])
            groups.append(part_win["subject_id"].iloc[0])

    X_fused = StandardScaler().fit_transform(np.stack(X_fused))
    y_fused = LabelEncoder().fit_transform(np.array(y_fused))
    groups = np.array(groups)

    pipeline = Pipeline(
        [
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softmax",
                    eval_metric="mlogloss",
                    num_class=len(np.unique(y_fused)),
                    random_state=42,
                    n_jobs=4,
                ),
            )
        ]
    )

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_fused, y_fused, groups), 1
    ):
        X_train, X_test = X_fused[train_idx], X_fused[test_idx]
        y_train, y_test = y_fused[train_idx], y_fused[test_idx]

        sample_weights = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"\nFold {fold} | Accuracy: {acc:.4f}")

    print("\n=== Final Results (Early Fusion: Pauses + Embeddings) ===")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred))


if __name__ == "__main__":
    main()
