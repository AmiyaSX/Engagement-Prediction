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
    pause_df["subject"] = pause_df["filename"].apply(lambda x: int(x.split("_")[0].split("-")[1]))
    pause_df["run"] = pause_df["filename"].apply(lambda x: int(x.split("_")[1].split("-")[1]))

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
                start_times = ev_data["start_time"].values
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
    """
    Loads transcript CSVs and attaches cached or newly generated Gemini embeddings.
    Returns a single concatenated DataFrame.
    """
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception as e:
            print(f"[ERROR] Skipping file {filename}: {e}")
            continue

        # Get label
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"[WARNING] No label for subject {subject_id}, run {run_id}")
            continue

        label = label_row["EngagementLevel"].values[0]

        # Load transcript CSV
        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            print(f"[WARNING] Missing 'transcription' column in {filename}")
            continue

        df["subject_id"] = subject_id
        df["start_sec"] = df["start_time"]
        df["end_sec"] = df["end_time"]
        df["mid_time"] = (df["start_sec"] + df["end_sec"]) / 2
        df["file"] = filename
        df["eng_level"] = label

        # Load or generate embeddings
        cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                embeddings = json.load(f)
            if len(embeddings) != len(df):
                print(f"[WARNING] Embedding length mismatch in {filename} (expected {len(df)}, got {len(embeddings)})")
                continue
            df["embedding"] = embeddings
        else:
            print(f"[INFO] Generating embeddings for {filename}")
            df["embedding"] = df["transcription"].apply(get_embedding)
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid transcript data was loaded.")

    return pd.concat(all_data, ignore_index=True)

def load_brain_data(window_size=11, step_size=5):
    X, y, groups = [], [], []
    metadata = []
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
                X.append(features[:, i : i + window_size])
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

# Hybrid Late Fusion for Engagement Prediction
# Model A: ROI + Pauses | Model B: Text Embeddings

# === MAIN PROCESS ===
def main():
    participant_df = load_transcripts("data/raw/transcripts/participant")
    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    matched_keys = set(participant_df["match_key"])

    X_brain, y_brain, groups_brain, metadata_brain = load_brain_data()
    pause_feature_dict = load_pause_features("data/raw/pause/prod_comp_gaps_pauses.csv")

    X_text, y_text, text_metadata = [], [], {}

    for key in tqdm(matched_keys):
        part = participant_df[participant_df["match_key"] == key]
        max_time = part["end_sec"].max()
        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            buffer_start = max(0, win_start - BUFFER)
            part_win = part[(part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)]

            if part_win.empty:
                continue

            try:
                part_emb = np.vstack(part_win["embedding"].values)
                part_avg = np.mean(part_emb, axis=0)
            except:
                part_avg = np.zeros(EMBEDDING_DIM)

            X_text.append(part_avg)
            y_text.append(part_win["eng_level"].iloc[0])
            win_key = (key, round(win_start, 1), round(win_end, 1))
            text_metadata[win_key] = len(X_text) - 1

    X_text = np.stack(X_text)
    y_text = np.array(y_text)

    X_a, X_b, y_combined, groups_combined = [], [], [], []

    for i, row in metadata_brain.iterrows():
        win_key = (
            f"sub-{row['subject_id']:02d}_run-{row['run_id']:02d}",
            round(row["win_start"], 1),
            round(row["win_end"], 1),
        )
        text_idx = text_metadata.get(win_key, None)
        pause_feat = pause_feature_dict.get(win_key, None)
        if text_idx is None or pause_feat is None:
            continue

        roi_feat = X_brain[i].reshape(-1)
        text_feat = X_text[text_idx]

        X_a.append(np.concatenate([roi_feat, pause_feat]))
        X_b.append(text_feat)
        y_combined.append(y_text[text_idx])
        groups_combined.append(row["subject_id"])

    X_a = StandardScaler().fit_transform(np.stack(X_a))
    X_b = StandardScaler().fit_transform(np.stack(X_b))
    y_combined = np.array(y_combined)
    groups_combined = np.array(groups_combined)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

    pipeline_a = Pipeline([
        ("classifier", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     objective="multi:softprob", num_class=len(le.classes_),
                                     eval_metric="mlogloss", random_state=42, n_jobs=4))])

    pipeline_b = Pipeline([
        ("classifier", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     objective="multi:softprob", num_class=len(le.classes_),
                                     eval_metric="mlogloss", random_state=42, n_jobs=4))])

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_a, y_encoded, groups_combined), 1):
        X_a_train, X_a_test = X_a[train_idx], X_a[test_idx]
        X_b_train, X_b_test = X_b[train_idx], X_b[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        pipeline_a.fit(X_a_train, y_train, classifier__sample_weight=compute_sample_weight("balanced", y_train))
        pipeline_b.fit(X_b_train, y_train, classifier__sample_weight=compute_sample_weight("balanced", y_train))

        prob_a = pipeline_a.predict_proba(X_a_test)
        prob_b = pipeline_b.predict_proba(X_b_test)

        fused_prob = 0.8 * prob_a + 0.2 * prob_b
        y_pred = np.argmax(fused_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"\nFold {fold} | Accuracy: {acc:.4f}")

    print("\n=== Final Results (Late Fusion) ===")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
