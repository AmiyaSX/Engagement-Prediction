# Combines ROI features + embeddings from both participant and operator for engagement prediction.

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

# Load and prepare mapping
genai.configure(api_key="My key")
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = mapping_df["eng_level"].str.lower().map({"low": "low", "normal": "normal", "high": "high"})

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

def load_transcripts(dir_path):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception:
            continue
        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            continue
        label_row = mapping_df[(mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)]
        if label_row.empty:
            continue
        label = label_row["EngagementLevel"].values[0]
        df["embedding"] = json.load(open(f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"))
        df["subject_id"] = subject_id
        df["run_id"] = run_id
        df["start_sec"] = df["start_time"]
        df["end_sec"] = df["end_time"]
        df["eng_level"] = label
        df["file"] = filename
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

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
            label_row = mapping_df[(mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)]
            if label_row.empty:
                continue
            label = label_row["EngagementLevel"].values[0]
            features = group[roi_cols].to_numpy().T
            timestamps = group["timestamp"].to_numpy()
            for i in range(0, features.shape[1] - window_size + 1, step_size):
                X.append(features[:, i : i + window_size])
                y.append(label)
                groups.append(subject_id)
                metadata.append({
                    "key": f"sub-{subject_id:02d}_run-{run_id:02d}",
                    "win_start": timestamps[i],
                    "win_end": timestamps[i + window_size - 1],
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "label": label,
                })
    return np.stack(X), np.array(y), np.array(groups), pd.DataFrame(metadata)

def main():
    print("Loading transcripts (participant and operator)...")
    participant_df = load_transcripts("data/raw/transcripts/participant")
    operator_df = load_transcripts("data/raw/transcripts/operator")
    if participant_df.empty or operator_df.empty:
        raise RuntimeError("Transcript data missing")

    print("Loading brain data...")
    X_brain, y_brain, groups_brain, metadata_brain = load_brain_data()
    metadata_brain["key"] = metadata_brain.apply(
        lambda row: f"sub-{row['subject_id']:02d}_run-{row['run_id']:02d}", axis=1
    )

    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    operator_df["match_key"] = operator_df["file"].apply(extract_id)
    matched_keys = set(participant_df["match_key"]).intersection(set(operator_df["match_key"]))

    X_combined, y_combined, groups_combined = [], [], []
    for key in tqdm(matched_keys):
        part = participant_df[participant_df["match_key"] == key]
        op = operator_df[operator_df["match_key"] == key]
        max_time = part["end_sec"].max()
        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            buffer_start = max(0, win_start - BUFFER)
            part_win = part[(part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)]
            op_win = op[(op["start_sec"] >= buffer_start) & (op["start_sec"] <= win_end)]
            if part_win.empty or op_win.empty:
                continue
            try:
                part_emb = np.mean(np.vstack(part_win["embedding"].values), axis=0)
                op_emb = np.mean(np.vstack(op_win["embedding"].values), axis=0)
            except Exception as e:
                print(f"[WARN] Embedding error at {key} {win_start}-{win_end}: {e}")
                continue
            combined_key = (key, round(win_start, 1), round(win_end, 1))
            roi_match = metadata_brain[
                (metadata_brain["key"] == key)
                & (np.round(metadata_brain["win_start"], 1) == round(win_start, 1))
                & (np.round(metadata_brain["win_end"], 1) == round(win_end, 1))
            ]
            if roi_match.empty:
                continue
            i = roi_match.index[0]
            roi_feat = X_brain[i].reshape(-1)
            combined_feat = np.concatenate([roi_feat, part_emb, op_emb])
            X_combined.append(combined_feat)
            y_combined.append(part_win["eng_level"].iloc[0])
            groups_combined.append(part_win["subject_id"].iloc[0])

    if not X_combined:
        raise RuntimeError("No valid combined samples found")

    print(f"[INFO] Final samples: {len(X_combined)}")
    X_combined = np.stack(X_combined)
    X_combined = StandardScaler().fit_transform(X_combined)
    y_combined = np.array(y_combined)
    groups_combined = np.array(groups_combined)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

    pipeline = Pipeline([
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
        ))
    ])

    logo = LeaveOneGroupOut()
    accs, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_combined, y_encoded, groups_combined), 1):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        w = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=w)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"Fold {fold} | Acc: {acc:.4f}")

    print("=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accs):.4f}  (± {np.std(accs):.4f})")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
