import os
import glob
import json
import re
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

MAPPING_CSV = "data/mapping/conditions.csv"
mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)


# -------------------------
# EMBEDDINGS
# -------------------------
@retry.Retry(timeout=300.0)
def get_embedding(text: str):
    result = genai.embed_content(
        model="gemini-embedding-exp-03-07",
        content=text,
        task_type="classification",
    )
    return result["embedding"]


def parse_subject_run(filename: str):
    """
    Robustly extract subject_id and run_id from filenames containing 'sub-XX' and 'run-YY'.
    Example: sub-01_something_run-02.csv -> (1, 2)
    """
    subj_m = re.search(r"sub-(\d+)", filename)
    run_m = re.search(r"run-(\d+)", filename)
    if not subj_m or not run_m:
        raise ValueError(f"Cannot parse subject/run from filename: {filename}")
    return int(subj_m.group(1)), int(run_m.group(1))


def make_match_key(subject_id: int, run_id: int):
    return f"sub-{subject_id:02d}_run-{run_id:02d}"


def load_transcripts_with_embeddings(dir_path: str, speaker_tag: str):
    """
    Loads transcript CSVs + embedding cache JSON (or computes if missing),
    returns one dataframe with speaker column.
    """
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)

        try:
            subject_id, run_id = parse_subject_run(filename)
        except Exception as e:
            print(f"[ERROR] Skipping {filename}: {e}")
            continue

        # label from mapping
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
        df["run_id"] = run_id
        df["start_sec"] = df["start_time"]
        df["end_sec"] = df["end_time"]
        df["file"] = filename
        df["eng_level"] = label
        df["speaker"] = speaker_tag
        df["match_key"] = make_match_key(subject_id, run_id)

        # cache path (separate per file)
        cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                embeddings = json.load(f)
            if len(embeddings) != len(df):
                print(f"[WARN] Embedding count mismatch for {filename}; skipping.")
                continue
            df["embedding"] = embeddings
        else:
            df["embedding"] = df["transcription"].fillna("").apply(get_embedding)
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# -------------------------
# PAUSE FEATURES
# -------------------------
def load_pause_features(file_path: str, window_size=12, step_size=5):
    pause_df = pd.read_csv(file_path)

    pause_df["subject"] = pause_df["filename"].apply(
        lambda x: int(x.split("_")[0].split("-")[1])
    )
    pause_df["run"] = pause_df["filename"].apply(
        lambda x: int(x.split("_")[1].split("-")[1])
    )

    pause_features = {}

    event_types = [
        "production",
        "comprehension",
        "pause_p",
        "pause_c",
        "gap_p2c",
        "gap_c2p",
    ]

    for (subject, run), group in pause_df.groupby(["subject", "run"]):
        max_time = float(group["start_time"].max() + group["duration"].max())
        key_id = make_match_key(subject, run)

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
                feat[f"duration_{ev}"] = float(durations.sum())
                feat[f"rate_{ev}"] = len(durations) / window_size
                feat[f"proportion_time_{ev}"] = float(durations.sum()) / window_size
                feat[f"mean_duration_{ev}"] = (
                    float(durations.mean()) if len(durations) else 0.0
                )
                feat[f"std_duration_{ev}"] = (
                    float(durations.std()) if len(durations) else 0.0
                )

            pause_key = (key_id, round(start, 1), round(end, 1))
            pause_features[pause_key] = np.array(list(feat.values()), dtype=float)

    return pause_features


def avg_embedding(win_df: pd.DataFrame):
    if win_df.empty:
        return np.zeros(EMBEDDING_DIM, dtype=float)
    try:
        emb = np.vstack(win_df["embedding"].values).astype(float)
        return np.mean(emb, axis=0)
    except Exception:
        return np.zeros(EMBEDDING_DIM, dtype=float)


# -------------------------
# MAIN
# -------------------------
def main():
    # 1) Load transcripts (participant + operator)
    participant_df = load_transcripts_with_embeddings(
        "data/raw/transcripts/participant", "participant"
    )
    operator_df = load_transcripts_with_embeddings(
        "data/raw/transcripts/operator", "operator"
    )

    if participant_df.empty or operator_df.empty:
        raise RuntimeError("Missing transcript data (participant/operator).")

    # Require both present for 'both embeddings'
    matched_keys = set(participant_df["match_key"]).intersection(
        set(operator_df["match_key"])
    )
    if not matched_keys:
        raise RuntimeError(
            "No overlapping (subject,run) keys between participant and operator transcripts."
        )

    # 2) Load pause features
    pause_feature_dict = load_pause_features(
        "data/raw/pause/prod_comp_gaps_pauses.csv", WINDOW_SIZE, STEP_SIZE
    )
    if not pause_feature_dict:
        raise RuntimeError("No pause features found.")

    X_fused, y_fused, groups = [], [], []

    for key in tqdm(sorted(matched_keys), desc="Building fused windows"):
        part = participant_df[participant_df["match_key"] == key]
        op = operator_df[operator_df["match_key"] == key]

        max_time = float(max(part["end_sec"].max(), op["end_sec"].max()))

        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            buffer_start = max(0, win_start - BUFFER)

            part_win = part[
                (part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)
            ]
            op_win = op[
                (op["start_sec"] >= buffer_start) & (op["start_sec"] <= win_end)
            ]

            if part_win.empty and op_win.empty:
                continue

            pause_key = (key, round(win_start, 1), round(win_end, 1))
            pause_feat = pause_feature_dict.get(pause_key, None)
            if pause_feat is None:
                continue

            part_avg = avg_embedding(part_win)
            op_avg = avg_embedding(op_win)

            fused_feat = np.concatenate([part_avg, op_avg, pause_feat], axis=0)

            # label: same for both; take from whichever non-empty
            label = (
                part_win["eng_level"].iloc[0]
                if not part_win.empty
                else op_win["eng_level"].iloc[0]
            )
            subj_id = (
                part_win["subject_id"].iloc[0]
                if not part_win.empty
                else op_win["subject_id"].iloc[0]
            )

            X_fused.append(fused_feat)
            y_fused.append(label)
            groups.append(subj_id)

    if len(X_fused) == 0:
        raise RuntimeError(
            "No fused samples created. Check windowing/key alignment and pause availability."
        )

    X_fused = np.stack(X_fused)
    y_fused = np.array(y_fused)
    groups = np.array(groups)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_fused)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softmax",
                    eval_metric="mlogloss",
                    num_class=len(le.classes_),
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

    print("\n=== Training & Evaluating (LOGO CV) ===")
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_fused, y_enc, groups), 1):
        X_train, X_test = X_fused[train_idx], X_fused[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        sample_weights = compute_sample_weight("balanced", y_train)

        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(
            f"Fold {fold:02d} | Subject(s): {np.unique(groups[test_idx])} | Acc: {acc:.4f}"
        )

    print(
        "\n=== Final Results (Early Fusion: Pauses + Participant+Operator Embeddings) ==="
    )
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (± {np.std(accuracies):.4f})")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
