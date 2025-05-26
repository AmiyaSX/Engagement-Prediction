import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
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

genai.configure(api_key="MY KEY")

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


def load_transcripts(dir_path):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        subject_id = int(filename.split("_")[0].split("-")[1])
        run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            continue
        label = label_row["EngagementLevel"].values[0]

        df = pd.read_csv(file_path)
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
                df["embedding"] = json.load(f)
        else:
            df["embedding"] = df["transcription"].apply(get_embedding)
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)
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


# === MAIN PROCESS ===
def main():
    print("Loading text transcript data...")
    participant_df = load_transcripts("data/raw/transcripts/participant")
    operator_df = load_transcripts("data/raw/transcripts/operator")

    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    operator_df["match_key"] = operator_df["file"].apply(extract_id)
    matched_keys = set(participant_df["match_key"]) & set(operator_df["match_key"])

    print(f"Found {len(matched_keys)} matched transcript pairs.")

    # Load brain data
    print("Loading brain ROI data...")
    X_brain, y_brain, groups_brain, metadata_brain = load_brain_data()
    print(f"Brain data shape: {X_brain.shape}")

    # Process transcript into text features
    X_text, y_text, time_marks_text, groups_text = [], [], [], []
    text_metadata = {}

    for key in tqdm(matched_keys):
        part = participant_df[participant_df["match_key"] == key]
        oper = operator_df[operator_df["match_key"] == key]

        max_time = max(part["end_sec"].max(), oper["end_sec"].max())
        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            part_win = part[
                (part["mid_time"] >= win_start) & (part["mid_time"] <= win_end)
            ]
            oper_win = oper[
                (oper["mid_time"] >= win_start) & (oper["mid_time"] <= win_end)
            ]

            if part_win.empty:
                continue

            try:
                part_emb = np.mean(np.vstack(part_win["embedding"].values), axis=0)
            except:
                part_emb = np.zeros(3072)

            try:
                oper_emb = np.mean(np.vstack(oper_win["embedding"].values), axis=0)
            except:
                oper_emb = np.zeros(3072)

            extra = np.array(
                [
                    len(part_win),
                    part_win["transcription"]
                    .apply(lambda t: len(str(t).split()))
                    .mean(),
                    part_win["transcription"].apply(lambda t: len(str(t).split())).sum()
                    / (part_win["end_sec"].sum() - part_win["start_sec"].sum() + 1e-6),
                ]
            )

            combined = np.concatenate([part_emb, oper_emb, extra])
            X_text.append(combined)
            y_text.append(part_win["eng_level"].iloc[0])
            time_marks_text.append((key, win_start, win_end))
            groups_text.append(part_win["subject_id"].iloc[0])
            # text_metadata[(key, round(win_start, 1), round(win_end, 1))] = (
            #     len(X_text) - 1
            # )
            text_metadata[(key, int(win_start * 10), int(win_end * 10))] = (
                len(X_text) - 1
            )

    X_text = np.stack(X_text)
    y_text = np.array(y_text)
    groups_text = np.array(groups_text)

    # Match windows between brain and text
    X_combined, y_combined, groups_combined = [], [], []
    for i, row in metadata_brain.iterrows():
        # win_key = (
        #     f"sub-{row['subject_id']:02d}_run-{row['run_id']:02d}",
        #     round(row["win_start"], 1),
        #     round(row["win_end"], 1),
        # )
        win_key = (
            f"sub-{row['subject_id']:02d}_run-{row['run_id']:02d}",
            int(row["win_start"] * 10),
            int(row["win_end"] * 10),
        )
        text_idx = text_metadata.get(win_key, None)
        if text_idx is None:
            # print(f"Skipping unmatched brain window: {win_key}")
            continue  # skip unmatched

        brain_feat = X_brain[i].reshape(-1)
        text_feat = X_text[text_idx]
        X_combined.append(np.concatenate([brain_feat, text_feat]))
        y_combined.append(y_text[text_idx])
        groups_combined.append(row["subject_id"])

    print(f"Combined samples: {len(X_combined)}")

    X_combined = np.stack(X_combined)
    y_combined = np.array(y_combined)
    groups_combined = np.array(groups_combined)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

    pipeline = Pipeline(
        steps=[
            # ("transform", MiniRocket(random_state=42)),
            (
                "classifier",
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
        ],
        memory=joblib.Memory(location="cache_dir", verbose=0),
    )

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_combined, y_encoded, groups_combined), 1
    ):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        sample_weights = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(
            f"\nFold {fold} | Test Subject: {groups_combined[test_idx[0]]} | Accuracy: {acc:.4f}"
        )

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
