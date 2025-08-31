import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.api_core import retry
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
import glob

WINDOW_SIZE = 12
STEP_SIZE = 5
STEP_SIZE = 5
BUFFER = 600

# Setup Gemini API client
genai.configure(api_key="My key")

# Load mapping
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)


def time_to_sec(t):
    h, m, s = map(float, t.split(":"))
    return h * 3600 + m * 60 + s


@retry.Retry(timeout=300.0)
def get_embedding(text):
    result = genai.embed_content(
        model="gemini-embedding-exp-03-07", content=text, task_type="classification"
    )
    return result["embedding"]


# Load and process one speaker's transcript files
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
            print(f"Missing label for subject {subject_id}, run {run_id}")
            continue
        label = label_row["EngagementLevel"].values[0]
        # print(f"Label for subject {subject_id}, run {run_id}: {label}")

        df = pd.read_csv(file_path)
        df["subject_id"] = subject_id
        df["start_sec"] = df["start_time"]
        df["end_sec"] = df["end_time"]
        df["mid_time"] = (df["start_sec"] + df["end_sec"]) / 2
        df["file"] = filename
        df["eng_level"] = label
        all_data.append(df)

        # Cache path for embeddings
        cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                df["embedding"] = json.load(f)
        else:
            print(f"Generating embeddings for {filename}")
            df["embedding"] = df["transcription"].apply(get_embedding)
            # Save embeddings as list
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def extract_id(file_name):
    parts = file_name.split("_")
    subject = parts[0]  # sub-01
    run = parts[2]  # run-01-transcriptions.csv
    run = run.split("-")[1].split(".")[0]  # 01
    return f"{subject}_run-{run}"


def main():

    # Load transcripts
    participant_df = load_transcripts("data/raw/transcripts/participant")
    operator_df = load_transcripts("data/raw/transcripts/operator")

    # Create normalized keys for matching
    participant_df["match_key"] = participant_df["file"].apply(extract_id)
    operator_df["match_key"] = operator_df["file"].apply(extract_id)

    all_keys = set(participant_df["match_key"]).intersection(
        set(operator_df["match_key"])
    )
    print(f"Found {len(all_keys)} matched transcript pairs.")

    X, y, time_marks, groups = [], [], [], []

    for key in tqdm(all_keys):
        part = participant_df[participant_df["match_key"] == key]
        oper = operator_df[operator_df["match_key"] == key]

        max_time = max(part["end_sec"].max(), oper["end_sec"].max())
        # max_time = part["end_sec"].max()

        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            # part_in_window = part[
            #     (part["mid_time"] >= win_start) & (part["mid_time"] <= win_end)
            # ]
            # oper_in_window = oper[
            #     (oper["mid_time"] >= win_start) & (oper["mid_time"] <= win_end)
            # ]
            buffer_start = max(0, win_start - BUFFER)
            part_in_window = part[
                (part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)
            ]
            oper_in_window = oper[
                (oper["start_sec"] >= buffer_start) & (oper["start_sec"] <= win_end)
            ]
            if part_in_window.empty:
                continue  # Skip empty windows

            part_embeds = (
                np.vstack(part_in_window["embedding"].values)
                if not part_in_window.empty
                else np.zeros((1, 3072))
            )
            oper_embeds = (
                np.vstack(oper_in_window["embedding"].values)
                if not oper_in_window.empty
                else np.zeros((1, 3072))
            )

            part_avg = np.mean(part_embeds, axis=0).reshape(-1)
            oper_avg = np.mean(oper_embeds, axis=0).reshape(-1)

            combined = np.concatenate([part_avg, oper_avg])
            X.append(combined)
            y.append(part["eng_level"].iloc[0])
            time_marks.append((key, win_start, win_end))
            groups.append(part["subject_id"].iloc[0])

    # Convert to arrays
    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

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
                        learning_rate=0.1,
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
        print(
            f"\nFold {fold} | Test Subject: {groups[test_idx[0]]} | Accuracy: {acc:.4f}"
        )

    print("Leave-One-Group-Out CV Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print("Classification Report:")
    print(
        classification_report(
            all_y_true, all_y_pred, target_names=le.classes_, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
