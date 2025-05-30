import pandas as pd
import os
import json
from datetime import timedelta
from google.api_core import retry
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import joblib
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import glob

WINDOW_SIZE = 12
STEP_SIZE = 5

# Setup Gemini API client
genai.configure(api_key="MY KEY")

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
        # print(filename)
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
            part_in_window = part[
                (part["mid_time"] >= win_start) & (part["mid_time"] <= win_end)
            ]
            oper_in_window = oper[
                (oper["mid_time"] >= win_start) & (oper["mid_time"] <= win_end)
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

            # ----------- Participant Features ------------
            # part_utterance_count = len(part_in_window)

            # part_utterance_lengths = part_in_window["transcription"].apply(
            #     lambda t: len(str(t).split())
            # )
            # part_avg_utterance_length = part_utterance_lengths.mean()

            # part_total_words = part_utterance_lengths.sum()
            # part_total_speaking_time = (
            #     part_in_window["end_sec"].sum() - part_in_window["start_sec"].sum()
            # )
            # part_speaking_rate = (
            #     part_total_words / part_total_speaking_time
            #     if part_total_speaking_time > 0
            #     else 0.0
            # )
            # # ----------- Combine Everything ------------
            # extra_features = np.array(
            #     [part_utterance_count, part_avg_utterance_length, part_speaking_rate]
            # )

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

    # X = X.reshape((X.shape[0], 1, X.shape[1]))

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

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
