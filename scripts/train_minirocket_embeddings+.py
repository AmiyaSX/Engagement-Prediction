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


# Load and process one transcript files
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
            with open(cache_path, "w") as f:
                json.dump(df["embedding"].tolist(), f)

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def extract_id(file_name):
    parts = file_name.split("_")
    subject = parts[0]  # sub-xx
    run = parts[2]
    run = run.split("-")[1].split(".")[0]
    return f"{subject}_run-{run}"


def main():

    # Load transcripts
    participant_df = load_transcripts("data/raw/transcripts/participant")

    # Create normalized keys for matching
    participant_df["match_key"] = participant_df["file"].apply(extract_id)

    all_keys = set(participant_df["match_key"])
    print(f"Found {len(all_keys)} matched transcript pairs.")

    X, y, time_marks, groups = [], [], [], []

    for key in tqdm(all_keys):
        part = participant_df[participant_df["match_key"] == key]

        max_time = part["end_sec"].max()

        for win_start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            part_in_window = part[
                (part["mid_time"] >= win_start) & (part["mid_time"] <= win_end)
            ]
            if part_in_window.empty:
                continue  # Skip empty windows

            part_embeds = (
                np.vstack(part_in_window["embedding"].values)
                if not part_in_window.empty
                else np.zeros((1, 3072))
            )
            part_avg = np.mean(part_embeds, axis=0).reshape(-1)

            # Augment with speaker/temporal features
            utterance_count = len(part_in_window)

            # Word count per utterance
            utterance_lengths = part_in_window["transcription"].apply(
                lambda t: len(str(t).split())
            )
            avg_utterance_length = utterance_lengths.mean()

            # Total words / total speaking time in the window
            total_words = utterance_lengths.sum()
            total_speaking_time = (
                part_in_window["end_sec"].sum() - part_in_window["start_sec"].sum()
            )
            speaking_rate = (
                total_words / total_speaking_time if total_speaking_time > 0 else 0.0
            )

            extra_features = np.array(
                [utterance_count, avg_utterance_length, speaking_rate]
            )
            combined_features = np.concatenate([part_avg, extra_features])

            X.append(combined_features)
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

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    subject_test_id = 5  # sub01
    train_mask = groups != subject_test_id
    test_mask = groups == subject_test_id

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]

    # Train MiniRocket pipeline
    pipeline = Pipeline(
        steps=[
            ("transform", MiniRocket(random_state=42)),
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

    # Predict & evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    )


if __name__ == "__main__":
    main()
