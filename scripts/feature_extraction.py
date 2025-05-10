import pandas as pd
import os
from glob import glob

# Constants
WINDOW_SIZE = 12  # seconds
STEP_SIZE = 5  # seconds

# Load mapping
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()

# Ensure EngagementLevel is standardized
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map(
        {
            "low": "low",
            "normal": "normal",
            "high": "high",
        }
    )
)


def pre_process_gaze_data(df):
    df = df.dropna(subset=["Timestamp"])
    df.columns = df.columns.str.strip()
    return df


def extract_participant_features(df, features):
    for col in [
        "Left Screen X",
        "Left Screen Y",
        "Right Screen X",
        "Right Screen Y",
        "Left Pupil Diameter",
        "Right Pupil Diameter",
    ]:
        valid = df[col].notna()
        features[f"{col}_valid_ratio"] = valid.mean()
        series = df[col].dropna()
        features[f"{col}_mean"] = series.mean()
        features[f"{col}_std"] = series.std()
        features[f"{col}_range"] = series.max() - series.min()

    # Eye event counts
    features["fixation_count"] = df["Left Fixation"].sum() + df["Right Fixaion"].sum()
    features["saccade_count"] = (
        df["Left Eye Saccade"].sum() + df["Right Eye Saccade"].sum()
    )
    features["blink_count"] = df["Left Blink"].sum() + df["Right Blink"].sum()

    return pd.Series(features)


def main():
    data_path = "data/raw/eyetracking/cleaned"
    output_path = os.path.join("data", "features")
    os.makedirs(output_path, exist_ok=True)

    files = sorted(glob(os.path.join(data_path, "*.csv")))

    all_features = []

    for filepath in files:
        df = pd.read_csv(filepath)
        df = pre_process_gaze_data(df)

        filename = os.path.basename(filepath)
        subject_id = int(filename.split("_")[1].split("-")[1])  # '1'
        run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])  # '1'

        # Get engagement label from mapping CSV
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
            continue

        label = label_row["EngagementLevel"].values[0]

        max_time = df["Timestamp"].max()

        start = 0
        window_idx = 0
        while start + WINDOW_SIZE <= max_time:
            end = start + WINDOW_SIZE
            window_df = df[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]

            if len(window_df) > 10:  # skip very short windows
                features = {}
                features["Label"] = label
                features["SubjectID"] = subject_id
                features["RunID"] = run_id
                features["WindowStart"] = start
                features["WindowEnd"] = end
                features["WindowIndex"] = window_idx
                features = extract_participant_features(window_df, features)
                all_features.append(features)
                window_idx += 1

            start += STEP_SIZE

    X = pd.DataFrame(all_features)
    output_file = os.path.join(output_path, "gaze_features.csv")
    X.to_csv(output_file, index=False)
    print(f"Feature extraction complete. Output saved to {output_file}")


if __name__ == "__main__":
    main()
