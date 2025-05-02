import pandas as pd
import os
from glob import glob


def extract_features_from_df(df):
    df = pre_process_gaze_data(df)
    return extract_participant_features(df)


def pre_process_gaze_data(df):
    df = df.dropna(subset=["Timestamp"])
    df["TimeSeconds"] = df["Timestamp"] - df["Timestamp"].min()
    df.columns = df.columns.str.strip()

    continuous_cols = [
        "Left Screen X",
        "Left Screen Y",
        "Left Pupil Diameter",
        "Right Screen X",
        "Right Screen Y",
        "Right Pupil Diameter",
    ]

    df[continuous_cols] = df[continuous_cols].interpolate(
        method="linear", limit_direction="forward"
    )
    df[continuous_cols] = df[continuous_cols].fillna(method="bfill")

    return df


def extract_participant_features(df):
    features = {}
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
    data_path = "data/raw/eyetracking"
    output_path = os.path.join("data", "features")
    os.makedirs(output_path, exist_ok=True)

    files = sorted(glob(os.path.join(data_path, "*.csv")))

    all_features = []
    subject_ids = []
    run_ids = []
    labels = {"run-01": 1, "run-02": 2, "run-03": 3}

    for filepath in files:
        df = pd.read_csv(filepath)
        features = extract_features_from_df(df)
        filename = os.path.basename(filepath)

        subject_id = filename.split("_")[0]  # e.g., 'sub-01'
        run_id = filename.split("_")[1].replace(".csv", "")  # e.g., 'run-01'

        features["SubjectID"] = subject_id
        features["RunID"] = run_id
        features["Label"] = labels.get(run_id, -1)  # Default to -1 if unknown run
        subject_ids.append(subject_id)
        run_ids.append(run_id)

        all_features.append(features)

    X = pd.DataFrame(all_features)
    output_file = os.path.join(output_path, "participant_features.csv")
    X.to_csv(output_file, index=False)
    print(f"Feature extraction complete. Output saved to {output_file}")


if __name__ == "__main__":
    main()
