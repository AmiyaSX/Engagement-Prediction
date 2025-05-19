import pandas as pd
import os

# Load mapping CSV
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df["eyetracking_duration"] = mapping_df.apply(
    lambda row: (
        row["eyetracking_end"] - row["eyetracking_start"]
        if pd.notna(row["eyetracking_start"]) and pd.notna(row["eyetracking_end"])
        else 600
    ),
    axis=1,
)

# Create dictionary for quick lookup
duration_map = {
    (row["subject_id"], row["run"]): (
        row["eyetracking_start"],
        row["eyetracking_duration"],
    )
    for _, row in mapping_df.iterrows()
}


def align_by_timestamp(df, duration):
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    df = df[df["Timestamp"] <= duration]
    return df


def main():
    data_path = "data/raw/eyetracking"
    cleaned_path = os.path.join(data_path, "cleaned")
    os.makedirs(cleaned_path, exist_ok=True)

    for file_name in os.listdir(data_path):
        if not file_name.endswith(".csv") or "cleaned_" in file_name:
            continue

        # Extract subject_id and run from filename
        try:
            parts = file_name.replace(".csv", "").split("_")
            subject_id = int(parts[0].split("-")[1])
            run = int(parts[1].split("-")[1])
        except Exception as e:
            print(f"Skipping file {file_name}: could not parse subject/run.")
            continue

        if (subject_id, run) not in duration_map:
            print(f"Skipping file {file_name}: no duration mapping found.")
            continue

        start_stamp, duration = duration_map[(subject_id, run)]

        if pd.isna(start_stamp):
            print(f"Using default 600s duration for file {file_name}.")
            start_stamp = 0
            duration = 600

        file_path = os.path.join(data_path, file_name)
        try:
            df = pd.read_csv(file_path)
            cleaned_df = align_by_timestamp(df, duration)
            cleaned_df.to_csv(
                os.path.join(cleaned_path, f"cleaned_{file_name}"), index=False
            )
            print(f"Cleaned: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()
