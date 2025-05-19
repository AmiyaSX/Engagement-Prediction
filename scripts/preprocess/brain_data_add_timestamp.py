import pandas as pd
import os
import glob

# Directory containing all ROI CSV files
data_path = "data/raw/brain"
sampling_rate = 1.2

# Process each file ending in _roi_data.csv
for file_path in glob.glob(os.path.join(data_path, "*_roi_data.csv")):
    print(f"Processing {file_path}...")

    df = pd.read_csv(file_path)
    timestamps = []

    # Add timestamps per run
    for _, group in df.groupby("run_index"):
        n = len(group)
        timestamps.extend([i * sampling_rate for i in range(n)])

    # Assign timestamps
    df["timestamp"] = timestamps

    # Save to a new file (optional: overwrite or save as a new version)
    output_path = file_path.replace(".csv", "_with_timestamps.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
