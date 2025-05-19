import pandas as pd
import os


def align_by_timestamp(df):
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    df = df[df["Timestamp"]]
    return df


def main():
    data_path = "data/raw/eyetracking/cleaned"
    sampled_path = os.path.join(data_path, "sampled")
    os.makedirs(sampled_path, exist_ok=True)

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        try:
            sampled_df = pd.read_csv(file_path)
            df = sampled_df.iloc[::20, :]
            df.to_csv(os.path.join(sampled_path, f"{file_name}"), index=False)
            print(f"Sampled: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()
