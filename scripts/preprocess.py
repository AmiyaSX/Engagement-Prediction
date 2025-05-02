import pandas as pd

df = pd.read_csv("data/your_data.csv")


def extract_features(group):
    features = {}
    # Basic stats
    for col in [
        "Left Screen X",
        "Left Screen Y",
        "Right Screen X",
        "Right Screen Y",
        "Left Pupil Diameter",
        "Right Pupil Diameter",
    ]:
        series = group[col].dropna()
        features[f"{col}_mean"] = series.mean()
        features[f"{col}_std"] = series.std()
        features[f"{col}_min"] = series.min()
        features[f"{col}_max"] = series.max()
        features[f"{col}_range"] = series.max() - series.min()

    # Eye movement: fixations and saccades
    features["fixation_count"] = (
        group["Left Fixation"].sum() + group["Right Fixaion"].sum()
    )
    features["saccade_count"] = (
        group["Left Eye Saccade"].sum() + group["Right Eye Saccade"].sum()
    )

    # Blinks
    features["blink_count"] = group["Left Blink"].sum() + group["Right Blink"].sum()

    return pd.Series(features)


features_df = df.groupby("window_id").apply(extract_features).reset_index()
