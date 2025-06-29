import os
import cv2
import pandas as pd
import numpy as np

input_csv_dir = "data/raw/eyetracking/sampled/train"
video_dir = "data/raw/video"
output_dir = "data/raw/eyetracking/feature"
os.makedirs(output_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# === Constants ===
frame_width, frame_height = 1280, 720
video_center_x, video_center_y = frame_width / 2, frame_height / 2
x_min, x_max = -640, 640
y_min, y_max = -360, 360
horizontal_offset, verdict_offset = 0, 0
detect_every = 5
fps = 25
window_size = 120
step_size = window_size


def center_normalize(x, src_min, src_max, center):
    norm = (x - src_min) / (src_max - src_min)
    return (norm - 0.5) * (2 * center)


def in_mutual_gaze_zone(gx, gy, face_rect):
    x, y, w, h = face_rect
    margin_x, margin_y = w * 0.1, h * 0.1
    return (x + margin_x <= gx <= x + w - margin_x) and (
        y + margin_y <= gy <= y + h - margin_y
    )


for fname in os.listdir(input_csv_dir):
    if not fname.endswith("_eye_tracking.csv"):
        continue

    file_base = fname.replace("cleaned_", "").replace("_eye_tracking.csv", "")

    if (
        file_base.find("05") > -1
        or file_base.find("08") > -1
        or file_base.find("15") > -1
        or file_base.find("07") > -1
        or file_base.find("19") > -1
        or file_base.find("28") > -1
        or file_base.find("33") > -1
    ):
        continue
    print(file_base)
    video_path = os.path.join(video_dir, f"{file_base}.mp4")
    csv_path = os.path.join(input_csv_dir, fname)
    output_csv_path = os.path.join(output_dir, f"{file_base}_tracking_resampled.csv")

    if not os.path.exists(video_path):
        print(f"Skipping {file_base}: video not found.")
        continue

    print(f"Processing {file_base}...")

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        mutual_gaze_flags = np.zeros(len(df), dtype=bool)
        df["LR_X_deviation"] = (df["Left Screen X"] - df["Right Screen X"]).abs()
        df["LR_Y_deviation"] = (df["Left Screen Y"] - df["Right Screen Y"]).abs()

        for eye in ["Left", "Right"]:
            df[f"{eye} Screen X"] = (
                center_normalize(df[f"{eye} Screen X"], x_min, x_max, video_center_x)
                + horizontal_offset
            )
            df[f"{eye} Screen Y"] = (
                center_normalize(df[f"{eye} Screen Y"], y_min, y_max, video_center_y)
                * -1
                + verdict_offset
            )

        cap = cv2.VideoCapture(video_path)
        last_face = None
        frame_idx = 0
        df_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(df):
                break

            if frame_idx % detect_every == 0 or last_face is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                last_face = faces[0] if len(faces) > 0 else None

            while df_idx < frame_idx * 40 and df_idx < len(df):
                row = df.iloc[df_idx]
                if last_face is not None and not any(
                    pd.isna(row[f"{eye} Screen {axis}"])
                    for eye in ["Left", "Right"]
                    for axis in ["X", "Y"]
                ):
                    left = (
                        int(video_center_x + row["Left Screen X"]),
                        int(video_center_y + row["Left Screen Y"]),
                    )
                    right = (
                        int(video_center_x + row["Right Screen X"]),
                        int(video_center_y + row["Right Screen Y"]),
                    )

                    left_in = in_mutual_gaze_zone(*left, last_face)
                    right_in = in_mutual_gaze_zone(*right, last_face)

                    if left_in and right_in:
                        mutual_gaze_flags[df_idx] = True

                df_idx += 1

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        df["mutual_gaze"] = mutual_gaze_flags.astype(int)

        features = []
        for start in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[start : start + window_size]
            feature_row = {
                "window_start": df.iloc[start]["Timestamp"],
                "mutual_gaze_ratio": window["mutual_gaze"].mean(),
                "Left_Screen_X_std": window["Left Screen X"].std(),
                "Left_Screen_X_range": window["Left Screen X"].max()
                - window["Left Screen X"].min(),
                "Left_Screen_Y_valid_ratio": window["Left Screen Y"].notna().mean(),
                "Left_Screen_Y_std": window["Left Screen Y"].std(),
                "Left_Screen_Y_range": window["Left Screen Y"].max()
                - window["Left Screen Y"].min(),
                "Right_Screen_X_valid_ratio": window["Right Screen X"].notna().mean(),
                "Right_Screen_Y_std": window["Right Screen Y"].std(),
                "Right_Screen_Y_range": window["Right Screen Y"].max()
                - window["Right Screen Y"].min(),
                "Left_Pupil_Diameter_std": window["Left Pupil Diameter"].std(),
                "Left_Pupil_Diameter_range": window["Left Pupil Diameter"].max()
                - window["Left Pupil Diameter"].min(),
                "Right_Pupil_Diameter_valid_ratio": window["Right Pupil Diameter"]
                .notna()
                .mean(),
                "LR_X_deviation_mean": window["LR_X_deviation"].mean(),
                "LR_Y_deviation_std": window["LR_Y_deviation"].std(),
                "fixation_count": window.get("Left Fixation", pd.Series()).sum()
                + window.get("Right Fixaion", pd.Series()).sum(),
                "mutual_fixation_count": (
                    (
                        (window["mutual_gaze"] == 1)
                        & (
                            (window.get("Left Fixation", pd.Series()) == 1)
                            | (window.get("Right Fixaion", pd.Series()) == 1)
                        )
                    ).sum()
                ),
                "blink_count": window.get("Left Blink", pd.Series()).sum()
                + window.get("Right Blink", pd.Series()).sum(),
                "saccade_count": window.get("Left Eye Saccade", pd.Series()).sum()
                + window.get("Right Eye Saccade", pd.Series()).sum(),
            }
            features.append(feature_row)

        feature_df = pd.DataFrame(features)
        feature_df.rename(columns={"window_start": "Timestamp"}, inplace=True)
        feature_df.to_csv(output_csv_path, index=False)
        print(f"✅ Saved features for {file_base} to {output_csv_path}")

    except Exception as e:
        print(f"❌ Failed to process {file_base}: {e}")
