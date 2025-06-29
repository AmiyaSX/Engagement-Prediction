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
fps = 25  # frames per second for velocity calc
window_size = 120  # samples per window (approx 1.2s)
step_size = window_size


def center_normalize(x, src_min, src_max, center):
    norm = (x - src_min) / (src_max - src_min)
    return (norm - 0.5) * (2 * center)


def compute_velocity(pos, fps):
    """Compute velocity as frame-to-frame difference * fps."""
    velocity = np.diff(pos) * fps
    velocity = np.append(velocity, np.nan)
    return velocity


def compute_acceleration(velocity, fps):
    """Compute acceleration as frame-to-frame difference of velocity * fps."""
    acceleration = np.diff(velocity) * fps
    acceleration = np.append(acceleration, np.nan)
    return acceleration


for fname in os.listdir(input_csv_dir):
    if not fname.endswith("_eye_tracking.csv"):
        continue

    file_base = fname.replace("cleaned_", "").replace("_eye_tracking.csv", "")

    # Optional: skip some subjects if needed (your original logic)
    if (
        file_base.find("15") > -1
    ):
        continue

    print(f"Processing {file_base}...")

    video_path = os.path.join(video_dir, f"{file_base}.mp4")
    csv_path = os.path.join(input_csv_dir, fname)
    output_csv_path = os.path.join(output_dir, f"{file_base}_tracking_resampled.csv")

    if not os.path.exists(video_path):
        print(f"Skipping {file_base}: video not found.")
        continue

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Calculate LR deviations
        df["LR_X_deviation"] = (df["Left Screen X"] - df["Right Screen X"]).abs()
        df["LR_Y_deviation"] = (df["Left Screen Y"] - df["Right Screen Y"]).abs()

        # Normalize gaze coordinates around center
        # for eye in ["Left", "Right"]:
        #     df[f"{eye} Screen X"] = (
        #         center_normalize(df[f"{eye} Screen X"], x_min, x_max, video_center_x)
        #         + horizontal_offset
        #     )
        #     df[f"{eye} Screen Y"] = (
        #         center_normalize(df[f"{eye} Screen Y"], y_min, y_max, video_center_y)
        #         * -1
        #         + verdict_offset
        #     )

        # Compute velocity and acceleration for gaze X and Y per eye
        for eye in ["Left", "Right"]:
            for axis in ["X", "Y"]:
                col = f"{eye} Screen {axis}"
                pos = df[col].to_numpy(dtype=float)
                # Handle missing data by forward filling for velocity calculation
                pos_ffill = pd.Series(pos).ffill().bfill().to_numpy()
                velocity = compute_velocity(pos_ffill, fps)
                acceleration = compute_acceleration(velocity, fps)
                df[f"{col}_vel"] = velocity
                df[f"{col}_acc"] = acceleration

        # Now extract features in windows
        features = []
        for start in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[start : start + window_size]

            feature_row = {
                "window_start": window["Timestamp"].iloc[0],

                # Left eye gaze stats
                "Left_Screen_X_mean": window["Left Screen X"].mean(),
                "Left_Screen_X_std": window["Left Screen X"].std(),
                "Left_Screen_X_range": window["Left Screen X"].max() - window["Left Screen X"].min(),
                "Left_Screen_X_valid_ratio": window["Left Screen X"].notna().mean(),

                "Left_Screen_Y_mean": window["Left Screen Y"].mean(),
                "Left_Screen_Y_std": window["Left Screen Y"].std(),
                "Left_Screen_Y_range": window["Left Screen Y"].max() - window["Left Screen Y"].min(),
                "Left_Screen_Y_valid_ratio": window["Left Screen Y"].notna().mean(),

                "Left_Pupil_Diameter_mean": window["Left Pupil Diameter"].mean(),
                "Left_Pupil_Diameter_std": window["Left Pupil Diameter"].std(),
                "Left_Pupil_Diameter_range": window["Left Pupil Diameter"].max() - window["Left Pupil Diameter"].min(),
                "Left_Pupil_Diameter_valid_ratio": window["Left Pupil Diameter"].notna().mean(),

                # Left eye velocity & acceleration stats
                "Left_Screen_X_vel_mean": window["Left Screen X_vel"].mean(),
                "Left_Screen_X_vel_std": window["Left Screen X_vel"].std(),
                "Left_Screen_X_acc_mean": window["Left Screen X_acc"].mean(),
                "Left_Screen_X_acc_std": window["Left Screen X_acc"].std(),

                "Left_Screen_Y_vel_mean": window["Left Screen Y_vel"].mean(),
                "Left_Screen_Y_vel_std": window["Left Screen Y_vel"].std(),
                "Left_Screen_Y_acc_mean": window["Left Screen Y_acc"].mean(),
                "Left_Screen_Y_acc_std": window["Left Screen Y_acc"].std(),

                # Right eye gaze stats
                "Right_Screen_X_mean": window["Right Screen X"].mean(),
                "Right_Screen_X_std": window["Right Screen X"].std(),
                "Right_Screen_X_range": window["Right Screen X"].max() - window["Right Screen X"].min(),
                "Right_Screen_X_valid_ratio": window["Right Screen X"].notna().mean(),

                "Right_Screen_Y_mean": window["Right Screen Y"].mean(),
                "Right_Screen_Y_std": window["Right Screen Y"].std(),
                "Right_Screen_Y_range": window["Right Screen Y"].max() - window["Right Screen Y"].min(),
                "Right_Screen_Y_valid_ratio": window["Right Screen Y"].notna().mean(),

                "Right_Pupil_Diameter_mean": window["Right Pupil Diameter"].mean(),
                "Right_Pupil_Diameter_std": window["Right Pupil Diameter"].std(),
                "Right_Pupil_Diameter_range": window["Right Pupil Diameter"].max() - window["Right Pupil Diameter"].min(),
                "Right_Pupil_Diameter_valid_ratio": window["Right Pupil Diameter"].notna().mean(),

                # Right eye velocity & acceleration stats
                "Right_Screen_X_vel_mean": window["Right Screen X_vel"].mean(),
                "Right_Screen_X_vel_std": window["Right Screen X_vel"].std(),
                "Right_Screen_X_acc_mean": window["Right Screen X_acc"].mean(),
                "Right_Screen_X_acc_std": window["Right Screen X_acc"].std(),

                "Right_Screen_Y_vel_mean": window["Right Screen Y_vel"].mean(),
                "Right_Screen_Y_vel_std": window["Right Screen Y_vel"].std(),
                "Right_Screen_Y_acc_mean": window["Right Screen Y_acc"].mean(),
                "Right_Screen_Y_acc_std": window["Right Screen Y_acc"].std(),

                # Eye quality features
                "LR_X_deviation_mean": window["LR_X_deviation"].mean(),
                "LR_Y_deviation_std": window["LR_Y_deviation"].std(),

                # Eye events counts
                "fixation_count": window.get("Left Fixation", pd.Series(dtype='int')).sum()
                                  + window.get("Right Fixation", pd.Series(dtype='int')).sum(),

                "blink_count": window.get("Left Blink", pd.Series(dtype='int')).sum()
                               + window.get("Right Blink", pd.Series(dtype='int')).sum(),

                "saccade_count": window.get("Left Eye Saccade", pd.Series(dtype='int')).sum()
                                + window.get("Right Eye Saccade", pd.Series(dtype='int')).sum(),
            }
            features.append(feature_row)

        feature_df = pd.DataFrame(features)
        feature_df.rename(columns={"window_start": "Timestamp"}, inplace=True)
        feature_df.to_csv(output_csv_path, index=False)
        print(f"✅ Saved features for {file_base} to {output_csv_path}")

    except Exception as e:
        print(f"❌ Failed to process {file_base}: {e}")
