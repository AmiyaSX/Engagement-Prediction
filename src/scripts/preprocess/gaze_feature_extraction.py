import cv2
import pandas as pd
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
last_face = None

# === Constants ===
frame_width, frame_height = 1280, 720
video_center_x, video_center_y = frame_width / 2, frame_height / 2
x_min, x_max = -640, 640
y_min, y_max = -360, 360
# offset for 33
# horizontal_offset = -250
# verdict_offset = 50
# offset for 28
# horizontal_offset = 0
# verdict_offset = 150
# offset for 19
# horizontal_offset = -150
# verdict_offset = 50
# offset for 8
horizontal_offset = -50
verdict_offset = 50
# horizontal_offset = 0
# verdict_offset = 0
detect_every = 5


file = "sub-08_run-03"
# --- Load background video ---
video_path = f"data/raw/video/{file}.mp4"
output_csv_path = f"data/raw/eyetracking/feature/{file}_tracking_resampled.csv"
df = pd.read_csv(f"data/raw/eyetracking/sampled/train/cleaned_{file}_eye_tracking.csv")
df.columns = df.columns.str.strip()
mutual_gaze_flags = np.zeros(len(df), dtype=bool)
df["LR_X_deviation"] = (df["Left Screen X"] - df["Right Screen X"]).abs()
df["LR_Y_deviation"] = (df["Left Screen Y"] - df["Right Screen Y"]).abs()

# Normalize gaze coordinates around (0, 0) at video center
def center_normalize(x, src_min, src_max, center):
    norm = (x - src_min) / (src_max - src_min)  # 0–1
    return (norm - 0.5) * (2 * center)  # center at 0


def in_mutual_gaze_zone(gx, gy, face_rect):
    x, y, w, h = face_rect
    margin_x, margin_y = w * 0.1, h * 0.1
    return (x + margin_x <= gx <= x + w - margin_x) and (
        y + margin_y <= gy <= y + h - margin_y
    )

# Normalize all gaze coordinates
for eye in ["Left", "Right"]:
    df[f"{eye} Screen X"] = (
        center_normalize(df[f"{eye} Screen X"], x_min, x_max, video_center_x)
        + horizontal_offset
    )
    df[f"{eye} Screen Y"] = (
        center_normalize(df[f"{eye} Screen Y"], y_min, y_max, video_center_y) * -1
        + verdict_offset
    )  # invert Y axis

df.to_csv(output_csv_path, index=False)
print(f"Resampled eyetracking data saved to {output_csv_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # process all rows tied to this frame
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

# === Append mutual gaze column ===
df["mutual_gaze"] = mutual_gaze_flags.astype(int)

# === Feature Extraction over 1.2s Windows ===
window_size = 120
step_size = window_size  # use non-overlapping; for sliding set step_size < window_size
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
        "fixation_count": window["Left Fixation"].sum() + window["Right Fixaion"].sum(),
        "mutual_fixation_count": (
            (
                (window["mutual_gaze"] == 1)
                & ((window["Left Fixation"] == 1) | (window["Right Fixaion"] == 1))
            ).sum()
        ),
        "blink_count": window["Left Blink"].sum() + window["Right Blink"].sum(),
        "saccade_count": window["Left Eye Saccade"].sum()
        + window["Right Eye Saccade"].sum(),
    }
    features.append(feature_row)

# === Save Feature Data ===
feature_df = pd.DataFrame(features)
feature_df.rename(columns={"window_start": "Timestamp"}, inplace=True)
feature_df.to_csv(output_csv_path, index=False)
print(f"Feature data saved to {output_csv_path}")
