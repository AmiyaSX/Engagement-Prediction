import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import pandas as pd

# Load Haar cascade for face detection (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
last_face = None
detect_every = 10


x_min = -640
x_max = 640
y_min = -360
y_max = 360
# Video resolution
frame_width = 1280
frame_height = 720

# horizontal_offset = 250
# verdict_offset = 100

# offset for 19
# horizontal_offset = -150
# verdict_offset = 50

# offset for 28
# horizontal_offset = 0
# verdict_offset = 150

# offset for 33
# horizontal_offset = -250
# verdict_offset = 50

# offset for 8
horizontal_offset = -50
verdict_offset = 50
file = "sub-08_run-01"
# --- Load background video ---
video_path = f"data/raw/video/{file}.mp4"
video_output = f"gaze_overlay_{file}.mp4"
# Save the resampled eyetracking data
output_csv_path = f"data/raw/eyetracking/{file}_eye_tracking_resampled.csv"
# Load the data
df = pd.read_csv(f"data/raw/eyetracking/sampled/train/cleaned_{file}_eye_tracking.csv")
df.columns = df.columns.str.strip()

# --- ⏱ Resample to match video FPS ---
cap = cv2.VideoCapture(video_path)
# Your video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps)
resample_interval = f"{int(1000 / fps)}ms"
# Convert 'Timestamp' to timedelta and set as index
df = df.set_index(pd.to_timedelta(df["Timestamp"], unit="s"))

# Drop the column 'Timestamp' if it's still in columns
df = df.drop(columns=["Timestamp"], errors="ignore")

# Resample and interpolate
df = df.resample(resample_interval).mean().interpolate()

# Reset index and convert it back to seconds
df = df.reset_index()
df.rename(columns={"index": "Timestamp"}, inplace=True)
df["Timestamp"] = df["Timestamp"].dt.total_seconds()


# Calculate center-relative coordinates ---
video_center_x = frame_width / 2
video_center_y = frame_height / 2


# Normalize gaze coordinates around (0, 0) at video center
def center_normalize(x, src_min, src_max, center):
    norm = (x - src_min) / (src_max - src_min)  # 0–1
    return (norm - 0.5) * (2 * center)  # center at 0


print(x_min)
print(x_max)
print(y_min)
print(y_max)

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


# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))


def in_mutual_gaze_zone(gx, gy, face_rect):
    x, y, w, h = face_rect
    margin_x, margin_y = w * 0.1, h * 0.1
    return (x + margin_x <= gx <= x + w - margin_x) and (
        y + margin_y <= gy <= y + h - margin_y
    )


# --- Loop through video frames and overlay gaze ---
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(df):
        break

    row = df.iloc[frame_idx]

    if frame_idx % detect_every == 0 or last_face is None:
        # Detect face (use first detected face)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        last_face = faces[0] if len(faces) > 0 else None
    if last_face is not None:
        face = faces[0]
        x, y, w, h = last_face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Skip frames where gaze is missing
        if any(
            pd.isna(row[f"{eye} Screen {axis}"])
            for eye in ["Left", "Right"]
            for axis in ["X", "Y"]
        ):
            frame_idx += 1
            continue
        # Convert center-relative to video coordinates
        left = (
            int(video_center_x + row["Left Screen X"]),
            int(video_center_y + row["Left Screen Y"]),
        )
        right = (
            int(video_center_x + row["Right Screen X"]),
            int(video_center_y + row["Right Screen Y"]),
        )

        cv2.circle(frame, left, 10, (255, 0, 0), -1)  # Blue for left eye
        cv2.circle(frame, right, 10, (0, 0, 255), -1)  # Red for right eye

        cv2.putText(
            frame,
            f"Time: {row['Timestamp']:.2f}s",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        # Check mutual gaze
        left_in = in_mutual_gaze_zone(*left, face)
        right_in = in_mutual_gaze_zone(*right, face)

        # Show mutual gaze status
        color = (0, 255, 255) if left_in and right_in else (0, 0, 0)
        cv2.putText(
            frame,
            f'Mutual Gaze: {"YES" if left_in and right_in else "NO"}',
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    else:
        cv2.putText(
            frame,
            f"Time: {row['Timestamp']:.2f}s",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "No face detected",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
