import pandas as pd
import numpy as np
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from xgboost import XGBClassifier
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Constants
WINDOW_SIZE = 1200  # samples, 12s
STEP_SIZE = 600  # samples, 6s

# Load mapping
mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()

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

train_data_path = "data/raw/eyetracking/sampled/train"
test_data_path = "data/raw/eyetracking/sampled/test"


def load_and_prepare_data(window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X, y, groups = [], [], []
    file_paths = glob.glob(f"{test_data_path}/*.csv")

    for file in file_paths:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        filename = os.path.basename(file)
        subject_id = int(filename.split("_")[1].split("-")[1])  # '1'
        run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])  # '1'

        # Extract 12 gaze channels
        features = (
            df[
                [
                    "Left Screen X",
                    "Left Screen Y",
                    "Right Screen X",
                    "Right Screen Y",
                    "Left Pupil Diameter",
                    "Right Pupil Diameter",
                    "Left Blink",
                    "Right Blink",
                    "Left Fixation",
                    "Right Fixaion",
                    "Left Eye Saccade",
                    "Right Eye Saccade",
                ]
            ]
            .to_numpy()
            .T
        )  # shape: (12, n_timepoints)

        # Get engagement label from mapping CSV
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
            continue

        label = label_row["EngagementLevel"].values[0]

        for i in range(0, features.shape[1] - window_size + 1, step_size):
            window = features[:, i : i + window_size]
            X.append(window)
            y.append(label)
            groups.append(subject_id)

    X = np.stack(X)  # shape (n_samples, 12, window_size)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


# Example pipeline with caching and reproducibility
pipeline = Pipeline(
    steps=[
        ("transform", MiniRocket(random_state=42)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,  # Number of trees
                max_depth=None,  # No max depth (you can set a value if needed)
                min_samples_leaf=2,  # Minimum number of samples per leaf
                max_features="sqrt",  # Number of features to consider at each split
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ],
    memory=joblib.Memory(location="cache_dir", verbose=0),  # Specify a cache directory
)

# Load data
X, y, groups = load_and_prepare_data()


# Split train/test dataset based on subject
subject_test_id = 1  # sub01
train_mask = groups != subject_test_id
test_mask = groups == subject_test_id

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)


# ------------- Evaluate ------------ #
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        conf_matrix,
        index=[f"Actual {label}" for label in pipeline.classes_],
        columns=[f"Predicted {label}" for label in pipeline.classes_],
    )
)

print("\nSample Predictions:")
for true, pred in zip(y_test[:10], y_pred[:10]):
    print(f"True: {true}, Predicted: {pred}")
