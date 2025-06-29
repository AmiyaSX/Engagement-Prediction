import pandas as pd
import numpy as np
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sktime.transformations.panel.rocket import MiniRocket
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Constants
WINDOW_SIZE = 10  # samples, 12s
STEP_SIZE = 5  # samples, 6s
subjects = [5, 7, 8, 11, 12, 13, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33]

# Hyperparameter grid for XGBoost
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 6],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
}

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


def load_roi_data(window_size=10, step_size=4):
    X, y, groups = [], [], []
    metadata = []
    file_paths = glob.glob("data/raw/brain/train/*_with_timestamps.csv")

    for file in file_paths:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        filename = os.path.basename(file)

        # Extract subject ID from filename: "sub-05_roi_data_with_timestamps.csv"
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
        except (IndexError, ValueError):
            print(f"Could not extract subject_id from filename: {filename}")
            continue

        if "run_index" not in df.columns:
            print(f"No 'run_index' column in {filename}")
            continue

        # Normalize ROI features per subject (enhanced more accuracy)
        roi_columns = [
            "comp_vs_prod_combined",
            "dmn_medial_tpj",
            "prod_vs_comp_combined",
        ]
        df[roi_columns] = (df[roi_columns] - df[roi_columns].mean()) / df[
            roi_columns
        ].std()

        # Group by run_index
        for run_id_str, group in df.groupby("run_index"):
            try:
                run_id = int(run_id_str.strip().split("-")[1])
            except (IndexError, ValueError):
                print(f"Could not parse run_id from value '{run_id_str}' in {filename}")
                continue

            # Lookup engagement level
            label_row = mapping_df[
                (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
            ]
            if label_row.empty:
                print(f"Warning: No mapping for Subject {subject_id}, Run {run_id}")
                continue

            label = label_row["EngagementLevel"].values[0]

            # Extract ROI features
            features = (
                group[
                    ["comp_vs_prod_combined", "dmn_medial_tpj", "prod_vs_comp_combined"]
                ]
                .to_numpy()
                .T
            )  # shape: (3, n_samples)

            timestamps = group["timestamp"].to_numpy()

            for i in range(0, features.shape[1] - window_size + 1, step_size):
                start_time = timestamps[i]
                end_time = timestamps[i + window_size - 1]
                window = features[:, i : i + window_size]
                X.append(window)
                y.append(label)
                groups.append(subject_id)

                # Record metadata
                metadata.append(
                    {
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": label,
                    }
                )

    if not X:
        raise ValueError("No valid data was loaded. Check input files and mappings.")

    X = np.stack(X)  # shape: (n_samples, 3, window_size)
    y = np.array(y)
    groups = np.array(groups)
    metadata_df = pd.DataFrame(metadata)
    return X, y, groups, metadata_df


X, y, groups, metadata_df = load_roi_data()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Encode string labels to integers
y_encoded = le.fit_transform(y)  # Now y_encoded contains 0, 1, 2

pipeline = Pipeline(
    steps=[
        ("transform", MiniRocket(random_state=42)),
        # (
        #     "classifier",
        #     RandomForestClassifier(
        #         n_estimators=100,  # Number of trees
        #         class_weight="balanced",
        #         max_depth=3,
        #         min_samples_leaf=2,
        #         max_features="sqrt",
        #         random_state=42,
        #         n_jobs=-1,
        #     ),
        # ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=len(le.classes_),
                eval_metric="mlogloss",
                subsample=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ],
    memory=joblib.Memory(location="cache_dir", verbose=0),
)

# Leave-One-Group-Out Cross Validation
# logo = LeaveOneGroupOut()
from sklearn.utils.class_weight import compute_sample_weight
# Save metadata
metadata_df.to_csv("window_metadata.csv", index=False)
print(f"\nSaved metadata for {len(metadata_df)} windows to 'window_metadata.csv'")

logo = LeaveOneGroupOut()
accuracies = []
fold = 1

for train_idx, test_idx in logo.split(X, y_encoded, groups):
    print(f"\nFold {fold} | Test Group: [{groups[test_idx[0]]}]")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    group_test = groups[test_idx[0]]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    accuracies.append(acc)
    fold += 1

print("\n=== LOGO CV Summary ===")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Std Accuracy: {np.std(accuracies):.4f}")

# subject_test_id = 7  # sub01
# train_mask = groups != subject_test_id
# test_mask = groups == subject_test_id

# y_encoded = le.fit_transform(y)
# X_train, y_train = X[train_mask], y_encoded[train_mask]
# X_test, y_test = X[test_mask], y_encoded[test_mask]

# from sklearn.utils.class_weight import compute_sample_weight

# sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
# pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

# # Predict
# y_pred = pipeline.predict(X_test)


# # Accuracy
# acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc:.4f}")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, zero_division=0))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(
#     pd.DataFrame(
#         conf_matrix,
#         index=[f"Actual {label}" for label in pipeline.classes_],
#         columns=[f"Predicted {label}" for label in pipeline.classes_],
#     )
# )

# import random

# # Get a list of random indices from the test set
# sample_indices = random.sample(range(len(y_test)), min(10, len(y_test)))

# for i in sample_indices:
#     print(f"True: {y_test[i]}, Predicted: {y_pred[i]}")
