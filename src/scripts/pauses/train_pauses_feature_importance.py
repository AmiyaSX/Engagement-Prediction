import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIG ===
data_path = "data/raw/pause/prod_comp_gaps_pauses_all_included.csv"
WINDOW_SIZE = 12  # seconds
STEP_SIZE = 5  # seconds
EVENT_TYPES = [
    "production",
    "comprehension",
    "pause_p",
    "pause_c",
    "gap_p2c",
    "gap_c2p",
]
total_window_time = WINDOW_SIZE

# === Load & Preprocess Data ===
df = pd.read_csv(data_path)
df["subject"] = df["filename"].apply(lambda x: int(x.split("_")[0].split("-")[1]))
df["run"] = df["filename"].apply(lambda x: int(x.split("_")[1].split("-")[1]))

X, y, groups, metadata = [], [], [], []

grouped = df.groupby(["subject", "run"])
for (subject, run), group in tqdm(grouped, desc="Processing windows"):
    max_time = group["start_time"].max() + group["duration"].max()
    for start in np.arange(0, max_time - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = group[(group["start_time"] >= start) & (group["start_time"] < end)]
        if window.empty:
            continue

        # Feature extraction
        feat = {}
        for ev in EVENT_TYPES:
            ev_data = window[window["event_type"] == ev]
            durations = ev_data["duration"].values
            start_times = ev_data["start_time"].values

            feat[f"count_{ev}"] = len(durations)
            feat[f"duration_{ev}"] = durations.sum()
            if len(durations) > 0:
                feat[f"mean_duration_{ev}"] = durations.mean()
                feat[f"std_duration_{ev}"] = durations.std()
                feat[f"max_duration_{ev}"] = durations.max()
                feat[f"min_duration_{ev}"] = durations.min()
                feat[f"median_duration_{ev}"] = np.median(durations)
                feat[f"rate_{ev}"] = len(durations) / total_window_time
                feat[f"proportion_time_{ev}"] = durations.sum() / total_window_time

                # Gaps before each event
                if len(start_times) > 1:
                    gaps = np.diff(start_times)
                    feat[f"gap_before_mean_{ev}"] = gaps.mean()
                    feat[f"gap_before_std_{ev}"] = gaps.std()
                else:
                    feat[f"gap_before_mean_{ev}"] = 0
                    feat[f"gap_before_std_{ev}"] = 0
            else:
                # Fill missing values to keep features aligned
                for stat in [
                    "mean_duration",
                    "std_duration",
                    "max_duration",
                    "min_duration",
                    "median_duration",
                    "rate",
                    "proportion_time",
                    "gap_before_mean",
                    "gap_before_std",
                ]:
                    feat[f"{stat}_{ev}"] = 0

        labels = window["eng_level"].unique()
        if len(labels) != 1:
            continue  # skip windows with inconsistent labels
        label = labels[0]

        X.append(feat)
        y.append(label)
        groups.append(subject)
        metadata.append(
            {
                "subject": subject,
                "run": run,
                "win_start": start,
                "win_end": end,
                "label": label,
            }
        )

# === Prepare Data ===
X = pd.DataFrame(X).fillna(0)
y = pd.Series(y)
groups = pd.Series(groups)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Model ===
pipeline = Pipeline(
    [
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=len(le.classes_),
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=4,
            ),
        )
    ]
)

logo = LeaveOneGroupOut()
accuracies = []
all_y_true, all_y_pred = [], []

print(f"\n=== Training on {len(X)} samples ===")

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_encoded, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    sample_weights = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold} | Subject {groups.iloc[test_idx[0]]} | Accuracy: {acc:.4f}")

# === Final Results ===
print("\n=== Final Results ===")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print("Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

# === Feature Importance Extraction ===
print("\n=== Extracting Feature Importances ===")
importances = pipeline.named_steps["classifier"].feature_importances_
feature_names = X.columns.tolist()

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

# Save to CSV
importance_df.to_csv("pause_feature_importance_no_duration.csv", index=False)

# Print top 10 features
print("\nTop 10 Important Features (from XGBoost):")
print(importance_df.head(10))

top_n = 10
importance_df.head(top_n).plot.barh(
    x="Feature", y="Importance", figsize=(8, 4), legend=False
)
plt.gca().invert_yaxis()
plt.title("Top 10 Pause-Based Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
