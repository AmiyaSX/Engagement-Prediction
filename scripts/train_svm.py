import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Set up caching directory
memory = joblib.Memory(location="svm_cache", verbose=0)

# === Step 1: Load the feature data ===
df = pd.read_csv("data/features/gaze_features.csv")
df = df.dropna()
# === Step 2: Prepare labels and features ===
# Target labels
y = df["Label"].astype(str)

# Drop metadata columns that aren't features
X = df.drop(
    columns=["Label", "SubjectID", "RunID", "WindowStart", "WindowEnd", "WindowIndex"]
)

# === Step 3: Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 4: Build pipeline (scaling + SVM) ===
# Build pipeline with memory argument
pipeline = Pipeline(
    [
        # ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ],
    memory=memory,
)

# === Step 5: Train ===
pipeline.fit(X_train, y_train)

# === Step 6: Predict ===
y_pred = pipeline.predict(X_test)

# === Step 7: Evaluate ===
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Step 8: Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)
labels = sorted(df["Label"].unique())
conf_df = pd.DataFrame(
    conf_matrix,
    index=[f"True {l}" for l in labels],
    columns=[f"Pred {l}" for l in labels],
)

print("\nConfusion Matrix:\n", conf_df)
