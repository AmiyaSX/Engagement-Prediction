import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

memory = joblib.Memory(location="svm_cache", verbose=0)

df = pd.read_csv("data/features/sub-01_features.csv")
df = df.dropna()

# Target labels
y = df["Label"].astype(str)

X = df.drop(
    columns=["Label", "SubjectID", "RunID", "WindowStart", "WindowEnd", "WindowIndex"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ],
    memory=memory,
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
labels = sorted(df["Label"].unique())
conf_df = pd.DataFrame(
    conf_matrix,
    index=[f"True {l}" for l in labels],
    columns=[f"Pred {l}" for l in labels],
)

print("\nConfusion Matrix:\n", conf_df)
