import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Folder where per-fold CSVs are saved
# imp_dir = Path(".")
imp_dir = Path("feature_importance")
# Load and concatenate all importance files
imp_files = sorted(imp_dir.glob("imp_fold*.csv"))
dfs = [pd.read_csv(f) for f in imp_files]
imp_all = pd.concat(dfs, ignore_index=True)

# Average importance across folds
imp_mean = (
    imp_all.groupby("Feature", as_index=False)["Importance"]
    .mean()
    .sort_values("Importance", ascending=False)
)

# Optional: check variability (std) across folds
imp_stats = (
    imp_all.groupby("Feature")["Importance"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
    .reset_index()
)

# Save averaged importance
imp_stats.to_csv("feature_importance/feature_importance_mean_std.csv", index=False)

# Plot Top N averaged importances
top_n = 15
plt.figure(figsize=(8, 4))
plt.barh(
    imp_stats["Feature"].head(top_n)[::-1],
    imp_stats["mean"].head(top_n)[::-1],
)
plt.xlabel("Mean Importance Score")
plt.title(f"Top {top_n} ROI + Pauses Feature Importances (Averaged Across Folds)")
plt.tight_layout()
plt.show()
