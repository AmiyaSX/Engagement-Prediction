# Combines pause features + ROI features on a common 12s/5s window grid and trains a multimodal classifier.

import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scipy.stats import skew, kurtosis, entropy
from numpy.fft import fft

# -----------------------------
# CONFIG
# -----------------------------
# Pause events CSV (long format)
PAUSE_PATH = "data/raw/pause/prod_comp_gaps_pauses.csv" 
# ROI time series (with 1.2s timestamps)
ROI_GLOB  = "data/raw/brain/new/train/*_with_timestamps.csv"  # SWTA version
# Fallback (contrast-based) columns if SWTA not present
ROI_COLS_SWTA = ["swta_comp_roi", "swta_dmn_roi", "swta_prod_roi"]
ROI_COLS_CONTRAST = ["comp_vs_prod_combined", "dmn_medial_tpj", "prod_vs_comp_combined"]

# Mapping file (subject,run -> eng_level)
MAPPING_CSV = "data/mapping/conditions.csv"

# Common window grid (seconds)
WIN_SIZE_S = 12.0      # 12 s
STEP_S     = 5.0       # 5 s stride (used by pauses)
TR_S       = 1.2       # fMRI TR
ROI_WIN_TR = 10        # 10 TR = 12 s
ROI_STEP_TR = 5        # 5 TR = 6 s

RANDOM_STATE = 42
N_JOBS = 4

# -----------------------------
# HELPERS
# -----------------------------

def make_window_grid(max_time, win=WIN_SIZE_S, step=STEP_S):
    """Return list of window starts (seconds) from 0 to max_time - win with given step."""
    if max_time < win:
        return []
    return np.arange(0.0, max_time - win + 1e-9, step)

def round_to_grid(t, step=STEP_S):
    """Snap a time value to the lower multiple of step (e.g., 17.3 -> 15 if step=5)."""
    return np.floor((t + 1e-9) / step) * step

# -----------------------------
# 1) PAUSE FEATURES
# -----------------------------

PAUSE_EVENT_TYPES = [
    "production",      # participant speaking
    "comprehension",   # confederate speaking
    "pause_p",         # pause within participant's turn
    "pause_c",         # pause within confederate's turn
    "gap_p2c",         # transitional gap P->C
    "gap_c2p",         # transitional gap C->P
]

def extract_pause_features_window(window, total_window_time=WIN_SIZE_S):
    """Extract stats per event type for a window (12s). Returns a flat dict of ~18*6 features."""
    feat = {}
    for ev in PAUSE_EVENT_TYPES:
        ev_data = window[window["event_type"] == ev]
        durations = ev_data["duration"].values
        start_times = ev_data["start_time"].values

        # counts + total
        feat[f"count_{ev}"] = len(durations)
        total_dur = float(durations.sum()) if len(durations) else 0.0
        feat[f"duration_{ev}"] = total_dur

        if len(durations) > 0:
            feat[f"mean_duration_{ev}"]   = float(durations.mean())
            feat[f"std_duration_{ev}"]    = float(durations.std())
            feat[f"max_duration_{ev}"]    = float(durations.max())
            feat[f"min_duration_{ev}"]    = float(durations.min())
            feat[f"median_duration_{ev}"] = float(np.median(durations))
            feat[f"rate_{ev}"]            = len(durations) / total_window_time
            feat[f"proportion_time_{ev}"] = total_dur / total_window_time

            if len(start_times) > 1:
                gaps = np.diff(start_times)
                feat[f"gap_before_mean_{ev}"] = float(gaps.mean())
                feat[f"gap_before_std_{ev}"]  = float(gaps.std())
            else:
                feat[f"gap_before_mean_{ev}"] = 0.0
                feat[f"gap_before_std_{ev}"]  = 0.0
        else:
            # fill
            for stat in [
                "mean_duration", "std_duration", "max_duration", "min_duration",
                "median_duration", "rate", "proportion_time", "gap_before_mean", "gap_before_std"
            ]:
                feat[f"{stat}_{ev}"] = 0.0
    return feat

def build_pause_feature_table():
    """Return a DataFrame of pause features per (subject, run, window_start), with label."""
    df = pd.read_csv(PAUSE_PATH)
    df["subject"] = df["filename"].apply(lambda x: int(x.split("_")[0].split("-")[1]))
    df["run"]     = df["filename"].apply(lambda x: int(x.split("_")[1].split("-")[1]))

    # Load mapping to labels (low/normal/high)
    mapping = pd.read_csv(MAPPING_CSV)
    mapping.columns = mapping.columns.str.strip()
    mapping["EngagementLevel"] = mapping["eng_level"].str.lower().map(
        {"low": "low", "normal": "normal", "high": "high"}
    )

    rows = []
    for (subj, run), group in tqdm(df.groupby(["subject", "run"]), desc="Pauses: windows"):
        max_time = float(group["start_time"].max() + group["duration"].max())
        for wstart in make_window_grid(max_time, WIN_SIZE_S, STEP_S):
            wend = wstart + WIN_SIZE_S
            win_df = group[(group["start_time"] >= wstart) & (group["start_time"] < wend)]
            if win_df.empty:
                continue

            # label consistency within window
            labels = win_df["eng_level"].unique()
            if len(labels) != 1:
                continue
            label = labels[0].lower()

            feats = extract_pause_features_window(win_df)
            feats.update({"subject": subj, "run": run, "window_start": float(wstart), "label": label})
            rows.append(feats)

    pause_feat_df = pd.DataFrame(rows)
    # Attach canonical label from mapping if needed
    return pause_feat_df

# -----------------------------
# 2) ROI FEATURES
# -----------------------------

def roi_window_features(window):
    """
    Extract per-ROI features (stats, derivatives, energy/entropy, temporal, FFT, Hjorth).
    window: np.ndarray shape (n_rois, n_samples)
    returns list of features concatenated across ROIs
    """
    feats = []
    for roi_signal in window:
        roi_signal = np.asarray(roi_signal, dtype=float)
        # basic
        mean_val = roi_signal.mean()
        std_val  = roi_signal.std()
        min_val  = roi_signal.min()
        max_val  = roi_signal.max()
        range_val = max_val - min_val
        skew_val = skew(roi_signal) if len(roi_signal) > 2 else 0.0
        kurt_val = kurtosis(roi_signal) if len(roi_signal) > 3 else 0.0

        # derivatives
        d1 = np.diff(roi_signal)
        d2 = np.diff(d1) if len(d1) > 0 else np.array([0.0])
        slope, intercept = np.polyfit(np.arange(len(roi_signal)), roi_signal, 1) if len(roi_signal) > 1 else (0.0, roi_signal[0] if len(roi_signal) else 0.0)

        # energy & entropy
        energy = float(np.sum(roi_signal**2))
        hist, _ = np.histogram(roi_signal, bins=10)
        ent  = float(entropy(hist + 1))

        # temporal
        zcr = int(((roi_signal[:-1] * roi_signal[1:]) < 0).sum()) if len(roi_signal) > 1 else 0
        autoc = float(np.corrcoef(roi_signal[:-1], roi_signal[1:])[0,1]) if len(roi_signal) > 1 else 0.0
        slope_changes = int(np.sum(np.diff(np.sign(d1)) != 0)) if len(d1) > 1 else 0

        # frequency
        fft_coeffs = np.abs(fft(roi_signal))
        fft_mean = float(fft_coeffs.mean())
        fft_std  = float(fft_coeffs.std())
        fft_power = float(np.sum(fft_coeffs**2))
        if len(fft_coeffs) > 1:
            peak_idx = int(np.argmax(fft_coeffs[1:]) + 1)
            peak_freq_norm = peak_idx / len(fft_coeffs)
        else:
            peak_freq_norm = 0.0

        # Hjorth
        var0 = np.var(roi_signal)
        var1 = np.var(d1) if len(d1) > 0 else 0.0
        var2 = np.var(d2) if len(d2) > 0 else 0.0
        mobility = np.sqrt(var1 / var0) if var0 > 0 else 0.0
        complexity = (np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0.0

        feats.extend([
            mean_val, std_val, min_val, max_val, range_val, skew_val, kurt_val,
            float(d1.mean()) if len(d1) else 0.0,
            float(d2.mean()) if len(d2) else 0.0,
            float(slope), float(intercept),
            energy, ent, float(zcr), autoc,
            fft_mean, fft_std, fft_power, peak_freq_norm,
            mobility, complexity, float(slope_changes),
        ])
    return feats

def build_roi_feature_table():
    """
    Build ROI feature table per (subject, run, window_start).
    Detects SWTA columns first, otherwise falls back to contrast columns.
    Windows: 10 TR (12s) with 5 TR stride (6s).
    """
    rows = []
    for file in tqdm(glob.glob(ROI_GLOB), desc="ROI: windows"):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        fname = os.path.basename(file)

        # subject id from filename e.g. sub-05_roi_data_with_timestamps.csv
        try:
            subject_id = int(fname.split("_")[0].split("-")[1])
        except Exception:
            print(f"Skip (subject parse failed): {fname}")
            continue

        # detect columns
        if all(c in df.columns for c in ROI_COLS_SWTA):
            roi_cols = ROI_COLS_SWTA
        elif all(c in df.columns for c in ROI_COLS_CONTRAST):
            roi_cols = ROI_COLS_CONTRAST
        else:
            print(f"Skip (ROI columns not found) in {fname}")
            continue

        if "run_index" not in df.columns or "timestamp" not in df.columns:
            print(f"Skip (missing run_index/timestamp) in {fname}")
            continue

        # z-score per subject (per file)
        df[roi_cols] = (df[roi_cols] - df[roi_cols].mean()) / df[roi_cols].std()

        for run_id_str, g in df.groupby("run_index"):
            try:
                run_id = int(run_id_str.strip().split("-")[1])
            except Exception:
                print(f"Skip (run parse failed) in {fname} -> {run_id_str}")
                continue

            ts = g["timestamp"].to_numpy()  # seconds, step = 1.2
            # ensure sorted
            order = np.argsort(ts)
            g = g.iloc[order]
            ts = ts[order]

            feats_mat = g[roi_cols].to_numpy().T  # (n_roi, n_samples)

            n_samples = feats_mat.shape[1]
            # step/size in TRs
            for i in range(0, n_samples - ROI_WIN_TR + 1, ROI_STEP_TR):
                start_time = float(ts[i])
                end_time   = float(ts[i + ROI_WIN_TR - 1])
                window = feats_mat[:, i:i+ROI_WIN_TR]

                f = roi_window_features(window)
                # snap start to 5s grid to match pause features
                wstart_grid = float(round_to_grid(start_time, STEP_S))
                rows.append({
                    "subject": subject_id,
                    "run": run_id,
                    "window_start": wstart_grid,
                    **{f"roi_f_{k}": v for k, v in enumerate(f)}
                })

    roi_feat_df = pd.DataFrame(rows)
    return roi_feat_df

# -----------------------------
# 3) MERGE + TRAIN
# -----------------------------

def main():
    # Pause feature table (with labels)
    pause_df = build_pause_feature_table()
    if pause_df.empty:
        raise RuntimeError("No pause features produced.")

    # Load canonical labels mapping and attach (sanity)
    mapping = pd.read_csv(MAPPING_CSV)
    mapping.columns = mapping.columns.str.strip()
    mapping["EngagementLevel"] = mapping["eng_level"].str.lower().map(
        {"low": "low", "normal": "normal", "high": "high"}
    )

    # ROI feature table
    roi_df = build_roi_feature_table()
    if roi_df.empty:
        raise RuntimeError("No ROI features produced.")

    # Merge on (subject, run, window_start)
    mm = pd.merge(
        pause_df,
        roi_df,
        on=["subject", "run", "window_start"],
        how="inner",
        validate="many_to_one"  # multiple pause windows may merge to one ROI window-start; adjust if needed
    )

    # Prepare X/y/groups
    y = mm["label"].astype(str).str.lower()
    groups = mm["subject"].astype(int)

    # Drop non-features
    drop_cols = ["label"]
    X = mm.drop(columns=drop_cols)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Classifier
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    pipeline = Pipeline([("classifier", clf)])

    # LOGO CV
    logo = LeaveOneGroupOut()
    accuracies = []
    all_true, all_pred = [], []

    print(f"\n=== Multimodal (Pauses + ROI) | N={len(X)} windows, D={X.shape[1]} features ===\n")
    for fold, (tr, te) in enumerate(logo.split(X, y_enc, groups), 1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y_enc[tr], y_enc[te]

        w = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=w)

        y_hat = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        accuracies.append(acc)
        all_true.extend(y_test)
        all_pred.extend(y_hat)

        print(f"Fold {fold:02d} | Test subject(s): {np.unique(groups.iloc[te])} | Acc: {acc:.4f}")

        # Save per-fold importance
        feature_names = X.columns.tolist()
        importances = pipeline.named_steps["classifier"].feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
                    .sort_values("importance", ascending=False)
        os.makedirs("feature_importance", exist_ok=True)
        imp_df.to_csv(f"feature_importance/imp_fold_{fold:02d}.csv", index=False)

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}  (± {np.std(accuracies):.4f})")
    print("Classification Report:")
    print(classification_report(all_true, all_pred, target_names=le.classes_))

    cm = confusion_matrix(all_true, all_pred)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in le.classes_],
                             columns=[f"Pred {c}" for c in le.classes_])
    print("\nConfusion Matrix:")
    print(cm_df)

if __name__ == "__main__":
    main()
