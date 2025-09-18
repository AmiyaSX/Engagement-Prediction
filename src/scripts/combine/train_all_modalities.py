# Combines pause features + ROI features on a common 12s/5s window grid and trains a multimodal classifier.

import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from numpy.fft import fft
import google.generativeai as genai
from google.api_core import retry

# === CONFIGURATION ===
WINDOW_SIZE = 12
STEP_SIZE = 5
EMBEDDING_DIM = 3072
BUFFER = 30

genai.configure(api_key="My key")

mapping_df = pd.read_csv("data/mapping/conditions.csv")
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)


# -----------------------------
# CONFIG
# -----------------------------
# Pause events CSV (long format)
PAUSE_PATH = "data/raw/pause/prod_comp_gaps_pauses.csv"
# ROI time series (with 1.2s timestamps)
ROI_GLOB = "data/raw/brain/new/train/*_with_timestamps.csv"  # SWTA version
ROI_GLOB_CONTRAST = "data/raw/brain/train/*_with_timestamps.csv"  # CONTRAST version
# Fallback (contrast-based) columns if SWTA not present
ROI_COLS_SWTA = ["swta_comp_roi", "swta_dmn_roi", "swta_prod_roi"]
ROI_COLS_CONTRAST = ["comp_vs_prod_combined", "dmn_medial_tpj", "prod_vs_comp_combined"]

# Mapping file (subject,run -> eng_level)
MAPPING_CSV = "data/mapping/conditions.csv"

# Common window grid (seconds)
WIN_SIZE_S = 12.0  # 12 s
STEP_S = 5.0  # 5 s stride (used by pauses)
TR_S = 1.2  # fMRI TR
ROI_WIN_TR = 10  # 10 TR = 12 s
ROI_STEP_TR = 5  # 5 TR = 6 s

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
    "production",  # participant speaking
    "comprehension",  # confederate speaking
    "pause_p",  # pause within participant's turn
    "pause_c",  # pause within confederate's turn
    "gap_p2c",  # transitional gap P->C
    "gap_c2p",  # transitional gap C->P
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
            feat[f"mean_duration_{ev}"] = float(durations.mean())
            feat[f"std_duration_{ev}"] = float(durations.std())
            feat[f"max_duration_{ev}"] = float(durations.max())
            feat[f"min_duration_{ev}"] = float(durations.min())
            feat[f"median_duration_{ev}"] = float(np.median(durations))
            feat[f"rate_{ev}"] = len(durations) / total_window_time
            feat[f"proportion_time_{ev}"] = total_dur / total_window_time

            if len(start_times) > 1:
                gaps = np.diff(start_times)
                feat[f"gap_before_mean_{ev}"] = float(gaps.mean())
                feat[f"gap_before_std_{ev}"] = float(gaps.std())
            else:
                feat[f"gap_before_mean_{ev}"] = 0.0
                feat[f"gap_before_std_{ev}"] = 0.0
        else:
            # fill
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
                feat[f"{stat}_{ev}"] = 0.0
    return feat


def build_pause_feature_table():
    """Return a DataFrame of pause features per (subject, run, window_start), with label."""
    df = pd.read_csv(PAUSE_PATH)
    df["subject"] = df["filename"].apply(lambda x: int(x.split("_")[0].split("-")[1]))
    df["run"] = df["filename"].apply(lambda x: int(x.split("_")[1].split("-")[1]))

    # Load mapping to labels (low/normal/high)
    mapping = pd.read_csv(MAPPING_CSV)
    mapping.columns = mapping.columns.str.strip()
    mapping["EngagementLevel"] = (
        mapping["eng_level"]
        .str.lower()
        .map({"low": "low", "normal": "normal", "high": "high"})
    )

    rows = []
    for (subj, run), group in tqdm(
        df.groupby(["subject", "run"]), desc="Pauses: windows"
    ):
        max_time = float(group["start_time"].max() + group["duration"].max())
        for wstart in make_window_grid(max_time, WIN_SIZE_S, STEP_S):
            wend = wstart + WIN_SIZE_S
            win_df = group[
                (group["start_time"] >= wstart) & (group["start_time"] < wend)
            ]
            if win_df.empty:
                continue

            # label consistency within window
            labels = win_df["eng_level"].unique()
            if len(labels) != 1:
                continue
            label = labels[0].lower()

            feats = extract_pause_features_window(win_df)
            feats.update(
                {
                    "subject": subj,
                    "run": run,
                    "window_start": float(wstart),
                    "label": label,
                }
            )
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
        std_val = roi_signal.std()
        min_val = roi_signal.min()
        max_val = roi_signal.max()
        range_val = max_val - min_val
        skew_val = skew(roi_signal) if len(roi_signal) > 2 else 0.0
        kurt_val = kurtosis(roi_signal) if len(roi_signal) > 3 else 0.0

        # derivatives
        d1 = np.diff(roi_signal)
        d2 = np.diff(d1) if len(d1) > 0 else np.array([0.0])
        slope, intercept = (
            np.polyfit(np.arange(len(roi_signal)), roi_signal, 1)
            if len(roi_signal) > 1
            else (0.0, roi_signal[0] if len(roi_signal) else 0.0)
        )

        # energy & entropy
        energy = float(np.sum(roi_signal**2))
        hist, _ = np.histogram(roi_signal, bins=10)
        ent = float(entropy(hist + 1))

        # temporal
        zcr = (
            int(((roi_signal[:-1] * roi_signal[1:]) < 0).sum())
            if len(roi_signal) > 1
            else 0
        )
        autoc = (
            float(np.corrcoef(roi_signal[:-1], roi_signal[1:])[0, 1])
            if len(roi_signal) > 1
            else 0.0
        )
        slope_changes = int(np.sum(np.diff(np.sign(d1)) != 0)) if len(d1) > 1 else 0

        # frequency
        fft_coeffs = np.abs(fft(roi_signal))
        fft_mean = float(fft_coeffs.mean())
        fft_std = float(fft_coeffs.std())
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
        complexity = (
            (np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0.0
        )

        feats.extend(
            [
                mean_val,
                std_val,
                min_val,
                max_val,
                range_val,
                skew_val,
                kurt_val,
                float(d1.mean()) if len(d1) else 0.0,
                float(d2.mean()) if len(d2) else 0.0,
                float(slope),
                float(intercept),
                energy,
                ent,
                float(zcr),
                autoc,
                fft_mean,
                fft_std,
                fft_power,
                peak_freq_norm,
                mobility,
                complexity,
                float(slope_changes),
            ]
        )
    return feats


# Combines pause features + both ROI versions (SWTA and v1) on a common 12s/5s window grid and trains a multimodal classifier.

# -----------------------------
# 2) ROI FEATURES
# -----------------------------


def build_roi_feature_table():
    """
    Build ROI feature table by merging features from two versions:
    - SWTA (ROI_GLOB)
    - Contrast-based (ROI_GLOB_CONSTRAST)
    Merges per (subject, run, window_start) and prefixes v1 features.
    """

    def load_roi_files(file_glob, expected_cols, prefix):
        rows = []
        for file in tqdm(glob.glob(file_glob), desc=f"ROI: {prefix}"):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            fname = os.path.basename(file)

            try:
                subject_id = int(fname.split("_")[0].split("-")[1])
            except Exception:
                continue

            if not all(c in df.columns for c in expected_cols):
                continue

            if "run_index" not in df.columns or "timestamp" not in df.columns:
                continue

            df[expected_cols] = (df[expected_cols] - df[expected_cols].mean()) / df[
                expected_cols
            ].std()

            for run_id_str, g in df.groupby("run_index"):
                try:
                    run_id = int(run_id_str.strip().split("-")[1])
                except Exception:
                    continue

                ts = g["timestamp"].to_numpy()
                order = np.argsort(ts)
                g = g.iloc[order]
                ts = ts[order]
                feats_mat = g[expected_cols].to_numpy().T

                n_samples = feats_mat.shape[1]
                for i in range(0, n_samples - ROI_WIN_TR + 1, ROI_STEP_TR):
                    start_time = float(ts[i])
                    window = feats_mat[:, i : i + ROI_WIN_TR]
                    f = roi_window_features(window)
                    wstart_grid = float(round_to_grid(start_time, STEP_S))
                    rows.append(
                        {
                            "subject": subject_id,
                            "run": run_id,
                            "window_start": wstart_grid,
                            **{f"{prefix}_f_{k}": v for k, v in enumerate(f)},
                        }
                    )
        return pd.DataFrame(rows)

    roi_df_swta = load_roi_files(ROI_GLOB, ROI_COLS_SWTA, "swta")
    roi_df_v1 = load_roi_files(ROI_GLOB_CONTRAST, ROI_COLS_CONTRAST, "comp_prod")

    if roi_df_swta.empty or roi_df_v1.empty:
        raise RuntimeError("One of the ROI versions is empty.")

    merged = pd.merge(
        roi_df_swta,
        roi_df_v1,
        on=["subject", "run", "window_start"],
        how="inner",
        validate="one_to_one",
    )
    return merged

# Combines pause features + both ROI versions (SWTA and v1) into pipeline A,
# and Gemini embeddings (from transcripts) into pipeline B.
# Final prediction is based on late fusion of predictions from both pipelines.

# Imports and CONFIG remain unchanged (not repeated here)

# [...] KEEP ALL YOUR EXISTING FUNCTIONS


# === ADDITIONAL TRANSCRIPT + EMBEDDING LOADER ===
def load_transcript_embeddings(dir_path):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)
        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception:
            continue
        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            continue

        cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
        if not os.path.exists(cache_path):
            continue

        with open(cache_path, "r") as f:
            embeddings = json.load(f)
        if len(embeddings) != len(df):
            continue
        # Get label
        label_row = mapping_df[
            (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
        ]
        if label_row.empty:
            print(f"[WARNING] No label for subject {subject_id}, run {run_id}")
            continue
        label = label_row["EngagementLevel"].values[0]
        df["embedding"] = embeddings
        df["subject"] = subject_id
        df["run"] = run_id
        df["start"] = df["start_time"]
        df["end"] = df["end_time"]
        df["mid"] = (df["start"] + df["end"]) / 2
        df["file"] = filename
        df["eng_level"] = label
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Combines pause features + both ROI versions (SWTA and v1) into pipeline A,
# and Gemini embeddings (from participant and operator transcripts) into pipeline B.
# Final prediction is based on early fusion of all features.

# [ALL IMPORTS REMAIN UNCHANGED]

# [...] KEEP ALL YOUR EXISTING FUNCTIONS


# === MODIFIED TRANSCRIPT EMBEDDING LOADER FOR BOTH PARTICIPANT & OPERATOR ===
def load_dual_embeddings(participant_dir, operator_dir):
    def load_embeddings(dir_path, speaker_tag):
        all_data = []
        for file_path in glob.glob(f"{dir_path}/*.csv"):
            filename = os.path.basename(file_path)
            try:
                subject_id = int(filename.split("_")[0].split("-")[1])
                run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
            except Exception:
                continue
            df = pd.read_csv(file_path)
            if "transcription" not in df.columns:
                continue

            cache_path = f"data/raw/transcripts/embeddings/{filename.replace('.csv', '')}_embeddings.json"
            if not os.path.exists(cache_path):
                continue

            with open(cache_path, "r") as f:
                embeddings = json.load(f)
            if len(embeddings) != len(df):
                continue
            label_row = mapping_df[
                (mapping_df["subject_id"] == subject_id) & (mapping_df["run"] == run_id)
            ]
            if label_row.empty:
                continue
            label = label_row["EngagementLevel"].values[0]
            df["embedding"] = embeddings
            df["subject"] = subject_id
            df["run"] = run_id
            df["start"] = df["start_time"]
            df["end"] = df["end_time"]
            df["speaker"] = speaker_tag
            df["eng_level"] = label
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    part_df = load_embeddings(participant_dir, "participant")
    op_df = load_embeddings(operator_dir, "operator")
    return pd.concat([part_df, op_df], ignore_index=True)


def main():
    pause_df = build_pause_feature_table()
    roi_df = build_roi_feature_table()
    text_df = load_dual_embeddings(
        "data/raw/transcripts/participant", "data/raw/transcripts/operator"
    )

    if pause_df.empty or roi_df.empty or text_df.empty:
        raise RuntimeError("Missing data for fusion model")

    mm = pd.merge(pause_df, roi_df, on=["subject", "run", "window_start"], how="inner")
    y_a = mm["label"].str.lower()
    groups = mm["subject"]
    X_a = mm.drop(columns=["label", "subject", "run", "window_start"])

    X_b, y_b, meta_b = [], [], []
    for (subj, run), group in text_df.groupby(["subject", "run"]):
        max_t = group["end"].max()
        for start in make_window_grid(max_t):
            end = start + WIN_SIZE_S
            buffer_start = max(0, start - BUFFER)
            win = group[(group["start"] >= buffer_start) & (group["start"] < end)]
            if win.empty:
                continue
            try:
                emb = np.mean(np.vstack(win["embedding"].values), axis=0)
            except:
                emb = np.zeros(EMBEDDING_DIM)
            label = win["eng_level"].iloc[0].lower()
            X_b.append(emb)
            y_b.append(label)
            meta_b.append((subj, run, float(round_to_grid(start))))

    df_b = pd.DataFrame(meta_b, columns=["subject", "run", "window_start"])
    df_b["label"] = y_b
    df_b["X_b"] = X_b
    df_b["key"] = list(zip(df_b["subject"], df_b["run"], df_b["window_start"]))
    df_b = df_b.set_index("key")

    mm["key"] = list(zip(mm["subject"], mm["run"], mm["window_start"]))

    X_a_final, X_b_final, y_final, groups_final = [], [], [], []
    for idx, row in mm.iterrows():
        key = row["key"]
        if key not in df_b.index:
            continue
        x_b = df_b.loc[[key], "X_b"].values[0]
        X_a_final.append(X_a.loc[idx].values)
        X_b_final.append(x_b)
        y_final.append(row["label"])
        groups_final.append(row["subject"])

    print(f"[INFO] Final samples: {len(X_a_final)}")
    X_a_final = np.stack(X_a_final)
    X_b_final = np.stack(X_b_final)
    X_final = np.stack(
        [np.concatenate([x_a, x_b]) for x_a, x_b in zip(X_a_final, X_b_final)]
    )
    y_final = np.array(y_final)
    groups_final = np.array(groups_final)

    print(f"[INFO] Fused input shape: {len(X_final)}")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_final)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softmax",
                    num_class=len(le.classes_),
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )

    logo = LeaveOneGroupOut()
    accs, all_y_true, all_y_pred = [], [], []

    print("=== Training & Evaluating (LOGO CV) ===")
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_final, y_encoded, groups_final), 1
    ):
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        w = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=w)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(
            f"Fold {fold:02d} | Subject(s): {np.unique(groups_final[test_idx])} | Acc: {acc:.4f}"
        )

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accs):.4f}  (± {np.std(accs):.4f})")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in le.classes_],
        columns=[f"Pred {c}" for c in le.classes_],
    )
    print("\nConfusion Matrix:")
    print(cm_df)


if __name__ == "__main__":
    main()
