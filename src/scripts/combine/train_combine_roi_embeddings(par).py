# Combines ROI engineered features + participant embeddings (par) for engagement prediction
# on a common 12s/STEP_SIZE seconds grid (LOGO CV).

import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import google.generativeai as genai
from google.api_core import retry

from scipy.stats import skew, kurtosis, entropy
from numpy.fft import fft

# === CONFIGURATION ===
WINDOW_SIZE = 12  # seconds
STEP_SIZE = 5  # seconds (set to 6 if you want 12s/6s grid)
EMBEDDING_DIM = 3072
BUFFER = 30  # seconds (look-back for embeddings)
TR_S = 1.2  # fMRI TR (seconds)

# Gemini (only used if you decide to generate embeddings; this script reads cached JSONs)
genai.configure(api_key="My key")

# Paths
MAPPING_CSV = "data/mapping/conditions.csv"
ROI_GLOB = "data/raw/brain/new/train/*_with_timestamps.csv"
ROI_COLS = ["swta_comp_roi", "swta_dmn_roi", "swta_prod_roi"]

PART_DIR = "data/raw/transcripts/participant"
EMB_CACHE_DIR = "data/raw/transcripts/embeddings"

# Load mapping + label map (run-level)
mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df.columns = mapping_df.columns.str.strip()
mapping_df["EngagementLevel"] = (
    mapping_df["eng_level"]
    .astype(str)
    .str.lower()
    .map({"low": "low", "normal": "normal", "high": "high"})
)

label_map = (
    mapping_df.assign(
        subject=mapping_df["subject_id"].astype(int),
        run=mapping_df["run"].astype(int),
    )
    .set_index(["subject", "run"])["EngagementLevel"]
    .astype(str)
    .str.lower()
    .to_dict()
)


# -------------------------
# OPTIONAL: embedding generation helper (not used if cache exists)
# -------------------------
@retry.Retry(timeout=300.0)
def get_embedding(text: str):
    result = genai.embed_content(
        model="gemini-embedding-exp-03-07",
        content=text,
        task_type="classification",
    )
    return result["embedding"]


# -------------------------
# ROI FEATURE EXTRACTION (your function)
# -------------------------
def roi_window_features(window):
    """
    Extract per-ROI features (stats, derivatives, energy/entropy, temporal, FFT, Hjorth).
    window: np.ndarray shape (n_rois, n_samples)
    returns list of features concatenated across ROIs
    """
    feats = []
    num_rois = window.shape[0]
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

    # --- CROSS-ROI FEATURES ---
    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            x = window[i].astype(float)
            y = window[j].astype(float)

            Xf = np.fft.rfft(x)
            Yf = np.fft.rfft(y)
            Sxy = Xf * np.conj(Yf)
            Sxx = (Xf * np.conj(Xf)).real + 1e-8
            Syy = (Yf * np.conj(Yf)).real + 1e-8
            coherence = float(np.mean((np.abs(Sxy) ** 2) / (Sxx * Syy)))

            max_xcorr = 0.0
            if np.std(x) > 0 and np.std(y) > 0 and len(x) > 4:
                for lag in (-2, -1, 0, 1, 2):
                    if lag < 0:
                        xs, ys = x[:lag], y[-lag:]
                    elif lag > 0:
                        xs, ys = x[lag:], y[:-lag]
                    else:
                        xs, ys = x, y
                    if len(xs) > 1:
                        r = float(np.corrcoef(xs, ys)[0, 1])
                        if np.isfinite(r):
                            max_xcorr = max(max_xcorr, r)

            feats.extend([coherence, max_xcorr])

    return feats


# -------------------------
# HELPERS
# -------------------------
def extract_key_from_filename(file_name: str) -> str:
    """
    Assumes: sub-01_..._run-02.csv
    Returns: "sub-01_run-02"
    """
    parts = file_name.split("_")
    subject = parts[0]  # sub-01
    run = parts[2].split("-")[1].split(".")[0]  # 02
    return f"{subject}_run-{run}"


def make_window_grid(max_time: float, win: float, step: float):
    if max_time < win:
        return []
    return np.arange(0.0, max_time - win + 1e-9, step)


def safe_mean_embedding(df_win: pd.DataFrame, dim: int):
    if df_win.empty:
        return np.zeros(dim, dtype=float)
    try:
        emb = np.vstack(df_win["embedding"].values).astype(float)
        return np.mean(emb, axis=0)
    except Exception:
        return np.zeros(dim, dtype=float)


# -------------------------
# TRANSCRIPTS (participant only; cached embeddings)
# -------------------------
def load_participant_with_cached_embeddings(dir_path: str):
    all_data = []
    for file_path in glob.glob(f"{dir_path}/*.csv"):
        filename = os.path.basename(file_path)

        try:
            subject_id = int(filename.split("_")[0].split("-")[1])
            run_id = int(filename.split("_")[2].replace(".csv", "").split("-")[1])
        except Exception:
            continue

        label = label_map.get((subject_id, run_id))
        if not label:
            continue

        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            continue

        cache_path = os.path.join(
            EMB_CACHE_DIR, f"{filename.replace('.csv','')}_embeddings.json"
        )
        if not os.path.exists(cache_path):
            # If you want to compute when missing, uncomment:
            # df["embedding"] = df["transcription"].fillna("").apply(get_embedding)
            # os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # with open(cache_path, "w") as f:
            #     json.dump(df["embedding"].tolist(), f)
            continue

        with open(cache_path, "r") as f:
            emb = json.load(f)
        if len(emb) != len(df):
            continue

        df["embedding"] = emb
        df["subject_id"] = subject_id
        df["run_id"] = run_id
        df["start_sec"] = df["start_time"].astype(float)
        df["end_sec"] = df["end_time"].astype(float)
        df["eng_level"] = str(label).lower()
        df["file"] = filename
        df["match_key"] = extract_key_from_filename(filename)

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# -------------------------
# ROI: CONSISTENT FEATURE TABLE ON SECONDS GRID
# -------------------------
def load_brain_features_on_grid(
    roi_glob=ROI_GLOB,
    roi_cols=ROI_COLS,
    win_size_s=WINDOW_SIZE,
    step_s=STEP_SIZE,
    tr_s=TR_S,
):
    """
    Builds engineered ROI features (roi_window_features) on the same seconds grid as embeddings.
    Returns a feature table with exact keys (key, win_start, win_end).
    """
    roi_win_tr = int(round(win_size_s / tr_s))  # 12/1.2=10 TR
    rows = []

    for file in glob.glob(roi_glob):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        fname = os.path.basename(file)

        try:
            subject_id = int(fname.split("_")[0].split("-")[1])
        except Exception:
            continue

        if "run_index" not in df.columns or "timestamp" not in df.columns:
            continue
        if not all(c in df.columns for c in roi_cols):
            continue

        # z-score within file (matches your other fusion scripts)
        df[list(roi_cols)] = (df[list(roi_cols)] - df[list(roi_cols)].mean()) / df[
            list(roi_cols)
        ].std()

        for run_id_str, g in df.groupby("run_index"):
            try:
                run_id = int(str(run_id_str).split("-")[1])
            except Exception:
                continue

            label = label_map.get((subject_id, run_id))
            if not label:
                continue
            label = str(label).lower()

            g = g.sort_values("timestamp")
            ts = g["timestamp"].to_numpy(dtype=float)
            feats_mat = g[list(roi_cols)].to_numpy(dtype=float).T  # (n_rois, T)

            if ts.size == 0:
                continue

            max_time = float(ts.max())
            for win_start in make_window_grid(max_time, win=win_size_s, step=step_s):
                win_end = win_start + win_size_s

                # first TR at/after win_start
                i = int(np.searchsorted(ts, win_start, side="left"))
                if i + roi_win_tr > feats_mat.shape[1]:
                    continue

                window = feats_mat[:, i : i + roi_win_tr]  # (n_rois, 10 TR)
                f = roi_window_features(window)

                rows.append(
                    {
                        "key": f"sub-{subject_id:02d}_run-{run_id:02d}",
                        "win_start": round(float(win_start), 1),
                        "win_end": round(float(win_end), 1),
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "label": label,
                        **{f"roi_f_{k}": float(v) for k, v in enumerate(f)},
                    }
                )

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df

    # Deduplicate (mean agg)
    feat_df = feat_df.groupby(
        ["key", "win_start", "win_end", "subject_id", "run_id", "label"], as_index=False
    ).mean()

    return feat_df


# -------------------------
# MAIN
# -------------------------
def main():
    print("Loading participant transcripts with cached embeddings...")
    participant_df = load_participant_with_cached_embeddings(PART_DIR)
    if participant_df.empty:
        raise RuntimeError(
            "Participant transcript data missing. Check embedding cache JSONs exist."
        )

    print("Loading ROI engineered features on the same 12s/STEP_SIZE grid...")
    roi_feat_df = load_brain_features_on_grid()
    if roi_feat_df.empty:
        raise RuntimeError("No ROI features generated. Check ROI files/columns.")

    roi_cols_feat = [c for c in roi_feat_df.columns if c.startswith("roi_f_")]

    # Lookup: (key, win_start, win_end) -> (roi_vec, label, subject_id)
    roi_index = {}
    for _, r in roi_feat_df.iterrows():
        roi_index[(r["key"], r["win_start"], r["win_end"])] = (
            r[roi_cols_feat].to_numpy(dtype=float),
            str(r["label"]).lower(),
            int(r["subject_id"]),
        )

    X_combined, y_combined, groups_combined = [], [], []

    print("Building fused windows (ROI + participant emb)...")
    for key, part in tqdm(participant_df.groupby("match_key"), desc="Fusing"):
        max_time = float(part["end_sec"].max())

        for win_start in np.arange(0.0, max_time - WINDOW_SIZE + 1e-9, STEP_SIZE):
            win_end = win_start + WINDOW_SIZE
            buffer_start = max(0.0, win_start - BUFFER)

            part_win = part[
                (part["start_sec"] >= buffer_start) & (part["start_sec"] <= win_end)
            ]
            part_emb = safe_mean_embedding(part_win, EMBEDDING_DIM)

            roi_key = (key, round(float(win_start), 1), round(float(win_end), 1))
            roi_pack = roi_index.get(roi_key)
            if roi_pack is None:
                continue

            roi_vec, label, subject_id = roi_pack
            combined_feat = np.concatenate([roi_vec, part_emb], axis=0)

            X_combined.append(combined_feat)
            y_combined.append(label)
            groups_combined.append(subject_id)

    if len(X_combined) == 0:
        raise RuntimeError(
            "No valid combined samples found. Check grid alignment and ROI coverage."
        )

    X_combined = np.stack(X_combined).astype(float)
    y_combined = np.array(y_combined, dtype=str)
    groups_combined = np.array(groups_combined, dtype=int)

    print(f"[INFO] Final samples: {len(X_combined)}")
    print(
        f"[INFO] Feature dim: {X_combined.shape[1]} (ROI={len(roi_cols_feat)} + part={EMBEDDING_DIM})"
    )

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
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

    print("\n=== Training & Evaluating (LOGO CV) ===")
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_combined, y_encoded, groups_combined), 1
    ):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        w = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=w)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        accs.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(
            f"Fold {fold:02d} | Subject(s): {np.unique(groups_combined[test_idx])} | Acc: {acc:.4f}"
        )

    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(accs):.4f}  (± {np.std(accs):.4f})")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
