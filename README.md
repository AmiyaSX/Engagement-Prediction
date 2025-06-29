# Engagement-Prediction

This project aims to predict engagement levels from multimodal behavioral and physiological data, including transcripts, eye-tracking, speech pauses, and brain signals. 

It uses a combination of statistical features, time series methods, and machine learning models to train and evaluate engagement classification across various conditions.

## Modalities & Training

| **Modality**   | **Description**                                                                                           | **Location**          |
| -------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| **Embeddings** | Vector representations extracted from transcripts (e.g., using sentence embeddings or contextual models). | `scripts/embeddings/` |
| **Gaze**       | Eye-tracking features such as fixation duration, saccades, and gaze variability.                          | `scripts/gaze/`       |
| **Pauses**     | Features based on speech activity timing: production, comprehension, pauses, and turn-taking gaps.        | `scripts/pauses/`     |
| **ROI**        | Neural activity features derived from brain region signals (Regions of Interest).                         | `scripts/roi/`        |
| **Combined**   | Multimodal models combining two or more of the above modalities (e.g., ROI + Gaze).                       | `scripts/combine/`    |


## Evaluation

- Cross-validation is performed using Leave-One-Subject-Out strategy.

- Results (accuracy, classification reports, confusion matrices) are logged under results/cross_validation/.

- Feature importance for ROI-based models is saved under results/feature_importance/.

## Setup
1. Install dependencies

```
pip install -r requirements.txt
```

2. Prepare Data
 
Ensure all required data files (e.g., gaze, embeddings, ROI, pause data) are downloaded.

Details in preprocessing refer to scripts in scripts/preprocess/. (Data alignment and feature extraction)



## Project Structure

```
.
├── data    # Uploaded on OneDrive. Included some raw and preprocessed data
│   ├── mapping
│   │   └── conditions.csv      # Label mapping condition
│   └── raw
│       ├── brain
│       │   ├── aligned
│       │   │   ├── sub-05_roi_data.csv
│       │   │   ├── ...
│       │   ├── new
│       │   │   ├── raw
│       │   │   │   ├── sub-05_roi_data.csv
│       │   │   │   ├── ...
│       │   │   └── train   # use in training, added timestamps (new)
│       │   │       ├── sub-05_roi_data_with_timestamps.csv
│       │   │       ├── ...
│       │   └── train   # use in training, added timestamps (old)
│       │       ├── sub-05_roi_data_with_timestamps.csv
│       │       ├── ...
│       ├── eyetracking
│       │   ├── cleaned
│       │   │   ├── cleaned_sub-01_run-01_eye_tracking.csv
│       │   │   ├── ...
│       │   ├── feature
│       │   │   ├── mutual_gaze     # Aligned with 1.2s per sample
│       │   │   │   ├── sub-05_run-01_tracking_resampled.csv
│       │   │   │   ├── ...
│       │   │   ├── offsets
│       │   │   │   ├── sub-15_run-01_tracking_resampled.csv
│       │   │   │   ├── sub-15_run-02_tracking_resampled.csv
│       │   │   │   └── sub-15_run-03_tracking_resampled.csv
│       │   │   ├── sub-05_run-01_tracking_resampled.csv
│       │   │   ├── ...
│       │   ├── mini_sampled
│       │   │   ├── cleaned_sub-01_run-01_eye_tracking.csv
│       │   │   ├── ...
│       │   ├── sampled
│       │       ├── cleaned_sub-01_run-01_eye_tracking.csv
│       │       ├── ..
│       ├── pause
│       │   └── prod_comp_gaps_pauses_all_included.csv
│       ├── transcripts
│       │   ├── embeddings      # Real in use. Genimi generated embeddings
│       │   │   ├── sub-01_operator_run-01_embeddings.json
│       │   │   ├── ...
│       │   ├── ignore
│       │   │   ├── sub-01_operator_run-01.csv
│       │   │   ├── ...
│       │   ├── operator
│       │   │   ├── sub-05_operator_run-01.csv
│       │   │   ├── ...
│       │   └── participant
│       │       ├── sub-05_participant_run-01.csv
│       │       ├── ...
│       └── video
│           ├── origin  # Origin Video of subjects in use
│           │   ├── sub-05_run-01.mp4
│           │   ├── ...
│           ├── subjects_manually_calibrate     # Applied offsets to see the expected eye points
│           │   ├── gaze_overlay_sub-07_run-03.mp4
│           │   ├── ...
│           └── video_with_eyetracking          # Video visualizing the eye-tracking and head tracking
│               ├── gaze_overlay_sub-05_run-01.mp4
│               ├── ...
├── requirements.txt    # Python library requirements
└── src     # Source code
    ├── mapping
    │   └── conditions.csv  # Label mapping condition
    ├── results
    │   ├── 25-05-10_minirocket_sample20.txt    # dummy
    │   ├── ...
    │   ├── cross_validation       # all the results recorded here
    │   │   ├── 02-06-25_embeddings&roi&gaze.txt
    │   │   ├── ...
    │   ├── feature_importance  # Examine features contribute most
    │   │   ├── new_roi
    │   │   │   ├── roi_feature_importance_sub-[11].csv
    │   │   │   ├── ...
    │   │   └── roi
    │   │       ├── roi_feature_importance_sub-[11].csv
    │   │       ├── ...
    │   └── model.txt   # Hyperparameter tuning result
    └── scripts     # Training and evaluating scripts
        ├── combine     # Multimodal training
        │   ├── train_combine_gaze_embeddings_and_roi.py
        │   ├── train_combine_gaze_with_roi.py
        │   └── train_combine_roi_with_embeddings.py
        ├── embeddings  # Embeddings(transcripts) data training
        │   ├── train_embeddings_only_oper.py
        │   ├── train_embeddings_only_par.py
        │   ├── train_minirocket_embeddings+.py
        │   └── train_minirocket_embeddings.py
        ├── gaze    # Gaze data training
        │   ├── test_gaze_minirocket.py
        │   ├── train_minirocket_gaze.py
        │   └── train_svm.py
        ├── pauses  # Pauses and turn-taking data training
        │   └── train_prod_comp_gaps_pauses.py
        ├── preprocess   # To preprocess multimodal data and some visualization scripts
        │   ├── brain_data_add_timestamp.py
        │   ├── data_align.py
        │   ├── eyetracking_visualization.py
        │   ├── feature_extraction.py
        │   ├── gaze_feature_extraction.py
        │   ├── gaze_feature_extraction_allfile.py
        │   ├── gaze_feature_extractionv2.py
        │   └── sample_data.py
        └── roi     # brain data training
            ├── train_minirocket_roi.py
            ├── train_roi.py
            └── train_roi_new.py
```

