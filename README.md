# Golf Swing Error-Driven Scoring Project

## Overview

This project implements a full pipeline to predict golf swing score bands from short videos, as per the DATATHON 2025 specification. Due to the constraint of extremely limited labeled data, this project follows a mandatory **error-driven approach**. Instead of training a single end-to-end model to predict a score, the system first learns to identify specific, individual swing errors from pose-based biomechanical features. A final score band is then assigned using a rule-based system derived from the detected errors.

This approach is more data-efficient, robust against small datasets, and provides a high degree of interpretability and explainability.

## Project Structure

The project is organized into a series of sequential Python scripts, each responsible for one step in the pipeline.

```
.
├── CUSTOM_DATASET/         # Input video data (not included in repo)
├── convert_metadata.py     # Step 1: Scans data and creates metadata CSVs.
├── extract_poses.py        # Step 2: Extracts MediaPipe poses for each video.
├── process_swings.py       # Step 3: Processes poses into static (address) or dynamic (phased) sets.
├── engineer_features.py    # Step 4: Calculates biomechanical features from poses.
├── train_error_models.py   # Step 5: Trains a One-Class SVM for each defined error type.
├── build_error_vectors.py  # Step 6: Uses trained models to generate an error vector for each video.
├── apply_scoring_rules.py  # Step 7: Applies rule-based logic to map error vectors to score bands.
├── validate.py             # Step 8: Validates predictions against a ground truth file.
├── generate_visualizations.py # Step 9: Creates plots for analysis and explainability.
├── requirements.txt        # All Python dependencies.
├── video_metadata.csv      # (Generated) Metadata for each video file.
├── error_metadata.csv      # (Generated) Metadata for each error type.
├── features.csv            # (Generated) Feature vectors for each video.
├── error_vectors.csv       # (Generated) Detected error flags for each video.
├── predictions.csv         # (Generated) Final score band predictions.
├── error_models/           # (Generated) Directory for serialized error detection models.
├── features/               # (Generated) Intermediate pose features.
├── pose_data/              # (Generated) Raw numpy arrays of pose data.
├── validation_results/     # (Generated) Validation metrics and plots.
└── visualizations/         # (Generated) Explainability plots.
```

## How to Run the Pipeline

1.  **Setup:** Place the video dataset into the `CUSTOM_DATASET` directory, following the specified folder structure.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute Scripts in Order:** Run the following scripts sequentially from your terminal. Each script processes the output of the previous one.

    ```bash
    python convert_metadata.py
    python extract_poses.py
    python process_swings.py
    python engineer_features.py
    python train_error_models.py
    python build_error_vectors.py
    python apply_scoring_rules.py
    ```

4.  **Validation & Visualization (Optional):**
    *   To validate, provide a `validation_ground_truth.csv` file and run:
      ```bash
      python validate.py
      ```
    *   To generate visualization plots:
      ```bash
      python generate_visualizations.py
      ```

## Technical Explanation

### Why Errors are Learned Instead of Scores

Directly training a model to predict a score (e.g., from 1 to 10) is a regression task that is notoriously difficult with small datasets. The model would likely overfit and fail to generalize. The **error-driven approach** circumvents this by breaking the problem down:

1.  **Data Efficiency:** Instead of learning a complex, continuous score, we train simple binary classifiers for well-defined technical errors (e.g., "Is there a `bad_knee_posture`?"). This requires far fewer examples per error type.
2.  **Interpretability:** This method is highly explainable. If a swing receives a low score, we can point to the exact technical errors that were detected (e.g., "Score is 4-6 because `bad_elbow_posture` and `bad_iron_wide_stance` were found"). This is impossible with a black-box regression model.
3.  **Robustness:** The final score is based on an aggregation of multiple independent signals. This makes the system more robust than relying on a single, monolithic model.

### The Validity of Static Videos

Many of the most critical swing faults occur during the **setup or address phase**, before the swing motion begins. These include fundamental errors like:
*   Incorrect posture (e.g., hunched back).
*   Improper distance from the ball.
*   Incorrect stance width.
*   Bad knee flexion.

The short, 1-2 second "static" videos are purpose-shot to isolate these specific setup flaws. Our pipeline correctly handles these by:
1.  Identifying them as 'static' based on their short duration.
2.  Extracting a single, representative pose that best captures the golfer's setup posture (using minimum joint variance to find the most stable frame).
3.  Calculating specific "static" biomechanical features from this pose.

By treating static videos as valid and distinct, we ensure that crucial setup errors are captured and factored into the final score.

### Alignment with Official Requirement

The scoring system is a direct and faithful implementation of the official DATATHON requirement. The core logic is as follows:

1.  **Error Detection:** The pipeline produces an `error_vector` for each swing, where each element indicates the presence or absence of a specific learned error.
2.  **Error Count:** We sum the number of detected errors to get a `Total Errors` count for the swing.
3.  **Rule-Based Mapping:** This `Total Errors` count is mapped directly to the score bands as specified in the official rules:

| Total Errors | Predicted Score Band |
|--------------|----------------------|
| ≥4           | 1–2                  |
| 3            | 2–4                  |
| 2            | 4–6                  |
| 1            | 6–8                  |
| 0            | 8–10                 |

This ensures that the project's output is fully compliant with the competition's scoring methodology. The code is also structured to easily accommodate weighted errors in the future, should a severity scale for different errors be provided.