from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from golf_pose.logging_utils import setup_logging


CALIB_FEATURES = ["error_score", "error_count", "max_prob", "pose_quality"]


def train_band_calibrator(predictions_csv: Path, model_dir: Path) -> None:
    """
    Train a small calibrator to map error_score/error_count/max_prob/pose_quality to band labels.
    Uses BTC predictions with true_band available.
    """
    logger = setup_logging("band_calibrator")
    frame = pd.read_csv(predictions_csv)
    if "true_band" not in frame.columns:
        raise ValueError("Predictions CSV must contain 'true_band' to train calibrator.")

    data = frame.dropna(subset=["true_band"]).copy()
    if data.empty:
        raise ValueError("No rows with true_band found for calibration.")

    X = data[CALIB_FEATURES].fillna(0.0)
    y = data["true_band"].astype(str)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    multi_class="auto",
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    y_pred = pipe.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "band_calibrator.joblib")

    logger.info("Calibrator trained and saved to %s", model_dir / "band_calibrator.joblib")
    logger.info("Classes: %s", sorted(y.unique().tolist()))
    with (model_dir / "band_calibrator_report.txt").open("w", encoding="utf-8") as f:
        f.write(pd.DataFrame(report).to_string())


def load_band_calibrator(model_dir: Path):
    path = model_dir / "band_calibrator.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
