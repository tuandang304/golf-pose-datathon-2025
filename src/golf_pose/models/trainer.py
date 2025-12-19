from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from golf_pose.logging_utils import setup_logging


GROUP_MAP = {
    "Bad Ball Position Too Far": "Ball Position",
    "Bad Ball Position Too Near": "Ball Position",
    "Bad Far From Ball": "Ball Position",
    "Bad Near To Ball": "Ball Position",
    "Bad Chip Narrow Stance": "Chip Stance",
    "Bad Chip Wide Stance": "Chip Stance",
    "Bad Iron Narrow Stance": "Iron Stance",
    "Bad Iron Wide Stance": "Iron Stance",
    "Bad Putting Narrow Stance": "Putting Stance",
    "Bad Putting Wide Stance": "Putting Stance",
    "Bad Putting Posture Hunched": "Putting Posture",
    "Bad Putting Posture Straight": "Putting Posture",
    "Bad Elbow Posture Backswing": "Elbow Posture",
    "Bad Elbow Posture Frontswing": "Elbow Posture",
    "Bad Knee Posture": "Knee Posture",
    "Bad Chin Position": "Chin Position",
    "Bad Chipping Swings": "Chipping Swing",
    "Bad Driver Swings": "Driver Swing",
    "Bad Iron Swings": "Iron Swing",
    "Bad Sandpit Swings": "Sandpit Swing",
}


def _map_labels(label_series: pd.Series) -> pd.Series:
    return label_series.astype(str).map(lambda x: GROUP_MAP.get(x, x))


def _load_data(features_csv: Path, use_grouped: bool) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    frame = pd.read_csv(features_csv)
    if "label" not in frame.columns:
        raise ValueError("Features CSV must contain 'label' column from manifest.")
    frame = frame.dropna(subset=["label"]).copy()
    if use_grouped:
        frame["target_label"] = _map_labels(frame["label"])
    else:
        frame["target_label"] = frame["label"].astype(str)
    y = frame["target_label"].astype(str).values
    numeric_frame = frame.select_dtypes(include=[np.number])
    feature_cols = numeric_frame.columns.tolist()
    X = numeric_frame.values
    return X, y, feature_cols, frame


def train_error_classifier(
    features_csv: Path,
    model_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    use_grouped: bool = False,
    oversample: bool = True,
    C: float = 1.0,
) -> None:
    """
    Train classifier on Golf-Pose features; save model, scaler, label mapping, and report.
    """
    logger = setup_logging("train_error_classifier")
    logger.info(
        "Training error classifier using %s (grouped=%s, oversample=%s, C=%.2f)",
        features_csv,
        use_grouped,
        oversample,
        C,
    )

    X, y, feature_cols, frame = _load_data(features_csv, use_grouped)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if oversample:
        ros = RandomOverSampler(random_state=random_state)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        multi_class="auto",
        C=C,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)

    report = classification_report(y_val, y_pred, output_dict=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "classifier.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(model.classes_.tolist(), model_dir / "labels.joblib")
    joblib.dump(feature_cols, model_dir / "feature_cols.joblib")

    with (model_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Model saved to %s", model_dir)
    logger.info("Labels: %s", model.classes_.tolist())
