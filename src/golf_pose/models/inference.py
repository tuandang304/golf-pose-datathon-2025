from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd

from golf_pose.logging_utils import setup_logging
from golf_pose.scoring.banding import score_and_band
from golf_pose.scoring.calibrator import load_band_calibrator


def _load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "classifier.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    labels = joblib.load(model_dir / "labels.joblib")
    feature_cols = joblib.load(model_dir / "feature_cols.joblib")
    return model, scaler, labels, feature_cols


def _select_predictions(proba: np.ndarray, labels: List[str], top_k: int = 3, threshold: float = 0.3) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for row in proba:
        indexed = list(zip(labels, row))
        indexed.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for idx, (label, score) in enumerate(indexed):
            if idx < top_k or score >= threshold:
                selected.append((label, float(score)))
        results.append(dict(selected))
    return results


def run_inference(
    features_csv: Path,
    model_dir: Path,
    output_csv: Path,
    device: str | None = None,
    top_k: int = 3,
    threshold: float = 0.3,
) -> None:
    """
    Run classifier on features; save predictions with error probabilities and top errors.
    """
    logger = setup_logging("btc_inference")
    logger.info(
        "Running inference with features %s to %s (top_k=%d, threshold=%.2f)",
        features_csv,
        output_csv,
        top_k,
        threshold,
    )

    model, scaler, labels, feature_cols = _load_artifacts(model_dir)
    frame = pd.read_csv(features_csv)

    X = frame[feature_cols].values
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    max_prob_list = proba.max(axis=1).tolist()
    top_errors = _select_predictions(proba, labels, top_k=top_k, threshold=threshold)
    scores = score_and_band(top_errors, threshold=threshold, max_prob_list=max_prob_list)

    calibrator = load_band_calibrator(model_dir)
    cal_band = None
    if calibrator is not None:
        calib_df = pd.DataFrame(
            {
                "error_score": [s["error_score"] for s in scores],
                "error_count": [s["error_count"] for s in scores],
                "max_prob": max_prob_list,
                "pose_quality": frame.get("valid_ratio", pd.Series([np.nan] * len(frame))).fillna(0.0),
            }
        )
        cal_band = calibrator.predict(calib_df)

    # Map true band if available in input features
    true_band_map = None
    if "band_label" in frame.columns:
        true_band_map = frame["band_label"].tolist()

    out_rows = []
    for idx, row in frame.iterrows():
        score_info = scores[idx]
        out = {
            "video_path": row.get("video_path", ""),
            "pred_label": preds[idx],
            "top_errors": top_errors[idx],
            "predicted_band_rule": score_info["predicted_band"],
            "predicted_band": cal_band[idx] if cal_band is not None else score_info["predicted_band"],
            "score_0_10": score_info["score_0_10"],
            "error_score": score_info["error_score"],
            "error_count": score_info["error_count"],
            "pose_quality": row.get("valid_ratio", None),
            "max_prob": score_info.get("max_prob", None),
        }
        if true_band_map is not None:
            out["true_band"] = true_band_map[idx]
        out_rows.append(out)

    out_frame = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_csv, index=False)
    logger.info("Predictions written: %s (%d rows)", output_csv, len(out_frame))
