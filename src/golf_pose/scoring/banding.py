from __future__ import annotations

from typing import Any, Dict, List

from golf_pose.logging_utils import setup_logging

DEFAULT_THRESH = 0.3


def _map_band(error_score: float, error_count: int) -> str:
    """
    Map aggregated error to band bucket using error_count primarily.
    """
    if error_count >= 4:
        return "Band 1-2"
    if error_count == 3:
        return "Band 2-4"
    if error_count == 2:
        return "Band 4-6"
    if error_count == 1:
        return "Band 6-8"
    return "Band 8-10"


def score_and_band(
    error_probabilities: List[Dict[str, float]],
    threshold: float = DEFAULT_THRESH,
    max_prob_list: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Apply rule-based scoring and map to band classification.
    error_probabilities: list of dict{error_label: prob}
    max_prob_list: optional list of max probabilities per sample (for uncertainty)
    """
    logger = setup_logging("scoring")
    logger.info("Scoring %d samples", len(error_probabilities))
    results = []
    for idx, probs in enumerate(error_probabilities):
        # Aggregate: sum probabilities as error_score
        error_score = float(sum(probs.values()))
        error_count = sum(1 for p in probs.values() if p >= threshold)
        band = _map_band(error_score, error_count)

        # Use max_prob to push up band if model uncertain
        max_p = max(probs.values()) if probs else 0.0
        if max_prob_list is not None and idx < len(max_prob_list):
            max_p = max_prob_list[idx]

        if max_p < 0.4:
            band = "Band 8-10"
        elif max_p < 0.6 and band in {"Band 4-6", "Band 6-8"}:
            band = "Band 6-8"

        # Simple score 0-10: penalize by error_score
        score = max(0.0, 10.0 - error_score * 5.0)
        results.append(
            {
                "error_score": error_score,
                "error_count": error_count,
                "predicted_band": band,
                "score_0_10": score,
                "max_prob": max_p,
            }
        )
    return results
