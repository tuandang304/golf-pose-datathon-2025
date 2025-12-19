from __future__ import annotations

from pathlib import Path

from golf_pose.logging_utils import setup_logging


def generate_reports(predictions_csv: Path, output_dir: Path) -> None:
    """
    Generate plots, statistics, and confusion matrix.
    """
    logger = setup_logging("reporting")
    logger.info("Generating reports from %s to %s", predictions_csv, output_dir)
    raise NotImplementedError("Module 7 implementation pending.")
