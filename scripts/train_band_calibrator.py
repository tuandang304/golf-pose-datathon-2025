import argparse
from pathlib import Path

from golf_pose.scoring.calibrator import train_band_calibrator
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train band calibrator using BTC predictions (with true_band).")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs") / "predictions" / "btc_predictions.csv",
        help="CSV with predictions including true_band.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs") / "models",
        help="Directory to store calibrator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("train_band_calibrator")
    logger.info("Training band calibrator from %s", args.predictions)
    train_band_calibrator(args.predictions, args.model_dir)


if __name__ == "__main__":
    main()
