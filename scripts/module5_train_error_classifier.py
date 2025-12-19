from pathlib import Path
import argparse

from golf_pose.models.trainer import train_error_classifier
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 5: Train error classifier.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs") / "features" / "features.csv",
        help="Path to features CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs") / "models",
        help="Directory to store trained model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-grouped", action="store_true", help="Use grouped labels instead of full 20 classes.")
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Disable oversampling on train split.",
    )
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for logistic regression.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module5")
    logger.info("Starting Module 5 (train classifier).")
    train_error_classifier(
        args.features,
        args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        use_grouped=args.use_grouped,
        oversample=not args.no_oversample,
        C=args.C,
    )


if __name__ == "__main__":
    main()
