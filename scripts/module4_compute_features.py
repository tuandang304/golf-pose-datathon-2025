from pathlib import Path
import argparse

from golf_pose.features.engineer import compute_features
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 4: Feature engineering.")
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path("outputs") / "keypoints",
        help="Directory containing extracted keypoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "features" / "features.csv",
        help="Destination CSV for engineered features.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV to attach labels (e.g., data/golf_pose_manifest.csv).",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=5,
        help="Minimum valid frames required to keep a sample.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Number of temporal segments for feature aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module4")
    logger.info("Starting Module 4 (feature engineering).")
    compute_features(
        keypoints_dir=args.keypoints_dir,
        output_csv=args.output,
        manifest_csv=args.manifest,
        min_valid_frames=args.min_valid_frames,
        segments=args.segments,
    )


if __name__ == "__main__":
    main()
