from pathlib import Path
import argparse

from golf_pose.data.golf_pose_manifest import build_golf_pose_manifest
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 1: Build Golf-Pose manifest.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to Golf-Pose dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "golf_pose_manifest.csv",
        help="Destination CSV for manifest.",
    )
    parser.add_argument("--view", type=str, default=None, help="Optional view filter (e.g., Back or Side).")
    parser.add_argument(
        "--include-good",
        action="store_true",
        help="Include good swings (default skips good swings).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module1")
    logger.info("Starting Module 1 (Golf-Pose manifest).")
    build_golf_pose_manifest(args.data_root, args.output, view=args.view, include_good=args.include_good)


if __name__ == "__main__":
    main()
