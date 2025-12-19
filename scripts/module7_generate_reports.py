from pathlib import Path
import argparse

from golf_pose.reporting.reporter import generate_reports
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 7: Reporting.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs") / "predictions" / "btc_predictions.csv",
        help="Path to BTC predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "reports",
        help="Directory to store generated reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module7")
    logger.info("Starting Module 7 (reporting).")
    generate_reports(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
