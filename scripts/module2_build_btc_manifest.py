from pathlib import Path
import argparse

from golf_pose.data.btc_manifest import build_btc_manifest
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 2: Build BTC manifest.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to BTC dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "btc_manifest.csv",
        help="Destination CSV for manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module2")
    logger.info("Starting Module 2 (BTC manifest).")
    build_btc_manifest(args.data_root, args.output)


if __name__ == "__main__":
    main()
