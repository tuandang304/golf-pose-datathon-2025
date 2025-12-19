from pathlib import Path
import argparse

from golf_pose.models.inference import run_inference
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 6: BTC inference + scoring.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs") / "features" / "features.csv",
        help="Path to BTC features CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs") / "models",
        help="Directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "predictions" / "btc_predictions.csv",
        help="Destination CSV for predictions.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None, help="Force device (default auto).")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K errors to include regardless of threshold.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold to keep an error.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module6")
    logger.info("Starting Module 6 (BTC inference).")
    run_inference(
        args.features,
        args.model_dir,
        args.output,
        device=args.device,
        top_k=args.top_k,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
