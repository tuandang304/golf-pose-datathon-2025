import argparse
from pathlib import Path
import pandas as pd
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch visualize pose overlays for a manifest of videos.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV (e.g., btc_manifest.csv).")
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path("outputs") / "keypoints_btc",
        help="Directory containing keypoint npz files.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs") / "predictions" / "btc_predictions.csv",
        help="Predictions CSV to overlay errors/bands.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "visuals",
        help="Directory to save rendered videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in frame.iterrows():
        video_path = Path(row["video_path"])
        out_path = args.output_dir / f"{video_path.stem}_viz.mp4"
        cmd = [
            sys.executable,
            "scripts/viz_pose_video.py",
            "--video",
            str(video_path),
            "--keypoints-dir",
            str(args.keypoints_dir),
            "--output",
            str(out_path),
            "--predictions",
            str(args.predictions),
        ]
        try:
            print("Rendering:", str(video_path))
        except Exception:
            pass
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
