from pathlib import Path
import argparse

from golf_pose.pose.extractor import extract_keypoints
from golf_pose.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3: Pose extraction with MediaPipe.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to golf_pose_manifest.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "keypoints",
        help="Directory to store keypoint files.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None, help="Force device (default auto).")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Downsample to target FPS (default 30).")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Landmark visibility threshold to count a point as valid.",
    )
    parser.add_argument(
        "--min-visible-landmarks",
        type=int,
        default=8,
        help="A frame is valid if it has at least this many visible landmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("module3")
    logger.info("Starting Module 3 (pose extraction).")
    from golf_pose.pose.extractor import load_manifest_video_paths

    video_paths = load_manifest_video_paths(args.manifest)
    extract_keypoints(
        video_paths=video_paths,
        output_dir=args.output_dir,
        device=args.device,
        target_fps=args.target_fps,
        visibility_threshold=args.visibility_threshold,
        min_visible_landmarks=args.min_visible_landmarks,
    )


if __name__ == "__main__":
    main()
