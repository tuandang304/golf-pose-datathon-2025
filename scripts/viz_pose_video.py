import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

from golf_pose.logging_utils import setup_logging


def build_npz_name(video_path: Path) -> str:
    safe_name = "__".join(video_path.parts[-3:]).replace(" ", "_")
    return f"{safe_name}.npz"


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    visibility_thresh: float = 0.5,
    draw_bbox: bool = False,
    bbox_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    h, w = frame.shape[:2]
    coords = []
    for x, y, z, vis in keypoints:
        if np.isnan(x) or np.isnan(y):
            coords.append(None)
            continue
        if vis < visibility_thresh:
            coords.append(None)
            continue
        coords.append((int(x * w), int(y * h)))

    # Draw joints
    for pt in coords:
        if pt is None:
            continue
        cv2.circle(frame, pt, 3, (0, 255, 0), -1)

    # Draw connections
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        pa, pb = coords[a], coords[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, pa, pb, (0, 128, 255), 2)

    if draw_bbox:
        valid_pts = [pt for pt in coords if pt is not None]
        if valid_pts:
            xs, ys = zip(*valid_pts)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    return frame


def _load_prediction(pred_csv: Path, video_path: Path) -> Dict[str, str]:
    frame = pd.read_csv(pred_csv)
    # Try exact match
    match = frame[frame["video_path"].astype(str) == str(video_path)]
    if match.empty:
        # Fallback: match by basename
        match = frame[frame["video_path"].astype(str).str.contains(video_path.stem, regex=False)]
    if match.empty:
        return {}
    row = match.iloc[0]
    info = {
        "pred_label": str(row.get("pred_label", "")),
        "predicted_band": str(row.get("predicted_band", "")),
        "top_errors": str(row.get("top_errors", "")),
        "score_0_10": str(row.get("score_0_10", "")),
    }
    return info


def visualize(video_path: Path, keypoints_dir: Path, output_path: Path, pred_info: Dict[str, str] | None = None) -> None:
    logger = setup_logging("viz_pose")
    npz_file = keypoints_dir / build_npz_name(video_path)
    if not npz_file.exists():
        # Fallback: search by stem containment
        candidates = list(keypoints_dir.glob(f"*{video_path.stem}*.npz"))
        if candidates:
            npz_file = candidates[0]
        else:
            raise FileNotFoundError(f"Keypoint file not found: {npz_file}")

    data = np.load(npz_file)
    kps = data["keypoints"]  # (T,33,4)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(kps)
    limit = min(len(kps), total_frames)

    for idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_skeleton(frame, kps[idx], draw_bbox=True)
        if pred_info:
            text = f"{pred_info.get('pred_label','')} | {pred_info.get('predicted_band','')}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info("Saved visualization to %s (frames=%d, fps=%.1f)", output_path, limit, fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pose keypoints overlayed on video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to source video.")
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path("outputs") / "keypoints_btc",
        help="Directory containing keypoint npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/visuals/<video_stem>_viz.mp4)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions CSV to overlay errors/band.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.output
    if out_path is None:
        out_path = Path("outputs") / "visuals" / f"{args.video.stem}_viz.mp4"
    pred_info = None
    if args.predictions and args.predictions.exists():
        pred_info = _load_prediction(args.predictions, args.video)
    visualize(args.video, args.keypoints_dir, out_path, pred_info=pred_info)


if __name__ == "__main__":
    main()
