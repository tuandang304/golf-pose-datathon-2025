from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from golf_pose.devices import select_device
from golf_pose.logging_utils import setup_logging
from golf_pose.paths import ensure_outputs


def _output_path(output_dir: Path, video_path: Path) -> Path:
    safe_name = "__".join(video_path.parts[-3:]).replace(" ", "_")
    return output_dir / f"{safe_name}.npz"


def _frame_skip(fps: float, target_fps: Optional[float]) -> int:
    if not target_fps or fps <= 0:
        return 1
    interval = max(int(round(fps / target_fps)), 1)
    return interval


def _landmarks_to_array(landmarks) -> np.ndarray:
    coords = []
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(coords, dtype=np.float32)


def extract_keypoints(
    video_paths: Iterable[Path],
    output_dir: Path,
    device: str | None = None,
    target_fps: Optional[float] = 30.0,
    visibility_threshold: float = 0.5,
    min_visible_landmarks: int = 8,
) -> None:
    """
    Extract pose keypoints for each video and save to output_dir as .npz files.

    - target_fps: downsample frames to this fps (skip-based). None to keep original.
    - visibility_threshold: min visibility to count a landmark as valid.
    - min_visible_landmarks: frame is valid if >= this many landmarks visible.
    """
    ensure_outputs()
    logger = setup_logging("pose_extractor")
    resolved_device = device or select_device()
    logger.info(
        "Starting pose extraction to %s (device=%s, target_fps=%s, visibility>=%.2f, min_visible=%d)",
        output_dir,
        resolved_device,
        target_fps,
        visibility_threshold,
        min_visible_landmarks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        for video_path in video_paths:
            out_path = _output_path(output_dir, video_path)
            if out_path.exists():
                logger.info("Skip existing keypoints: %s", out_path.name)
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning("Cannot open video: %s", video_path)
                continue

            orig_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            frame_interval = _frame_skip(orig_fps, target_fps)
            frames = []
            valid_frames = 0
            total_frames = 0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_interval > 1 and (frame_idx % frame_interval != 0):
                    frame_idx += 1
                    continue
                frame_idx += 1
                total_frames += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    arr = _landmarks_to_array(results.pose_landmarks)
                    visible = (arr[:, 3] >= visibility_threshold).sum()
                    if visible >= min_visible_landmarks:
                        valid_frames += 1
                    frames.append(arr)
                else:
                    frames.append(np.full((33, 4), np.nan, dtype=np.float32))

            cap.release()

            if not frames:
                logger.warning("No frames processed for %s", video_path)
                continue

            keypoints = np.stack(frames, axis=0)  # (T, 33, 4)
            valid_ratio = valid_frames / float(total_frames) if total_frames else 0.0
            used_fps = target_fps or orig_fps

            np.savez_compressed(
                out_path,
                keypoints=keypoints,
                fps=used_fps,
                original_fps=orig_fps,
                total_frames=total_frames,
                valid_frames=valid_frames,
                valid_ratio=valid_ratio,
                device=resolved_device,
            )
            logger.info(
                "Saved keypoints %s | frames=%d valid=%.2f fps=%.1f (orig %.1f)",
                out_path.name,
                keypoints.shape[0],
                valid_ratio,
                used_fps,
                orig_fps,
            )


def load_manifest_video_paths(manifest_csv: Path, col: str = "video_path") -> Iterable[Path]:
    """
    Load video paths from manifest CSV.
    """
    frame = pd.read_csv(manifest_csv)
    if col not in frame.columns:
        raise ValueError(f"Column {col} not in manifest")
    return [Path(p) for p in frame[col].tolist()]
