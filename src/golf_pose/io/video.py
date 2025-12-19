from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2


def get_video_metadata(path: Path) -> Tuple[float, int, float]:
    """
    Return (fps, frame_count, duration_seconds) for the given video.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps: float = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_seconds: float = frame_count / fps if fps > 0 else 0.0
    capture.release()
    return fps, frame_count, duration_seconds


def is_video_file(path: Path) -> bool:
    """
    Simple check for common video extensions.
    """
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
