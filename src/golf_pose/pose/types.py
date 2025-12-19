from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PoseSample:
    """
    Container for a single video's pose keypoints.
    """

    video_path: Path
    keypoints: np.ndarray  # shape: (frames, landmarks, 4) -> x, y, z, visibility
    fps: Optional[float] = None


@dataclass
class PoseMetadata:
    """
    Basic metadata for pose extraction outputs.
    """

    fps: float
    frame_count: int
    device: str
