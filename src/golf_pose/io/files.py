from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .video import is_video_file


def list_videos(root: Path) -> List[Path]:
    """
    Recursively list video files under a root directory.
    """
    return [p for p in root.rglob("*") if p.is_file() and is_video_file(p)]


def ensure_parent(path: Path) -> None:
    """
    Create parent directory for a file path if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
