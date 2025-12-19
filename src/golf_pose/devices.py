from __future__ import annotations

from typing import Literal

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore

Device = Literal["cpu", "gpu"]


def select_device(prefer_gpu: bool = True) -> Device:
    """
    Select CPU or GPU. Prefers GPU when available via OpenCV CUDA.
    """
    if prefer_gpu and cv2 is not None:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if isinstance(count, int) and count > 0:
            return "gpu"
    return "cpu"
