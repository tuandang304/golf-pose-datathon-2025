from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from golf_pose.logging_utils import setup_logging


HIP_L = 23
HIP_R = 24
SHOULDER_L = 11
SHOULDER_R = 12


def _load_manifest_labels(
    manifest_csv: Optional[Path],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    if manifest_csv is None:
        return {}, {}, {}
    frame = pd.read_csv(manifest_csv)
    by_path: Dict[str, Dict[str, str]] = {}
    by_stem: Dict[str, Dict[str, str]] = {}
    by_name: Dict[str, Dict[str, str]] = {}
    for _, row in frame.iterrows():
        info = {
            "label": row.get("label", ""),
            "error_class": row.get("error_class", row.get("label", "")),
            "view": row.get("view", ""),
            "band_label": row.get("band_label", ""),
            "environment": row.get("environment", ""),
        }
        full_path = str(row["video_path"])
        by_path[full_path] = info
        by_stem[Path(full_path).stem] = info
        by_name[Path(full_path).name] = info
    return by_path, by_stem, by_name


def _normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Center at mid-hip and scale by shoulder distance.
    keypoints: (T, 33, 4)
    """
    coords = keypoints[..., :3]  # ignore visibility for normalization
    hips = coords[:, [HIP_L, HIP_R], :]
    shoulders = coords[:, [SHOULDER_L, SHOULDER_R], :]

    mid_hip = np.nanmean(hips, axis=1)  # (T, 3)
    shoulder_dist = np.linalg.norm(shoulders[:, 0, :] - shoulders[:, 1, :], axis=1)  # (T,)

    # Avoid zeros
    shoulder_dist[shoulder_dist == 0] = np.nan

    normed = coords - mid_hip[:, None, :]
    normed = normed / shoulder_dist[:, None, None]
    return normed


def _compute_velocity(normed: np.ndarray) -> np.ndarray:
    """
    Velocity magnitude between consecutive frames. Shape: (T-1,)
    """
    diffs = np.diff(normed, axis=0)
    vel = np.linalg.norm(diffs, axis=2)  # (T-1, landmarks)
    # Aggregate per-frame magnitude
    return np.nanmean(vel, axis=1)


def _segment_indices(total_frames: int, segments: int) -> List[Tuple[int, int]]:
    segs = []
    if total_frames <= 0 or segments <= 0:
        return segs
    step = total_frames / segments
    for i in range(segments):
        start = int(round(i * step))
        end = int(round((i + 1) * step))
        segs.append((start, max(end, start + 1)))
    return segs


def _agg_features(normed: np.ndarray, vel: np.ndarray, segments: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    # Global stats
    for axis, name in enumerate(["x", "y", "z"]):
        vals = normed[..., axis].reshape(-1)
        feats[f"global_{name}_mean"] = float(np.nanmean(vals))
        feats[f"global_{name}_std"] = float(np.nanstd(vals))
        feats[f"global_{name}_min"] = float(np.nanmin(vals))
        feats[f"global_{name}_max"] = float(np.nanmax(vals))

    feats["vel_mean"] = float(np.nanmean(vel)) if vel.size else 0.0
    feats["vel_std"] = float(np.nanstd(vel)) if vel.size else 0.0
    feats["vel_max"] = float(np.nanmax(vel)) if vel.size else 0.0

    # Segment stats
    seg_idxs = _segment_indices(normed.shape[0], segments)
    for idx, (s, e) in enumerate(seg_idxs):
        seg = normed[s:e]
        seg_vel = vel[s : max(s, e - 1)] if vel.size else np.array([])
        prefix = f"seg{idx+1}"
        for axis, name in enumerate(["x", "y", "z"]):
            vals = seg[..., axis].reshape(-1)
            feats[f"{prefix}_{name}_mean"] = float(np.nanmean(vals))
            feats[f"{prefix}_{name}_std"] = float(np.nanstd(vals))
        feats[f"{prefix}_vel_mean"] = float(np.nanmean(seg_vel)) if seg_vel.size else 0.0
        feats[f"{prefix}_vel_std"] = float(np.nanstd(seg_vel)) if seg_vel.size else 0.0
        feats[f"{prefix}_vel_max"] = float(np.nanmax(seg_vel)) if seg_vel.size else 0.0

    return feats


def compute_features(
    keypoints_dir: Path,
    output_csv: Path,
    manifest_csv: Optional[Path] = None,
    min_valid_frames: int = 5,
    segments: int = 3,
) -> None:
    """
    Normalize keypoints and compute features with temporal segments.
    """
    logger = setup_logging("feature_engineering")
    logger.info(
        "Computing features from %s to %s (segments=%d, min_valid_frames=%d)",
        keypoints_dir,
        output_csv,
        segments,
        min_valid_frames,
    )

    labels_by_path, labels_by_stem, labels_by_name = _load_manifest_labels(manifest_csv)
    rows: List[Dict[str, float]] = []

    files = sorted(keypoints_dir.glob("*.npz"))
    for npz_path in files:
        data = np.load(npz_path)
        keypoints = data["keypoints"]  # (T, 33, 4)
        fps = float(data.get("fps", 0.0))
        valid_ratio = float(data.get("valid_ratio", 0.0))
        total_frames = int(data.get("total_frames", keypoints.shape[0]))

        # Filter valid frames
        valid_mask = ~np.isnan(keypoints[..., 0])
        valid_frames = valid_mask.any(axis=1)
        valid_count = int(valid_frames.sum())
        if valid_count < min_valid_frames:
            logger.warning("Skipping %s: not enough valid frames (%d)", npz_path.name, valid_count)
            continue

        normed = _normalize_keypoints(keypoints)
        vel = _compute_velocity(normed)
        feats = _agg_features(normed, vel, segments=segments)

        video_path_str = data.files  # not stored; infer from filename
        feats["feature_file"] = npz_path.name
        feats["fps"] = fps
        feats["valid_ratio"] = valid_ratio
        feats["total_frames"] = total_frames

        # Recover video path hint and label mapping
        feats["video_path"] = npz_path.stem.replace("__", "/")
        label_info = None
        # Try exact path mapping (unlikely to match) then stem-based mapping.
        label_info = labels_by_path.get(feats["video_path"])
        if label_info is None:
            base = Path(npz_path.stem).stem  # drop .mp4/.mov
            label_info = labels_by_stem.get(base)
        if label_info is None:
            # Try matching by original filename (last token after __)
            parts = npz_path.stem.split("__")
            if parts:
                filename = parts[-1]
                label_info = labels_by_name.get(filename)
                if label_info is None:
                    label_info = labels_by_stem.get(Path(filename).stem)
        if label_info:
            feats.update(label_info)

        rows.append(feats)

    frame = pd.DataFrame(rows)
    # One-hot view if present
    if "view" in frame.columns:
        view_dummies = pd.get_dummies(frame["view"], prefix="view")
        frame = pd.concat([frame.drop(columns=["view"]), view_dummies], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    logger.info("Features written: %s (%d rows, %d cols)", output_csv, len(frame), len(frame.columns))
