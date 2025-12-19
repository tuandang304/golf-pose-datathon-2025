from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from golf_pose.data.manifest_builder import ManifestRecord, save_manifest
from golf_pose.io.files import list_videos
from golf_pose.io.video import get_video_metadata
from golf_pose.logging_utils import setup_logging


def _normalize_view(name: str) -> str:
    lowered = name.lower()
    if "back" in lowered:
        return "Back"
    if "side" in lowered:
        return "Side"
    return name


def build_golf_pose_manifest(
    data_root: Path,
    output_csv: Path,
    view: Optional[str] = None,
    include_good: bool = False,
) -> None:
    """
    Traverse Golf-Pose dataset and generate golf_pose_manifest.csv.

    Expected fields: video_path, view, error_class, fps, frame_count.
    """
    logger = setup_logging("golf_pose_manifest")
    logger.info("Building Golf-Pose manifest at %s (view=%s, include_good=%s)", output_csv, view, include_good)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    view_filter = view.lower() if view else None
    records: List[ManifestRecord] = []

    for view_dir in [p for p in data_root.iterdir() if p.is_dir()]:
        normalized_view = _normalize_view(view_dir.name)
        if view_filter and normalized_view.lower() != view_filter:
            logger.debug("Skipping view %s (filtered)", normalized_view)
            continue

        # Prefer an explicit "Bad Swings" folder; otherwise use view_dir directly.
        bad_root = view_dir / "Bad Swings"
        if not bad_root.is_dir():
            logger.warning("Bad Swings folder missing under %s; using %s directly.", view_dir, view_dir)
            bad_root = view_dir

        for error_dir in [p for p in bad_root.iterdir() if p.is_dir()]:
            name_lower = error_dir.name.lower()
            if "good" in name_lower:
                # Skip good swings by default
                if include_good:
                    pass
                else:
                    logger.debug("Skipping good swings folder: %s", error_dir)
                    continue

            error_class = error_dir.name
            video_files = list_videos(error_dir)
            if not video_files:
                logger.debug("No videos found in %s", error_dir)
                continue

            for video_path in video_files:
                try:
                    fps, frame_count, duration = get_video_metadata(video_path)
                except FileNotFoundError:
                    logger.warning("Could not open video (missing): %s", video_path)
                    continue
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed metadata for %s: %s", video_path, exc)
                    fps, frame_count, duration = 0.0, 0, 0.0

                records.append(
                    ManifestRecord(
                        video_path=video_path,
                        label=error_class,
                        metadata={
                            "view": normalized_view,
                            "error_class": error_class,
                            "fps": fps,
                            "frame_count": frame_count,
                            "duration_sec": duration,
                            "is_good": "good" in name_lower,
                        },
                    )
                )

    if not records:
        logger.warning("No records found. Manifest will be empty: %s", output_csv)

    save_manifest(records, output_csv)
    logger.info("Manifest written: %s (%d rows)", output_csv, len(records))
