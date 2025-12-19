from __future__ import annotations

from pathlib import Path
from typing import List

from golf_pose.data.manifest_builder import ManifestRecord, save_manifest
from golf_pose.io.files import list_videos
from golf_pose.io.video import get_video_metadata
from golf_pose.logging_utils import setup_logging


def _infer_environment(name: str) -> str:
    lowered = name.lower()
    if "indoor" in lowered or "trong" in lowered:
        return "Indoor"
    if "outdoor" in lowered or "ngo" in lowered:
        return "Outdoor"
    return name


def _infer_view(filename: str) -> str:
    lowered = filename.lower()
    if "back" in lowered:
        return "Back"
    if "side" in lowered:
        return "Side"
    return "Unknown"


def build_btc_manifest(data_root: Path, output_csv: Path) -> None:
    """
    Traverse BTC dataset and generate btc_manifest.csv.

    Expected fields: video_path, band_label, environment, view, fps, duration.
    """
    logger = setup_logging("btc_manifest")
    logger.info("Building BTC manifest at %s", output_csv)

    if not data_root.exists():
        raise FileNotFoundError(f"BTC data root not found: {data_root}")

    records: List[ManifestRecord] = []

    for env_dir in [p for p in data_root.iterdir() if p.is_dir()]:
        environment = _infer_environment(env_dir.name)
        for band_dir in [p for p in env_dir.iterdir() if p.is_dir()]:
            band_label = band_dir.name.strip()
            videos = list_videos(band_dir)
            if not videos:
                logger.debug("No videos in %s", band_dir)
                continue
            for video_path in videos:
                try:
                    fps, frame_count, duration = get_video_metadata(video_path)
                except FileNotFoundError:
                    logger.warning("Missing video: %s", video_path)
                    continue
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed metadata for %s: %s", video_path, exc)
                    fps, frame_count, duration = 0.0, 0, 0.0

                records.append(
                    ManifestRecord(
                        video_path=video_path,
                        label=band_label,
                        metadata={
                            "band_label": band_label,
                            "environment": environment,
                            "view": _infer_view(video_path.name),
                            "fps": fps,
                            "frame_count": frame_count,
                            "duration_sec": duration,
                        },
                    )
                )

    if not records:
        logger.warning("No BTC records found. Manifest will be empty: %s", output_csv)

    save_manifest(records, output_csv)
    logger.info("BTC manifest written: %s (%d rows)", output_csv, len(records))
