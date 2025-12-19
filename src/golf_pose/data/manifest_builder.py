from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from golf_pose.io.files import ensure_parent


@dataclass
class ManifestRecord:
    """
    Generic manifest record holder.
    """

    video_path: Path
    label: str
    metadata: Dict[str, Any]


def save_manifest(records: List[ManifestRecord], output_csv: Path) -> None:
    """
    Save manifest records to CSV.
    """
    ensure_parent(output_csv)
    rows = []
    for record in records:
        row = {"video_path": str(record.video_path), "label": record.label}
        row.update(record.metadata)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
