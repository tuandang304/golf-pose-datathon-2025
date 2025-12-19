from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AppConfig:
    """
    Basic configuration container. Extend per module needs.
    """

    data_root: Optional[Path] = None
    outputs_dir: Optional[Path] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a JSON file. Returns defaults when path is None.
    """
    if path is None:
        return AppConfig()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return AppConfig(
        data_root=Path(raw["data_root"]) if raw.get("data_root") else None,
        outputs_dir=Path(raw["outputs_dir"]) if raw.get("outputs_dir") else None,
        extras=raw.get("extras", {}),
    )
