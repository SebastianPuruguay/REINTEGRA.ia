from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def format_timestamp(seconds: float, include_tenths: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds % 60
    if include_tenths:
        return f"{minutes:02d}:{remaining:04.1f}"
    return f"{minutes:02d}:{int(remaining):02d}"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
