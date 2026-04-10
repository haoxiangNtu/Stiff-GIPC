"""Utility helpers for path resolution and math conversions."""

from __future__ import annotations

import math
import os
from pathlib import Path


def resolve_asset_path(path: str, assets_dir: str = "") -> str:
    """Resolve a relative asset path against the StiffGIPC assets directory."""
    if os.path.isabs(path):
        return path

    if assets_dir:
        candidate = os.path.join(assets_dir, path)
        if os.path.exists(candidate):
            return candidate

    project_root = Path(__file__).parent.parent
    for base in [project_root / "Assets", project_root]:
        candidate = base / path
        if candidate.exists():
            return str(candidate)

    return path


def deg2rad(degrees: float) -> float:
    return math.radians(degrees)


def rad2deg(radians: float) -> float:
    return math.degrees(radians)
