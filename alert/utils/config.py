from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Deep-merge YAML files left-to-right. Later files override earlier ones."""
    merged: dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(data)}: {path}")
        merged = _deep_merge(merged, data)
    return merged
