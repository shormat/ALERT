"""Dataset and output roots resolved from env vars with repo-relative defaults."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_env_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _resolve(env_var: str, default: Path) -> Path:
    _load_env_once()
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def data_raw_root() -> Path:
    return _resolve("DATA_RAW_ROOT", REPO_ROOT / "data" / "raw")


def data_processed_root() -> Path:
    return _resolve("DATA_PROCESSED_ROOT", REPO_ROOT / "data" / "processed")


def output_root() -> Path:
    return _resolve("OUTPUT_ROOT", REPO_ROOT / "outputs")
