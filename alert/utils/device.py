from __future__ import annotations

import torch


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_device(config: dict) -> str:
    requested = str(config.get("device", "auto")).lower()
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("device='mps' requested but MPS backend is not available")
        return "mps"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if _mps_available():
            return "mps"
        return "cpu"
    if requested.isdigit() or requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={requested!r} requested but CUDA is not available")
        return f"cuda:{requested}" if requested.isdigit() else requested
    raise ValueError(f"unknown device specification: {requested!r}")
