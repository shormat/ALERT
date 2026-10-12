from __future__ import annotations

from alert.tracking.base import Tracker


def build_tracker(config: dict) -> Tracker:
    model_key = str(config.get("tracking", {}).get("model", "")).lower()

    if model_key in ("bytetrack", "byte_track"):
        from alert.tracking.models.bytetrack import ByteTrackTracker
        return ByteTrackTracker(config)

    if model_key in ("ocsort", "oc_sort"):
        from alert.tracking.models.ocsort import OCSortTracker
        return OCSortTracker(config)

    raise ValueError(
        f"Unknown tracking model: '{model_key}'. "
        "Expected one of: bytetrack, ocsort."
    )
