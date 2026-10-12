from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alert.tracking.base import TrackedObject
    from alert.segmentation.base import SegmentationResult
    from alert.risk.risk_reasoner import RiskResult

_COOLDOWN_SEC = 3.0


class VoiceAlertSystem:
    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("alerts", {})
        self._enabled = bool(cfg.get("enabled", True))
        self._cooldown = float(cfg.get("cooldown_sec", _COOLDOWN_SEC))
        self._warn_thresh = float(cfg.get("warn_thresh", 0.3))
        self._crit_thresh = float(cfg.get("crit_thresh", 0.6))
        self._last_alert_time: float = 0.0
        self._engine = None

        if self._enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
            except Exception:
                self._engine = None

    def update(
        self,
        tracks: list,
        seg_result=None,
        frame_shape: tuple = (720, 1280),
        risk_result=None,
    ) -> str | None:
        if not self._enabled or risk_result is None:
            return None

        scene_risk = risk_result.scene_risk
        level = risk_result.alert_level

        if level == "INFO":
            return None

        now = time.monotonic()
        if now - self._last_alert_time < self._cooldown:
            return None

        if level == "CRITICAL":
            min_ttc = float("inf")
            min_label = "obstacle"
            for t in tracks:
                ttc = risk_result.ttc_estimates.get(t.track_id, float("inf"))
                if ttc < min_ttc:
                    min_ttc = ttc
                    min_label = t.label
            if min_ttc < float("inf"):
                msg = f"Warning! {min_label} in {min_ttc:.1f} seconds"
            else:
                msg = "Danger ahead — reduce speed"
        else:
            msg = "Caution — potential hazard ahead"

        self._last_alert_time = now
        self._speak(msg)
        return msg

    def _speak(self, text: str) -> None:
        if self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass
