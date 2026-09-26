"""Motion gate: skip frames where nothing moved so the heavy models run less.

Two methods:
* ``diff``  - absolute difference against the last *kept* frame (cheap, default).
  Comparing with the last kept frame (not the previous sampled one) means slow
  drift still accumulates until it crosses the threshold.
* ``mog2``  - OpenCV MOG2 background subtractor; more robust to lighting flicker.

A heartbeat keeps one frame every ``heartbeat_seconds`` even in a fully static
scene, so a parked vehicle or an unattended bag is still searchable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GateStats:
    seen: int = 0
    kept: int = 0
    skipped: int = 0
    kept_by_heartbeat: int = 0
    scores: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "frames_seen": self.seen, "frames_kept": self.kept, "frames_skipped": self.skipped,
            "kept_by_heartbeat": self.kept_by_heartbeat,
            "skip_ratio": round(self.skipped / self.seen, 4) if self.seen else 0.0,
        }


class MotionGate:
    def __init__(self, enabled: bool = True, method: str = "diff", diff_threshold: int = 25,
                 min_changed_fraction: float = 0.002, heartbeat_seconds: float = 30.0,
                 downscale_width: int = 320, **_ignored):
        if method not in ("diff", "mog2"):
            raise ValueError(f"unknown motion method {method!r}")
        self.enabled = enabled
        self.method = method
        self.diff_threshold = diff_threshold
        self.min_changed_fraction = min_changed_fraction
        self.heartbeat_seconds = heartbeat_seconds
        self.downscale_width = downscale_width
        self.stats = GateStats()
        self._ref: np.ndarray | None = None
        self._last_kept_t: float | None = None
        self._mog = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=32, detectShadows=False) \
            if method == "mog2" else None

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.downscale_width / w
        small = cv2.resize(frame, (self.downscale_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def motion_score(self, frame: np.ndarray) -> float:
        """Fraction of pixels considered 'changed'. Updates internal model (mog2)."""
        g = self._prep(frame)
        if self.method == "mog2":
            mask = self._mog.apply(g)
            return float(np.count_nonzero(mask > 0)) / mask.size
        if self._ref is None:
            return 1.0
        diff = cv2.absdiff(g, self._ref)
        return float(np.count_nonzero(diff > self.diff_threshold)) / diff.size

    def keep(self, frame: np.ndarray, t: float) -> bool:
        self.stats.seen += 1
        score = self.motion_score(frame)
        self.stats.scores.append(score)
        first = self._last_kept_t is None
        heartbeat = (not first) and (t - self._last_kept_t) >= self.heartbeat_seconds
        moved = score >= self.min_changed_fraction
        keep = (not self.enabled) or first or moved or heartbeat
        if keep:
            self.stats.kept += 1
            if heartbeat and not moved and self.enabled:
                self.stats.kept_by_heartbeat += 1
            self._last_kept_t = t
            if self.method == "diff":
                self._ref = self._prep(frame)
        else:
            self.stats.skipped += 1
        return keep
