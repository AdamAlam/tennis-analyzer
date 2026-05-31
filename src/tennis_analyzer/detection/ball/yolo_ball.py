"""MVP ball tracker backed by the YOLO ``sports ball`` / ``ball`` class.

Single-frame and not very robust on fast, blurred balls - this is the fallback until the
TrackNet temporal model (Step 5) is wired in. Picks the highest-confidence ball-like box.
"""

from __future__ import annotations

import numpy as np

from .base import BallObservation, BallTracker

# COCO class id for "sports ball" (used by stock yolo11n.pt). A fine-tuned model may instead
# expose a dedicated "ball" label, which we also match by name below.
_BALL_LABELS = {"sports ball", "ball", "tennis ball"}


class YoloBallTracker(BallTracker):
    def __init__(self, detector, conf: float = 0.5):
        if detector is None:
            raise ValueError("YoloBallTracker requires an existing YoloDetector instance.")
        self.detector = detector
        self.conf = conf

    def update(self, frame: np.ndarray) -> BallObservation | None:
        best = None
        for det in self.detector.detect(frame):
            if det.label.lower() in _BALL_LABELS and det.conf >= self.conf:
                if best is None or det.conf > best.conf:
                    cx, cy = det.center
                    best = BallObservation(x=cx, y=cy, conf=det.conf)
        return best
