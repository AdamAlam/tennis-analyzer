"""Player tracker (Step 2) - STUB.

Target behavior
----------------
Replace per-frame detection with Ultralytics' built-in tracking so each player keeps a stable
id across frames:

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

Then:
- copy each box's ``id`` into :attr:`Detection.track_id`,
- keep only ``person`` detections, and reduce to the two most likely players (e.g. the two
  largest / most court-central boxes) to drop ball kids, line judges, and crowd.

This wraps the same YOLO model already loaded in the pipeline; pass it in to avoid a second
model load.
"""

from __future__ import annotations

import numpy as np

from ..detection.base import Detection


class PlayerTracker:
    def __init__(self, model, tracker: str = "bytetrack.yaml", persist: bool = True,
                 device: str = "cuda:0", half: bool = True):
        self.model = model
        self.tracker = tracker
        self.persist = persist
        self.device = device
        self.half = half
        raise NotImplementedError(
            "PlayerTracker is a Step-2 stub. Implement model.track(...) here and map ids "
            "onto Detection.track_id."
        )

    def update(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - stub
        raise NotImplementedError
