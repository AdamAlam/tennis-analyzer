"""Court keypoint detector (Step 4) - STUB.

Target behavior
----------------
Run the pretrained TennisCourtDetector keypoint CNN to locate the court's reference points
(typically 14: baseline/sideline/service-line intersections) in image coordinates.

Implementation outline
-----------------------
1. Load the pretrained court-keypoint weights onto ``device``.
2. Per call: preprocess the frame to the model input, forward pass, decode per-keypoint
   heatmaps -> (x, y) image points + confidences.
3. Return the ordered keypoints so :class:`CourtHomography` can map them to known real-world
   court coordinates.
4. The pipeline only calls this every ``court.refresh_every`` frames (cheap for a fixed camera).

Pretrained weights: see ``scripts/download_models.py`` (yastrebksv/TennisCourtDetector).
"""

from __future__ import annotations

import numpy as np


class CourtDetector:
    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True):
        self.weights = weights
        self.device = device
        self.half = half
        raise NotImplementedError(
            "CourtDetector is a Step-4 stub. Load the TennisCourtDetector model and return "
            "ordered court keypoints here."
        )

    def detect(self, frame: np.ndarray):  # pragma: no cover - stub
        """Return an array of court keypoints, shape (N, 2) in image pixels."""
        raise NotImplementedError
