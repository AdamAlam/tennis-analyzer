"""Court keypoint detector (Step 4).

Runs the pretrained TennisCourtDetector keypoint CNN to locate the court's 14 reference points
(baseline/sideline/service-line intersections) in image coordinates.

Per call:
1. Resize the frame to the model input (640x360) and normalize to ``[0, 1]``.
2. Forward pass -> ``(1, 14, 360, 640)``; per-channel sigmoid gives 14 keypoint heatmaps.
3. For each channel, threshold + locate the brightest blob centroid -> ``(x, y)`` + confidence.
4. Rescale points back to the original frame size and return them in the canonical order so
   :class:`~tennis_analyzer.court.homography.CourtHomography` can match them to known reference
   coordinates.

Weights: the published ``model_tennis_court_det.pt`` from
``github.com/yastrebksv/TennisCourtDetector`` (placed at ``models/court/...`` by
``scripts/download_models.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..nn import BallTrackerNet

_DEFAULT_INPUT = (640, 360)
_NUM_KEYPOINTS = 14
# The pretrained model emits 15 heatmaps: 14 court keypoints + 1 extra court-center channel
# (used during training for better convergence). We only consume the first 14.
_NUM_OUTPUTS = 15


class CourtDetector:
    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True,
                 input_size: tuple[int, int] = _DEFAULT_INPUT, min_conf: float = 0.5):
        import torch

        self.torch = torch
        self.weights = weights
        self.device = device
        self.half = bool(half) and str(device).startswith("cuda")
        self.in_w, self.in_h = input_size
        self.min_conf = min_conf

        self.model = BallTrackerNet(in_channels=3, out_channels=_NUM_OUTPUTS)
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        if self.half:
            self.model.half()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return ``(14, 3)`` array of ``(x, y, conf)`` keypoints in original image pixels.

        Missing/low-confidence keypoints have ``conf == 0`` and ``NaN`` coordinates.
        """
        torch = self.torch
        h0, w0 = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.in_w, self.in_h)).astype(np.float32) / 255.0
        chw = np.rollaxis(small, 2, 0)[np.newaxis, ...]
        inp = torch.from_numpy(chw).to(self.device)
        inp = inp.half() if self.half else inp.float()

        with torch.no_grad():
            out = self.model(inp)                       # (1, 14, H, W)
            heatmaps = torch.sigmoid(out)[0].detach().cpu().numpy()

        sx, sy = w0 / self.in_w, h0 / self.in_h
        thresh = int(np.clip(self.min_conf, 0.0, 1.0) * 255)

        points = np.full((_NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
        for k in range(_NUM_KEYPOINTS):
            hm = (heatmaps[k] * 255).astype(np.uint8)
            _, binary = cv2.threshold(hm, thresh, 255, cv2.THRESH_BINARY)
            if not binary.any():
                points[k, 2] = 0.0
                continue
            num, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
            if num <= 1:
                points[k, 2] = 0.0
                continue
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cx, cy = centroids[largest]
            points[k] = (cx * sx, cy * sy, float(heatmaps[k].max()))
        return points
