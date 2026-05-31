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

        The peak of each keypoint heatmap is found with a single vectorized ``argmax`` on the
        GPU (then only 14 scalars are copied to the CPU), instead of running an OpenCV blob
        detection per channel - this keeps the stage to ~model-forward time.
        """
        torch = self.torch
        h0, w0 = frame.shape[:2]

        # Keep BGR (the pretrained court model was trained on OpenCV-decoded BGR frames).
        small = cv2.resize(frame, (self.in_w, self.in_h)).astype(np.float32) / 255.0
        chw = np.rollaxis(small, 2, 0)[np.newaxis, ...]
        inp = torch.from_numpy(chw).to(self.device)
        inp = inp.half() if self.half else inp.float()

        with torch.no_grad():
            out = self.model(inp)[0]                              # (15, H, W)
            heat = torch.sigmoid(out[:_NUM_KEYPOINTS].float())    # (14, H, W)
            hh, hw = heat.shape[1], heat.shape[2]
            conf, idx = heat.reshape(_NUM_KEYPOINTS, -1).max(dim=1)
            conf = conf.cpu().numpy()
            idx = idx.cpu().numpy()

        ys, xs = np.divmod(idx, hw)
        sx, sy = w0 / self.in_w, h0 / self.in_h

        points = np.full((_NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
        valid = conf >= self.min_conf
        points[valid, 0] = xs[valid] * sx
        points[valid, 1] = ys[valid] * sy
        points[:, 2] = np.where(valid, conf, 0.0)
        return points
