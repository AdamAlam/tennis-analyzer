"""TrackNet ball tracker (Step 5) - STUB.

Target behavior
----------------
TrackNet is a temporal heatmap CNN purpose-built for tiny, fast balls. It consumes a stack of
3 consecutive (grayscale) frames -> 9 input channels and predicts a single-channel heatmap
whose peak is the ball center. This dramatically outperforms single-frame detectors on
motion-blurred tennis balls.

Implementation outline
-----------------------
1. Load the pretrained TrackNet tennis weights (``weights``) onto ``device`` (FP16 if ``half``).
2. Maintain a rolling buffer of the last 3 frames; resize each to the model input (e.g.
   360x640), convert to grayscale, stack -> tensor of shape (1, 9, H, W).
3. Forward pass -> heatmap; threshold + take the argmax/centroid as the ball pixel.
4. Rescale the pixel back to the original frame size; return a :class:`BallObservation`.
5. (Recommended) Feed positions through a Kalman filter (``filterpy``) to smooth the track
   and bridge single-frame misses. See ``reset()`` for scene-cut handling.

Pretrained weights: see ``scripts/download_models.py`` for the TrackNet source/URL.
"""

from __future__ import annotations

import numpy as np

from .base import BallObservation, BallTracker


class TrackNetBallTracker(BallTracker):
    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True, conf: float = 0.5):
        self.weights = weights
        self.device = device
        self.half = half
        self.conf = conf
        raise NotImplementedError(
            "TrackNetBallTracker is a Step-5 stub. Implement the 3-frame heatmap model here, "
            "or keep detection.ball.backend = 'yolo' for now."
        )

    def update(self, frame: np.ndarray) -> BallObservation | None:  # pragma: no cover - stub
        raise NotImplementedError
