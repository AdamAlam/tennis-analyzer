"""TrackNet ball tracker (Step 5).

TrackNet is a temporal heatmap CNN purpose-built for tiny, fast, motion-blurred balls. It
consumes a stack of 3 consecutive RGB frames (-> 9 input channels) and predicts a heatmap
whose bright peak is the ball center. This dramatically outperforms single-frame detectors.

Pipeline per frame:
1. Maintain a rolling buffer of the last 3 frames, each resized to ``input_size`` (W, H) and
   normalized to ``[0, 1]``; concatenate on the channel axis -> ``(1, 9, H, W)``.
2. Forward pass -> ``(1, 256, H, W)``; ``argmax`` over the channel dim recovers a single
   0..255 intensity heatmap (the network predicts a Gaussian blob around the ball).
3. Threshold the heatmap and take the brightest blob's centroid as the ball pixel.
4. Rescale the pixel back to the original frame resolution and return a
   :class:`BallObservation`.
5. Feed the position through a Kalman filter (:class:`BallKalman`) to smooth the track and
   bridge single-frame misses.

Weights: the published ``model_best.pt`` from ``github.com/yastrebksv/TrackNet`` (placed at
``models/tracknet/...`` by ``scripts/download_models.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

from ...nn import BallTrackerNet
from .base import BallObservation, BallTracker
from .kalman import BallKalman

# The pretrained model was trained at 640x360 (W x H).
_DEFAULT_INPUT = (640, 360)


class TrackNetBallTracker(BallTracker):
    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True,
                 conf: float = 0.5, input_size: tuple[int, int] = _DEFAULT_INPUT,
                 use_kalman: bool = True, max_dist: float = 100.0):
        import torch

        self.torch = torch
        self.weights = weights
        self.device = device
        self.half = bool(half) and str(device).startswith("cuda")
        # ``conf`` (0..1) maps onto the 0..255 heatmap threshold.
        self.thresh = int(np.clip(conf, 0.0, 1.0) * 255)
        self.in_w, self.in_h = input_size
        self.max_dist = max_dist
        self._last_xy: tuple[float, float] | None = None

        self.model = BallTrackerNet(in_channels=9, out_channels=256)
        state = torch.load(weights, map_location=device)
        # Some checkpoints wrap the weights under a key; accept either layout.
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        if self.half:
            self.model.half()

        self._frames: list[np.ndarray] = []
        self._kalman = BallKalman() if use_kalman else None

    # ------------------------------------------------------------------ #
    def _preprocess(self) -> "np.ndarray":
        # Newest frame first, matching the reference implementation's ordering. Frames are
        # kept in BGR (no color conversion) because the pretrained model was trained on
        # OpenCV-decoded BGR frames.
        stacked = np.concatenate(self._frames[::-1], axis=2)  # (H, W, 9), BGR triples
        stacked = stacked.astype(np.float32) / 255.0
        chw = np.rollaxis(stacked, 2, 0)                      # (9, H, W)
        return chw[np.newaxis, ...]                           # (1, 9, H, W)

    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray) -> BallObservation | None:
        torch = self.torch
        h0, w0 = frame.shape[:2]

        # Keep BGR (the pretrained TrackNet was trained on OpenCV-decoded BGR frames).
        small = cv2.resize(frame, (self.in_w, self.in_h))
        self._frames.append(small)
        if len(self._frames) > 3:
            self._frames.pop(0)

        measured: tuple[float, float] | None = None
        conf = 0.0
        if len(self._frames) == 3:
            inp = torch.from_numpy(self._preprocess()).to(self.device)
            inp = inp.half() if self.half else inp.float()
            with torch.no_grad():
                out = self.model(inp)                         # (1, 256, Hm, Wm)
                hm_h, hm_w = out.shape[2], out.shape[3]
                # Per-pixel predicted intensity (0..255), then the single brightest pixel.
                # This is the ball peak - robust to large dim regions and avoids a full
                # heatmap copy + connectedComponents on the CPU.
                intensity = out.argmax(dim=1).reshape(-1)     # (Hm*Wm,) on GPU
                peak_val, peak_idx = intensity.max(dim=0)
            pv = float(peak_val.item())
            if pv >= self.thresh:
                idx = int(peak_idx.item())
                py, px = divmod(idx, hm_w)
                # Rescale from model-input space back to the original frame.
                sx, sy = w0 / hm_w, h0 / hm_h
                candidate = (px * sx, py * sy)
                # Outlier gate: reject implausible jumps from the last accepted position.
                if self._last_xy is not None:
                    jump = np.hypot(candidate[0] - self._last_xy[0],
                                    candidate[1] - self._last_xy[1])
                    if jump > self.max_dist:
                        candidate = None
                if candidate is not None:
                    measured = candidate
                    self._last_xy = candidate
                    conf = pv / 255.0

        # Smooth / bridge gaps with the Kalman filter when enabled.
        if self._kalman is not None:
            estimate = self._kalman.update(measured)
            if estimate is None:
                return None
            # Carry detector confidence when present; flag predicted-only frames low.
            return BallObservation(x=estimate[0], y=estimate[1],
                                   conf=conf if measured is not None else 0.1)

        if measured is None:
            return None
        return BallObservation(x=measured[0], y=measured[1], conf=conf)

    def reset(self) -> None:
        self._frames.clear()
        self._last_xy = None
        if self._kalman is not None:
            self._kalman.reset()
