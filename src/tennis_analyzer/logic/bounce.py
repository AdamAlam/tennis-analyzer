"""Bounce detection from the ball trajectory (Step 6).

A bounce shows up in the image as a local *maximum* in the ball's ``y`` (remember image ``y``
grows downward: the ball descends, hits the court, then ascends), i.e. the vertical velocity
flips from positive (going down) to negative (going up). We smooth the track first so detector
jitter does not create phantom bounces, then map the bounce pixel through the court homography
to classify it in/out.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .events import Bounce

try:
    from scipy.signal import savgol_filter

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


class BounceDetector:
    """Detect bounces from a stream of ball positions.

    Parameters
    ----------
    history:
        Number of recent positions to keep for smoothing/derivatives.
    min_vy:
        Minimum downward speed (pixels/frame) just before the apex to count as a real bounce
        (filters out near-stationary jitter).
    cooldown:
        Minimum frames between two reported bounces.
    """

    def __init__(self, history: int = 30, min_vy: float = 2.0, cooldown: int = 8):
        self.history = history
        self.min_vy = min_vy
        self.cooldown = cooldown
        self._xs: deque[float] = deque(maxlen=history)
        self._ys: deque[float] = deque(maxlen=history)
        self._idx: deque[int] = deque(maxlen=history)
        self._ts: deque[float] = deque(maxlen=history)
        self._last_bounce_idx = -(10**9)

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()
        self._idx.clear()
        self._ts.clear()
        self._last_bounce_idx = -(10**9)

    def _smooth_y(self) -> np.ndarray:
        ys = np.asarray(self._ys, dtype=float)
        if _HAVE_SCIPY and len(ys) >= 7:
            win = min(len(ys) if len(ys) % 2 == 1 else len(ys) - 1, 11)
            if win >= 5:
                return savgol_filter(ys, win, 2)
        return ys

    def update(self, ball_xy, frame_index: int, t_seconds: float,
               homography=None, singles: bool = True) -> Bounce | None:
        """Feed one ball position; return a :class:`Bounce` on the frame a bounce is confirmed.

        ``ball_xy`` may be ``None`` when the ball was not found this frame.
        """
        if ball_xy is None:
            return None

        self._xs.append(float(ball_xy[0]))
        self._ys.append(float(ball_xy[1]))
        self._idx.append(frame_index)
        self._ts.append(t_seconds)

        if len(self._ys) < 5:
            return None

        ys = self._smooth_y()
        # Examine the second-to-last sample as a candidate apex (needs one frame after it).
        c = len(ys) - 2
        vy_before = ys[c] - ys[c - 1]      # >0 means descending
        vy_after = ys[c + 1] - ys[c]       # <0 means ascending
        is_apex = vy_before > self.min_vy and vy_after < -self.min_vy * 0.5

        if not is_apex:
            return None
        if (self._idx[c] - self._last_bounce_idx) < self.cooldown:
            return None

        self._last_bounce_idx = self._idx[c]
        bx, by = self._xs[c], ys[c]

        x_court = y_court = None
        inside = None
        if homography is not None:
            try:
                x_court, y_court = homography.to_court_meters((bx, by))
                inside = homography.is_inside_court((bx, by), singles=singles)
            except Exception:
                pass

        return Bounce(
            frame_index=self._idx[c],
            t_seconds=self._ts[c],
            x_img=bx,
            y_img=float(by),
            x_court_m=x_court,
            y_court_m=y_court,
            inside=inside,
        )
