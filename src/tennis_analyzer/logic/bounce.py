"""Bounce detection from the ball trajectory (Step 6) - STUB.

Target behavior
----------------
Detect bounces from the smoothed ball track: a bounce is a local minimum in the ball's image
``y`` (it stops descending and starts ascending) near the court plane, i.e. a sign change in
vertical velocity. Combine with the court homography to classify the bounce as in/out.

Implementation outline
-----------------------
1. Maintain a rolling history of ball positions (from the BallTracker).
2. Smooth (e.g. Savitzky-Golay via scipy) and compute vertical velocity.
3. Flag a bounce when velocity flips from + to - with sufficient magnitude.
4. Map the bounce pixel through CourtHomography -> court meters -> in/out -> emit a Bounce.
"""

from __future__ import annotations

from .events import Bounce


class BounceDetector:
    def __init__(self, history: int = 30):
        self.history = history
        raise NotImplementedError(
            "BounceDetector is a Step-6 stub. Implement trajectory-based bounce detection here."
        )

    def update(self, ball_xy, frame_index: int, t_seconds: float) -> Bounce | None:  # pragma: no cover  # noqa: E501
        raise NotImplementedError
