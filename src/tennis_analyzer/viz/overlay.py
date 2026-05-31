"""Draw analysis results onto frames for the live preview window."""

from __future__ import annotations

import cv2
import numpy as np

from ..detection.ball.base import BallObservation
from ..detection.base import Detection

# BGR colors per label; unknown labels fall back to cyan.
_COLORS = {
    "person": (0, 255, 0),
    "sports ball": (0, 165, 255),
    "ball": (0, 165, 255),
    "tennis ball": (0, 165, 255),
}
_DEFAULT_COLOR = (255, 255, 0)
_BALL_COLOR = (0, 165, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels (and track ids when present) in place."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        color = _COLORS.get(det.label.lower(), _DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tag = f"{det.label} {det.conf:.2f}"
        if det.track_id is not None:
            tag += f" #{det.track_id}"
        cv2.putText(frame, tag, (x1, max(12, y1 - 6)), _FONT, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_ball(frame: np.ndarray, ball: BallObservation | None) -> np.ndarray:
    """Mark the current ball position with a filled circle + crosshair."""
    if ball is None:
        return frame
    x, y = int(ball.x), int(ball.y)
    cv2.circle(frame, (x, y), 6, _BALL_COLOR, 2)
    cv2.drawMarker(frame, (x, y), _BALL_COLOR, cv2.MARKER_CROSS, 14, 1)
    return frame


def draw_hud(frame: np.ndarray, fps: float, extra: str | None = None) -> np.ndarray:
    """Draw an FPS readout (and optional status text) in the top-left corner."""
    text = f"{fps:5.1f} FPS"
    if extra:
        text += f"  |  {extra}"
    # Drop shadow for readability over bright broadcast frames.
    cv2.putText(frame, text, (11, 25), _FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), _FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
