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


def draw_court_keypoints(frame: np.ndarray, keypoints) -> np.ndarray:
    """Draw detected court keypoints (``(N, 3)`` x/y/conf, NaN rows skipped)."""
    if keypoints is None:
        return frame
    for kp in keypoints:
        x, y = kp[0], kp[1]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    return frame


def draw_ball_trace(frame: np.ndarray, trace) -> np.ndarray:
    """Draw a fading poly-line through recent ball positions."""
    pts = [p for p in trace if p is not None]
    for i in range(1, len(pts)):
        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), _BALL_COLOR, 2, cv2.LINE_AA)
    return frame


def draw_minimap(frame: np.ndarray, minimap: np.ndarray | None, pad: int = 12) -> np.ndarray:
    """Composite a top-down minimap into the bottom-right corner of the frame."""
    if minimap is None:
        return frame
    h, w = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    if mh + pad >= h or mw + pad >= w:
        return frame
    y1, x1 = h - mh - pad, w - mw - pad
    roi = frame[y1:y1 + mh, x1:x1 + mw]
    cv2.addWeighted(minimap, 0.85, roi, 0.15, 0, roi)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x1 + mw, y1 + mh), (200, 200, 200), 1)
    return frame


def draw_events(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw recent event log lines in the top-right corner."""
    h, w = frame.shape[:2]
    for i, line in enumerate(lines[-6:]):
        y = 24 + i * 22
        size = cv2.getTextSize(line, _FONT, 0.55, 1)[0]
        x = w - size[0] - 12
        cv2.putText(frame, line, (x + 1, y + 1), _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_score(frame: np.ndarray, score: dict | None) -> np.ndarray:
    """Draw a compact score/state line under the FPS HUD."""
    if not score:
        return frame
    parts = []
    if "state" in score:
        parts.append(str(score["state"]).upper())
    if "rally" in score:
        parts.append(f"rally {score['rally']}")
    if "points_far" in score and "points_near" in score:
        parts.append(f"pts {score['points_far']}-{score['points_near']}")
    if score.get("names"):
        parts.append(" / ".join(score["names"]))
    if score.get("scores"):
        parts.append(" ".join(score["scores"]))
    text = "  |  ".join(parts)
    if not text:
        return frame
    cv2.putText(frame, text, (11, 49), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 48), _FONT, 0.6, (80, 220, 255), 1, cv2.LINE_AA)
    return frame
