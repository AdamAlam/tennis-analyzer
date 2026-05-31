"""Fast region screen capture built on ``mss``.

Grabs a fixed rectangle of the screen and returns contiguous BGR ``numpy`` arrays ready for
OpenCV / YOLO. A simple FPS cap avoids spinning the CPU faster than needed.
"""

from __future__ import annotations

import time

import numpy as np
from mss import mss

from .base import FrameSource


class ScreenCapture(FrameSource):
    """Grab a region of the screen as BGR frames.

    Parameters
    ----------
    monitor:
        ``mss`` monitor index (1 = primary display).
    region:
        Dict with ``top``, ``left``, ``width``, ``height`` in pixels.
    target_fps:
        Upper bound on capture rate; 0 disables the cap.
    """

    def __init__(self, monitor: int, region: dict, target_fps: int = 60):
        self._region = {
            "top": region["top"],
            "left": region["left"],
            "width": region["width"],
            "height": region["height"],
            "mon": monitor,
        }
        self._min_dt = 1.0 / target_fps if target_fps else 0.0
        self._sct = mss()
        self._last = 0.0

    def grab(self) -> np.ndarray:
        # Throttle to target_fps so we don't burn CPU grabbing faster than we can process.
        if self._min_dt:
            wait = self._min_dt - (time.perf_counter() - self._last)
            if wait > 0:
                time.sleep(wait)
        self._last = time.perf_counter()

        shot = self._sct.grab(self._region)        # BGRA buffer
        frame = np.asarray(shot)[:, :, :3]         # drop alpha -> BGR
        return np.ascontiguousarray(frame)         # live stream: never None

    def close(self) -> None:
        self._sct.close()
