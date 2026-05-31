"""Video-file frame source via OpenCV, optionally paced to real time.

When ``realtime`` is True we sleep to match the file's native FPS so analysis feels like a
live broadcast; when False we decode as fast as the GPU allows (useful for batch analysis).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import FrameSource


class VideoFileSource(FrameSource):
    """Read frames from a video file.

    Parameters
    ----------
    path:
        Path to the video file.
    loop:
        Restart from the first frame when the file ends.
    realtime:
        Pace playback to the video's native FPS.
    """

    def __init__(self, path: str, loop: bool = False, realtime: bool = True):
        self.path = path
        self.loop = loop
        self.realtime = realtime

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._dt = 1.0 / self.fps
        self._last = 0.0

    def grab(self) -> np.ndarray | None:
        if self.realtime:
            wait = self._dt - (time.perf_counter() - self._last)
            if wait > 0:
                time.sleep(wait)
        self._last = time.perf_counter()

        ok, frame = self.cap.read()                # frames are already BGR
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None                        # genuine EOF
        return frame

    def close(self) -> None:
        self.cap.release()
