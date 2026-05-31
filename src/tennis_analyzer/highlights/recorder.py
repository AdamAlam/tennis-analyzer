"""Highlight recorder (Step 8).

Keeps a rolling ring buffer of the most recent ``pre_seconds`` of frames. When an interesting
event fires (point won, long rally), it flushes the buffered pre-roll plus the next
``post_seconds`` of frames to an MP4 in ``output_dir``.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class HighlightRecorder:
    def __init__(self, output_dir: str, fps: float, pre_seconds: float = 6.0,
                 post_seconds: float = 3.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, float(fps))
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds

        self._pre = deque(maxlen=int(self.pre_seconds * self.fps))
        self._writer: cv2.VideoWriter | None = None
        self._post_remaining = 0
        self._size: tuple[int, int] | None = None

    # ------------------------------------------------------------------ #
    def push(self, frame: np.ndarray) -> None:
        """Append a frame; if a clip is recording, also write it (and finish the post-roll)."""
        self._size = (frame.shape[1], frame.shape[0])

        if self._writer is not None:
            self._writer.write(frame)
            self._post_remaining -= 1
            if self._post_remaining <= 0:
                self._writer.release()
                self._writer = None
        else:
            self._pre.append(frame.copy())

    def trigger(self, reason: str = "") -> None:
        """Start a clip: flush the pre-roll buffer, then keep recording the post-roll."""
        if self._writer is not None:
            # Already recording: extend the post-roll instead of starting a new file.
            self._post_remaining = max(self._post_remaining, int(self.post_seconds * self.fps))
            return
        if self._size is None:
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_reason = "".join(c if c.isalnum() else "_" for c in reason) or "event"
        path = self.output_dir / f"{stamp}_{safe_reason}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, self.fps, self._size)

        for f in self._pre:
            self._writer.write(f)
        self._pre.clear()
        self._post_remaining = int(self.post_seconds * self.fps)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
