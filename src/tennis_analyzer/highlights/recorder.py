"""Highlight recorder (Step 8) - STUB.

Target behavior
----------------
Keep a rolling ring buffer of the most recent ``pre_seconds`` of frames. When an interesting
event fires (long rally, point won, ace), flush the buffered pre-roll plus the next
``post_seconds`` of frames to an MP4 in ``output_dir`` via ``cv2.VideoWriter``.

Implementation outline
-----------------------
1. ``collections.deque(maxlen = int(pre_seconds * fps))`` of frames, appended every frame.
2. On trigger: open a VideoWriter, dump the deque, then keep writing for ``post_seconds``.
3. Name clips by timestamp / event type; optionally burn in the overlay.
"""

from __future__ import annotations

import numpy as np


class HighlightRecorder:
    def __init__(self, output_dir: str, fps: float, pre_seconds: float = 6.0,
                 post_seconds: float = 3.0):
        self.output_dir = output_dir
        self.fps = fps
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        raise NotImplementedError(
            "HighlightRecorder is a Step-8 stub. Implement the ring buffer + VideoWriter flush."
        )

    def push(self, frame: np.ndarray) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def trigger(self, reason: str = "") -> None:  # pragma: no cover - stub
        raise NotImplementedError
