"""Background-thread frame prefetch.

Wraps any :class:`FrameSource` and grabs the next frame on a worker thread, so frame decode
(video) or screen grab overlaps the GPU inference of the previous frame instead of blocking it
on the critical path. With a 1-deep hand-off the consumer always gets the freshest frame.
"""

from __future__ import annotations

import threading

import numpy as np

from .base import FrameSource


class ThreadedFrameSource(FrameSource):
    def __init__(self, inner: FrameSource):
        self.inner = inner
        self.fps = getattr(inner, "fps", 30.0)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._has_frame = False
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            frame = self.inner.grab()
            with self._cond:
                if self._stopped:
                    return
                self._frame = frame
                self._has_frame = True
                self._cond.notify()
                if frame is None:  # end of stream; stop prefetching
                    return

    def grab(self) -> np.ndarray | None:
        with self._cond:
            while not self._has_frame and not self._stopped:
                self._cond.wait()
            frame = self._frame
            self._has_frame = False
            return frame

    def close(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self.inner.close()
