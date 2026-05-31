"""Common interface for anything that yields BGR frames.

Both the live screen grabber and the video-file reader implement :class:`FrameSource`, so
everything downstream (detection, tracking, logic, viz) is completely source-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameMeta:
    """Lightweight metadata that travels alongside a frame."""

    index: int                 # monotonically increasing frame counter
    fps: float | None = None   # native source FPS if known (file); None for live screen


class FrameSource(ABC):
    """A source of BGR ``numpy`` frames.

    ``grab()`` returns the next frame, or ``None`` when the stream has ended (e.g. a video
    file reaching EOF). Live sources never return ``None``.
    """

    #: Frame rate of the source. Subclasses set this (file = native FPS, screen = target FPS).
    fps: float = 30.0

    @abstractmethod
    def grab(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` if the stream has ended."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources (capture handles, sockets, etc.)."""

    def __iter__(self):
        """Iterate frames until the source is exhausted."""
        while True:
            frame = self.grab()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
