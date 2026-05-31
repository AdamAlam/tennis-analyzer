"""The swappable ball-tracker interface.

A ``BallTracker`` consumes frames one at a time (allowing temporal models like TrackNet to
keep state across frames) and returns the current ball position, if any.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BallObservation:
    """The ball's estimated position in image coordinates for a single frame."""

    x: float
    y: float
    conf: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


class BallTracker(ABC):
    """Estimate the ball position from a stream of frames."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> BallObservation | None:
        """Process the next frame and return the ball position, or ``None`` if not found."""

    def reset(self) -> None:
        """Clear any temporal state (e.g. on a scene cut). Default: no-op."""
        return None
