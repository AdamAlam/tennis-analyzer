"""Detection data type and the detector interface shared across the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detected object in image coordinates.

    Attributes
    ----------
    xyxy:
        Bounding box as ``(x1, y1, x2, y2)`` in pixels.
    conf:
        Detector confidence in ``[0, 1]``.
    cls_id:
        Numeric class id from the model.
    label:
        Human-readable class name.
    track_id:
        Stable id assigned by the tracker (Step 2); ``None`` for raw detections.
    """

    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    label: str
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class Detector(ABC):
    """Anything that turns a frame into a list of :class:`Detection`."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame."""
