"""Image <-> top-down court mapping and in/out tests (Step 4) - STUB.

Target behavior
----------------
Given court keypoints in the image and their known positions on a real (ITF) court in meters,
compute a homography (``cv2.findHomography``) between image pixels and top-down court coords.
That lets us:
- project a ball-bounce pixel to court meters and test it against the singles/doubles lines
  (``is_inside_court``) for in/out calls,
- project player foot positions to the court for position stats / heatmaps,
- render a 2D minimap.

Real-court reference (singles): 23.77 m (length) x 8.23 m (width); doubles width 10.97 m.
"""

from __future__ import annotations

import numpy as np


class CourtHomography:
    def __init__(self, image_points: np.ndarray, court_points_m: np.ndarray):
        """``image_points`` and ``court_points_m`` are matched (N, 2) arrays."""
        self.image_points = image_points
        self.court_points_m = court_points_m
        raise NotImplementedError(
            "CourtHomography is a Step-4 stub. Compute H with cv2.findHomography and "
            "implement to_court / is_inside_court here."
        )

    def to_court(self, pt_img: tuple[float, float]) -> tuple[float, float]:  # pragma: no cover
        """Map an image pixel to top-down court coordinates (meters)."""
        raise NotImplementedError

    def is_inside_court(self, pt_img: tuple[float, float], singles: bool = True) -> bool:  # noqa: E501  # pragma: no cover
        """Return True if the image point lies inside the (singles/doubles) court."""
        raise NotImplementedError
