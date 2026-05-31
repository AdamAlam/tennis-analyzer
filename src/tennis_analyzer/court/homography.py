"""Image <-> top-down court mapping and in/out tests (Step 4).

Given detected court keypoints in the image and their known reference positions (see
:mod:`tennis_analyzer.court.reference`), we compute a homography with ``cv2.findHomography``
between image pixels and the top-down court frame. That lets us:

- project a ball-bounce pixel to court coordinates and test it against the singles/doubles
  lines (:meth:`is_inside_court`) for in/out calls,
- project player foot positions to the court for position stats,
- render a 2D minimap (:meth:`draw_minimap`).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import reference as ref


class CourtHomography:
    """Homography between image pixels and the top-down court reference frame.

    Parameters
    ----------
    image_points:
        ``(14, 2)`` (or ``(14, 3)`` with confidence) detected keypoints, in the canonical
        order. Rows that are ``NaN`` / zero-confidence are ignored.
    court_points_m:
        Matched reference points; defaults to :data:`reference.KEYPOINTS`.
    """

    def __init__(self, image_points: np.ndarray, court_points_m: np.ndarray | None = None):
        court = ref.KEYPOINTS if court_points_m is None else np.asarray(court_points_m,
                                                                        dtype=np.float32)
        img = np.asarray(image_points, dtype=np.float32)
        conf = img[:, 2] if img.shape[1] >= 3 else np.ones(len(img))
        img_xy = img[:, :2]

        valid = np.isfinite(img_xy).all(axis=1) & (conf > 0)
        if valid.sum() < 4:
            raise ValueError(
                f"Need >=4 valid court keypoints to fit a homography, got {int(valid.sum())}."
            )

        self.image_points = img_xy
        self.court_points_m = court
        self._n_valid = int(valid.sum())
        self.H, _ = cv2.findHomography(img_xy[valid], court[valid], cv2.RANSAC, 5.0)
        if self.H is None:
            raise ValueError("cv2.findHomography failed to find a valid court homography.")
        self.H_inv = np.linalg.inv(self.H)

    @property
    def num_valid_points(self) -> int:
        return self._n_valid

    # ------------------------------------------------------------------ #
    def to_court(self, pt_img: tuple[float, float]) -> tuple[float, float]:
        """Map an image pixel to top-down court coordinates (reference pixels)."""
        src = np.array([[[pt_img[0], pt_img[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)[0][0]
        return (float(dst[0]), float(dst[1]))

    def to_court_meters(self, pt_img: tuple[float, float]) -> tuple[float, float]:
        """Map an image pixel to court coordinates in meters (origin at the court corner)."""
        cx, cy = self.to_court(pt_img)
        x_m = (cx - ref.X_DOUBLES_LEFT) * ref.METERS_PER_REF_PX
        y_m = (cy - ref.Y_TOP) * ref.METERS_PER_REF_PX
        return (x_m, y_m)

    def to_image(self, pt_court: tuple[float, float]) -> tuple[float, float]:
        """Map a top-down court point back to image pixels."""
        src = np.array([[[pt_court[0], pt_court[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_inv)[0][0]
        return (float(dst[0]), float(dst[1]))

    def is_inside_court(self, pt_img: tuple[float, float], singles: bool = True,
                        margin: float = 0.0) -> bool:
        """Return True if the image point lies inside the (singles/doubles) court.

        ``margin`` (reference pixels) widens the bounds to absorb keypoint/bounce noise; the
        real ITF line-width tolerance is small, so keep this near 0 for strict calls.
        """
        cx, cy = self.to_court(pt_img)
        x_left = ref.X_SINGLES_LEFT if singles else ref.X_DOUBLES_LEFT
        x_right = ref.X_SINGLES_RIGHT if singles else ref.X_DOUBLES_RIGHT
        return (
            (x_left - margin) <= cx <= (x_right + margin)
            and (ref.Y_TOP - margin) <= cy <= (ref.Y_BOTTOM + margin)
        )

    # ------------------------------------------------------------------ #
    def draw_minimap(self, points_img: list[tuple[float, float, tuple[int, int, int]]],
                     scale: float = 0.06) -> np.ndarray:
        """Render a small top-down court with the given points projected onto it.

        ``points_img`` is a list of ``(x_img, y_img, bgr_color)`` markers (ball, players).
        Returns a BGR image suitable for compositing into a corner of the frame.
        """
        w = int(ref.REF_WIDTH * scale)
        h = int(ref.REF_HEIGHT * scale)
        canvas = np.full((h, w, 3), 30, dtype=np.uint8)

        def s(pt: tuple[float, float]) -> tuple[int, int]:
            return (int(pt[0] * scale), int(pt[1] * scale))

        for p1, p2 in ref.reference_lines():
            cv2.line(canvas, s(p1), s(p2), (200, 200, 200), 1, cv2.LINE_AA)

        for x_img, y_img, color in points_img:
            cx, cy = self.to_court((x_img, y_img))
            cv2.circle(canvas, s((cx, cy)), 4, color, -1, cv2.LINE_AA)
        return canvas
