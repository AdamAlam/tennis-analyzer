"""A small constant-velocity Kalman filter for the ball track.

The TrackNet detector is strong but still misses the ball on some frames (occlusion, motion
blur at contact, leaving frame). A 2D constant-velocity Kalman filter smooths the measured
positions and, crucially, *predicts* through short gaps so the overlay/track does not flicker.

State vector: ``[x, y, vx, vy]`` in image pixels / pixels-per-frame.
"""

from __future__ import annotations

import numpy as np

try:  # filterpy is a declared dependency; degrade gracefully if it is missing.
    from filterpy.kalman import KalmanFilter

    _HAVE_FILTERPY = True
except Exception:  # pragma: no cover - exercised only when filterpy is absent
    _HAVE_FILTERPY = False


class BallKalman:
    """Smooth/predict the ball position across frames.

    Parameters
    ----------
    process_var:
        Process noise; larger = trust the motion model less (track reacts faster).
    measurement_var:
        Measurement noise; larger = trust detections less (track is smoother).
    max_misses:
        After this many consecutive missed detections the filter stops predicting and
        reports no position (the track is considered lost).
    """

    def __init__(self, process_var: float = 1.0, measurement_var: float = 4.0,
                 max_misses: int = 8):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.max_misses = max_misses
        self._misses = 0
        self._initialized = False
        self._kf: "KalmanFilter | None" = None

    # ------------------------------------------------------------------ #
    def _build(self, x: float, y: float) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0.0, 0.0], dtype=float)
        # State transition: constant velocity (dt = 1 frame).
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        # We only measure position.
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        kf.P *= 1000.0
        kf.R = np.eye(2) * self.measurement_var
        kf.Q = np.eye(4) * self.process_var
        self._kf = kf
        self._initialized = True
        self._misses = 0

    # ------------------------------------------------------------------ #
    def update(self, measurement: tuple[float, float] | None) -> tuple[float, float] | None:
        """Advance one frame.

        ``measurement`` is the detected ``(x, y)`` or ``None`` when the detector missed.
        Returns the smoothed ``(x, y)`` estimate, or ``None`` while the track is lost.
        """
        if not _HAVE_FILTERPY:
            # No filter available: pass the raw measurement straight through.
            return measurement

        if not self._initialized:
            if measurement is None:
                return None
            self._build(*measurement)
            return tuple(self._kf.x[:2])

        self._kf.predict()
        if measurement is not None:
            self._kf.update(np.asarray(measurement, dtype=float))
            self._misses = 0
        else:
            self._misses += 1
            if self._misses > self.max_misses:
                self._initialized = False
                return None
        return (float(self._kf.x[0]), float(self._kf.x[1]))

    def reset(self) -> None:
        """Forget the track (e.g. on a scene cut)."""
        self._initialized = False
        self._misses = 0
        self._kf = None
