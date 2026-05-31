"""Ball tracking: a swappable interface with a YOLO fallback and a TrackNet implementation.

The MVP uses :class:`YoloBallTracker`; switch ``detection.ball.backend`` to ``tracknet`` once
the TrackNet weights are in place (Step 5) without touching the pipeline.
"""

from .base import BallObservation, BallTracker

__all__ = ["BallObservation", "BallTracker", "build_ball_tracker"]


def build_ball_tracker(cfg, yolo_detector=None) -> BallTracker:
    """Construct the configured ball tracker.

    Parameters
    ----------
    cfg:
        The validated :class:`AppConfig`.
    yolo_detector:
        An existing :class:`YoloDetector` to reuse for the YOLO backend (avoids loading a
        second model). Required when ``backend == "yolo"``.
    """
    backend = cfg.detection.ball.backend
    if backend == "yolo":
        from .yolo_ball import YoloBallTracker

        return YoloBallTracker(detector=yolo_detector, conf=cfg.detection.ball.conf)
    if backend == "tracknet":
        from .tracknet_ball import TrackNetBallTracker

        return TrackNetBallTracker(
            weights=cfg.detection.ball.tracknet_weights,
            device=cfg.device,
            half=cfg.half_precision,
            conf=cfg.detection.ball.conf,
        )
    raise ValueError(f"Unknown ball backend: {backend!r}")
