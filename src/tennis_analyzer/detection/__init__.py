"""Object detection: YOLO for players (and a YOLO ball fallback for the MVP)."""

from .base import Detection, Detector
from .yolo_detector import YoloDetector

__all__ = ["Detection", "Detector", "YoloDetector"]
