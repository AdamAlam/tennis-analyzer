"""The per-frame orchestration loop.

The :class:`Pipeline` ties the stages together: pull a frame from the source, run detection
(and, where enabled, the ball tracker), draw the overlay, and render the preview window.
Later stages (tracking, court, logic, OCR, highlights) plug in here behind their config flags
without changing ``main.py``.
"""

from __future__ import annotations

import time

import cv2

from .capture.base import FrameSource
from .config import AppConfig
from .detection.ball import build_ball_tracker
from .detection.yolo_detector import YoloDetector
from .viz.overlay import draw_ball, draw_detections, draw_hud


class Pipeline:
    def __init__(self, cfg: AppConfig, source: FrameSource):
        self.cfg = cfg
        self.source = source

        # --- Detection (players + COCO ball fallback) ------------------------
        self.detector = YoloDetector(
            weights=cfg.detection.yolo.weights,
            device=cfg.device,
            half=cfg.half_precision,
            conf=cfg.detection.yolo.conf,
            iou=cfg.detection.yolo.iou,
            imgsz=cfg.detection.yolo.imgsz,
            classes=cfg.detection.yolo.classes,
        )

        # --- Ball tracker (reuses the YOLO model for the 'yolo' backend) -----
        self.ball_tracker = build_ball_tracker(cfg, yolo_detector=self.detector)

        # EMA-smoothed FPS counter for the HUD.
        self._fps = 0.0
        self._prev_t = time.perf_counter()

    # ------------------------------------------------------------------ #
    def _tick_fps(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_t
        self._prev_t = now
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        return self._fps

    def process_frame(self, frame):
        """Run all enabled stages on one frame and return the annotated frame."""
        detections = self.detector.detect(frame)
        ball = self.ball_tracker.update(frame)

        # NOTE: future stages slot in here, each gated by its config flag:
        #   if self.cfg.tracking...:  detections = self.tracker.update(frame, detections)
        #   if self.cfg.court.enabled: court = self.court.detect(frame)
        #   if self.cfg.logic.enabled: events = self.match_state.step(...)

        draw_detections(frame, detections)
        draw_ball(frame, ball)
        return frame

    def run(self) -> None:
        """Main loop: capture -> process -> display until EOF or the user quits."""
        cfg = self.cfg
        try:
            for frame in self.source:
                frame = self.process_frame(frame)
                fps = self._tick_fps()

                if cfg.viz.show_window:
                    if cfg.viz.draw_fps:
                        draw_hud(frame, fps)
                    cv2.imshow(cfg.viz.window_name, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
        finally:
            self.source.close()
            cv2.destroyAllWindows()
