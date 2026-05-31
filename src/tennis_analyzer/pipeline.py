"""The per-frame orchestration loop.

The :class:`Pipeline` ties the stages together: pull a frame from the source, run detection /
player tracking and ball tracking, then (where enabled) court keypoints + homography, game
logic, scoreboard OCR, and highlight recording, and finally draw the overlay and render the
preview window. Every optional stage is gated by its config flag, so this one file scales from
the MVP up to the full analyzer without changing ``main.py``.
"""

from __future__ import annotations

import time
from collections import deque

import cv2

from .capture.base import FrameSource
from .config import AppConfig
from .detection.ball import build_ball_tracker
from .detection.yolo_detector import YoloDetector
from .logic.events import PointWon, Serve
from .viz.overlay import (
    draw_ball,
    draw_ball_trace,
    draw_court_keypoints,
    draw_detections,
    draw_events,
    draw_hud,
    draw_minimap,
    draw_score,
)


class Pipeline:
    def __init__(self, cfg: AppConfig, source: FrameSource):
        self.cfg = cfg
        self.source = source
        self.fps = getattr(source, "fps", 30.0) or 30.0

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

        # --- Optional stages, each behind its config flag -------------------
        self.player_tracker = None
        if cfg.tracking.enabled:
            from .tracking.tracker import PlayerTracker

            self.player_tracker = PlayerTracker(
                model=self.detector.model,
                tracker=cfg.tracking.tracker,
                persist=cfg.tracking.persist,
                device=cfg.device,
                half=cfg.half_precision,
                conf=cfg.detection.yolo.conf,
                iou=cfg.detection.yolo.iou,
                imgsz=cfg.detection.yolo.imgsz,
                max_players=cfg.tracking.max_players,
            )

        self.court_detector = None
        if cfg.court.enabled:
            from .court.detector import CourtDetector

            self.court_detector = CourtDetector(
                weights=cfg.court.weights,
                device=cfg.device,
                half=cfg.half_precision,
                input_size=(640, 360),
                min_conf=cfg.court.min_conf,
            )
        self.homography = None
        self.keypoints = None

        self.bounce_detector = None
        self.match_state = None
        if cfg.logic.enabled:
            from .logic.bounce import BounceDetector
            from .logic.match_state import MatchState

            self.bounce_detector = BounceDetector(
                min_vy=cfg.logic.bounce_min_vy,
                cooldown=cfg.logic.bounce_cooldown,
            )
            self.match_state = MatchState(ball_lost_frames=cfg.logic.ball_lost_frames)

        self.scoreboard = None
        if cfg.ocr.enabled and cfg.ocr.region is not None:
            from .ocr.scoreboard import ScoreboardReader

            self.scoreboard = ScoreboardReader(
                region=cfg.ocr.region,
                refresh_every=cfg.ocr.refresh_every,
                gpu=str(cfg.device).startswith("cuda"),
            )

        self.recorder = None
        if cfg.highlights.enabled:
            from .highlights.recorder import HighlightRecorder

            self.recorder = HighlightRecorder(
                output_dir=cfg.highlights.output_dir,
                fps=self.fps,
                pre_seconds=cfg.highlights.pre_seconds,
                post_seconds=cfg.highlights.post_seconds,
            )

        # State carried across frames.
        self._trace: deque = deque(maxlen=cfg.viz.trace_len)
        self._event_log: deque[str] = deque(maxlen=12)
        self._score: dict = {}
        self._frame_index = 0

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

    def _update_court(self, frame) -> None:
        """Refresh court keypoints + homography (cheap; only every refresh_every frames)."""
        if self.court_detector is None:
            return
        if self._frame_index % max(1, self.cfg.court.refresh_every) != 0 and self.homography:
            return
        from .court.homography import CourtHomography

        self.keypoints = self.court_detector.detect(frame)
        try:
            self.homography = CourtHomography(self.keypoints)
        except ValueError:
            # Not enough confident keypoints this refresh; keep the previous homography.
            pass

    # ------------------------------------------------------------------ #
    def process_frame(self, frame):
        """Run all enabled stages on one frame and return the annotated frame."""
        cfg = self.cfg
        idx = self._frame_index
        t = idx / self.fps

        # Players: tracked (stable ids) when enabled, else raw detections.
        if self.player_tracker is not None:
            detections = self.player_tracker.update(frame)
        else:
            detections = self.detector.detect(frame)

        ball = self.ball_tracker.update(frame)
        ball_xy = ball.xy if ball is not None else None
        self._trace.append(ball_xy)

        self._update_court(frame)

        # Game logic.
        if self.match_state is not None:
            bounce = self.bounce_detector.update(
                ball_xy, idx, t, homography=self.homography, singles=cfg.court.singles
            )
            if bounce is not None:
                verdict = "in" if bounce.inside else ("out" if bounce.inside is False else "?")
                self._event_log.append(f"[{t:5.1f}s] bounce ({verdict})")
            for ev in self.match_state.step(
                idx, t, ball_xy, bounce=bounce, players=detections,
                homography=self.homography, singles=cfg.court.singles,
            ):
                self._handle_event(ev, t)
            self._score.update(self.match_state.score)

        # Scoreboard OCR (throttled internally).
        if self.scoreboard is not None:
            self._score.update(self.scoreboard.read(frame, idx))

        annotated = self._draw(frame, detections, ball)

        # Highlights: buffer every frame; clips are triggered by events above.
        if self.recorder is not None:
            self.recorder.push(annotated)

        self._frame_index += 1
        return annotated

    def _handle_event(self, ev, t: float) -> None:
        if isinstance(ev, Serve):
            self._event_log.append(f"[{t:5.1f}s] serve (#{ev.server_track_id})")
        elif isinstance(ev, PointWon):
            self._event_log.append(
                f"[{t:5.1f}s] point #{ev.winner_track_id} ({ev.reason}, rally {ev.rally_length})"
            )
            if self.recorder is not None and ev.rally_length >= 1:
                self.recorder.trigger(f"point_{ev.reason}")
        # Long-rally highlight even before the point ends.
        if (self.recorder is not None and self.match_state is not None
                and self.match_state.rally_length == self.cfg.highlights.min_rally):
            self.recorder.trigger("long_rally")

    def _draw(self, frame, detections, ball):
        cfg = self.cfg
        if cfg.viz.draw_court and self.keypoints is not None:
            draw_court_keypoints(frame, self.keypoints)
        if cfg.viz.draw_tracks:
            draw_detections(frame, detections)
        if cfg.viz.draw_trace:
            draw_ball_trace(frame, list(self._trace))
        draw_ball(frame, ball)

        if cfg.viz.draw_minimap and self.homography is not None:
            pts = []
            if ball is not None:
                pts.append((ball.x, ball.y, (0, 165, 255)))
            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                pts.append((0.5 * (x1 + x2), y2, (0, 255, 0)))
            try:
                draw_minimap(frame, self.homography.draw_minimap(pts))
            except Exception:
                pass

        if self._score:
            draw_score(frame, self._score)
        if self._event_log:
            draw_events(frame, list(self._event_log))
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
            if self.recorder is not None:
                self.recorder.close()
            cv2.destroyAllWindows()
