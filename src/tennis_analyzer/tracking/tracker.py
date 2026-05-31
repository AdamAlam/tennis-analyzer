"""Player tracker (Step 2).

Replaces per-frame detection with Ultralytics' built-in multi-object tracking so each player
keeps a stable id across frames:

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], ...)

We then:
- copy each box's tracker ``id`` into :attr:`Detection.track_id`,
- keep only the ``person`` class, and
- reduce to the two most likely players (largest, most court-central boxes) to drop ball kids,
  line judges, and the crowd.

This wraps the same YOLO model already loaded by the pipeline (pass it in to avoid a second
model load).
"""

from __future__ import annotations

import numpy as np

from ..detection.base import Detection

_PERSON_CLASS = 0


class PlayerTracker:
    def __init__(self, model, tracker: str = "bytetrack.yaml", persist: bool = True,
                 device: str = "cuda:0", half: bool = True, conf: float = 0.3,
                 iou: float = 0.5, imgsz: int = 640, max_players: int = 2):
        self.model = model
        self.tracker = tracker
        self.persist = persist
        self.device = device
        self.half = bool(half) and str(device).startswith("cuda")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_players = max_players

    def update(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame,
            persist=self.persist,
            tracker=self.tracker,
            classes=[_PERSON_CLASS],
            device=self.device,
            half=self.half,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        names = result.names
        h, w = frame.shape[:2]
        cx_img = w / 2.0

        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id != _PERSON_CLASS:
                continue
            track_id = int(box.id) if box.id is not None else None
            detections.append(
                Detection(
                    xyxy=tuple(box.xyxy[0].tolist()),
                    conf=float(box.conf),
                    cls_id=cls_id,
                    label=names[cls_id],
                    track_id=track_id,
                )
            )

        return self._keep_players(detections, cx_img)

    def _keep_players(self, dets: list[Detection], cx_img: float) -> list[Detection]:
        """Reduce to the ``max_players`` most plausible players.

        Score combines box area (players are large, foreground people) with horizontal
        centrality (players are near the court's vertical centerline, crowd is at the edges).
        """
        if len(dets) <= self.max_players:
            return dets

        def score(d: Detection) -> float:
            cx, _ = d.center
            centrality = 1.0 - min(1.0, abs(cx - cx_img) / max(cx_img, 1.0))
            return d.area * (0.5 + 0.5 * centrality)

        return sorted(dets, key=score, reverse=True)[: self.max_players]
