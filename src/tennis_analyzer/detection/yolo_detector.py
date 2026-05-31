"""Ultralytics YOLO11 detector wrapper.

Runs FP16 inference on CUDA by default. For the MVP this detects the COCO ``person`` class
(players) and, as a placeholder, the COCO ``sports ball`` class. After fine-tuning (Step 3)
point ``weights`` at the tennis model and restrict ``classes`` to cut false positives.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .base import Detection, Detector


class YoloDetector(Detector):
    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        half: bool = True,
        conf: float = 0.30,
        iou: float = 0.50,
        imgsz: int = 640,
        classes: list[int] | None = None,
    ):
        self.model = YOLO(weights)
        self.model.to(device)
        self.device = device
        # FP16 is only valid on CUDA; force it off on CPU smoke tests.
        self.half = half and str(device).startswith("cuda")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            device=self.device,
            half=self.half,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False,
        )[0]

        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls)
            detections.append(
                Detection(
                    xyxy=tuple(box.xyxy[0].tolist()),
                    conf=float(box.conf),
                    cls_id=cls_id,
                    label=names[cls_id],
                )
            )
        return detections
