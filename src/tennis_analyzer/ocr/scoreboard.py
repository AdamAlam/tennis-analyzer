"""Scoreboard reader (Step 7) - STUB.

Target behavior
----------------
Run EasyOCR on the configured scoreboard sub-region (``ocr.region``), throttled to every
``ocr.refresh_every`` frames, to read player names and the score (games/sets/points).

Implementation outline
-----------------------
1. Lazily construct ``easyocr.Reader(['en'], gpu=True)`` (heavy; build once).
2. Crop the scoreboard region; optionally upscale/threshold for small fonts.
3. ``reader.readtext(crop)`` -> text boxes; parse names/score with light regex.
4. Stabilize noisy reads with temporal voting (keep the most frequent recent value).
5. Optionally reconcile the OCR score with the logic-derived score (Step 6).
"""

from __future__ import annotations

import numpy as np


class ScoreboardReader:
    def __init__(self, region, refresh_every: int = 15, gpu: bool = True):
        self.region = region
        self.refresh_every = refresh_every
        self.gpu = gpu
        raise NotImplementedError(
            "ScoreboardReader is a Step-7 stub. Build an EasyOCR reader and parse the "
            "scoreboard crop here."
        )

    def read(self, frame: np.ndarray, frame_index: int):  # pragma: no cover - stub
        raise NotImplementedError
