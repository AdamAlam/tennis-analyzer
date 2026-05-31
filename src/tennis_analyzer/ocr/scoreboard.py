"""Scoreboard reader (Step 7).

Runs EasyOCR on the configured scoreboard sub-region, throttled to every ``refresh_every``
frames, to read player names and the score. OCR on small broadcast fonts is noisy, so reads
are stabilized with temporal majority voting before being surfaced.
"""

from __future__ import annotations

import re
from collections import Counter, deque

import cv2
import numpy as np


class ScoreboardReader:
    def __init__(self, region, refresh_every: int = 15, gpu: bool = True,
                 vote_window: int = 5, upscale: float = 2.0):
        self.region = region
        self.refresh_every = max(1, refresh_every)
        self.gpu = gpu
        self.upscale = upscale

        self._reader = None  # built lazily; EasyOCR import is heavy
        self._last_result: dict = {}
        self._history: deque[tuple] = deque(maxlen=vote_window)

    # ------------------------------------------------------------------ #
    def _ensure_reader(self):
        if self._reader is None:
            import easyocr  # heavy import; defer until first real read

            self._reader = easyocr.Reader(["en"], gpu=self.gpu)
        return self._reader

    def _crop(self, frame: np.ndarray) -> np.ndarray | None:
        r = self.region
        top = getattr(r, "top", None) if not isinstance(r, dict) else r.get("top")
        left = getattr(r, "left", None) if not isinstance(r, dict) else r.get("left")
        width = getattr(r, "width", None) if not isinstance(r, dict) else r.get("width")
        height = getattr(r, "height", None) if not isinstance(r, dict) else r.get("height")
        if None in (top, left, width, height):
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(left)), max(0, int(top))
        x2, y2 = min(w, int(left + width)), min(h, int(top + height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if self.upscale and self.upscale != 1.0:
            crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale,
                              interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _parse(texts: list[str]) -> dict:
        """Light heuristic parse: name-like tokens vs score-like tokens."""
        names: list[str] = []
        scores: list[str] = []
        for t in texts:
            t = t.strip()
            if not t:
                continue
            if re.fullmatch(r"[0-9]{1,3}|[AD]|[0-9]{1,3}[-:][0-9]{1,3}", t.upper()):
                scores.append(t)
            elif re.search(r"[A-Za-z]{2,}", t):
                names.append(t)
        return {"names": names[:2], "scores": scores[:6], "raw": texts}

    def _vote(self, parsed: dict) -> dict:
        """Stabilize by keeping the most frequent recent reading."""
        key = (tuple(parsed["names"]), tuple(parsed["scores"]))
        self._history.append(key)
        most_common, _ = Counter(self._history).most_common(1)[0]
        names, scores = most_common
        return {"names": list(names), "scores": list(scores), "raw": parsed["raw"]}

    # ------------------------------------------------------------------ #
    def read(self, frame: np.ndarray, frame_index: int) -> dict:
        """Return the (stabilized) scoreboard reading, refreshing only every N frames."""
        if frame_index % self.refresh_every != 0 and self._last_result:
            return self._last_result

        crop = self._crop(frame)
        if crop is None:
            return self._last_result

        reader = self._ensure_reader()
        texts = [str(t) for t in reader.readtext(crop, detail=0)]
        self._last_result = self._vote(self._parse(texts))
        return self._last_result
