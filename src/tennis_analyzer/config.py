"""Typed configuration loaded and validated from ``config/config.yaml``.

Every tunable in the system lives in the YAML file and is surfaced here as a validated
pydantic model, so the rest of the code can rely on correct types and sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Leaf models
# --------------------------------------------------------------------------- #
class RegionCfg(BaseModel):
    """A rectangular pixel region (screen capture area or scoreboard sub-region)."""

    top: int
    left: int
    width: int
    height: int


class FileSourceCfg(BaseModel):
    path: str = "data/match.mp4"
    loop: bool = False
    realtime: bool = True


class SourceCfg(BaseModel):
    type: Literal["screen", "file"] = "screen"
    file: FileSourceCfg = Field(default_factory=FileSourceCfg)


class CaptureCfg(BaseModel):
    monitor: int = 1
    region: RegionCfg
    target_fps: int = 60


class YoloCfg(BaseModel):
    weights: str = "models/yolo/yolo11n.pt"
    conf: float = 0.30
    iou: float = 0.50
    imgsz: int = 640
    classes: list[int] | None = None


class BallCfg(BaseModel):
    backend: Literal["yolo", "tracknet"] = "yolo"
    tracknet_weights: str = "models/tracknet/tracknet_tennis.pt"
    conf: float = 0.50


class DetectionCfg(BaseModel):
    yolo: YoloCfg = Field(default_factory=YoloCfg)
    ball: BallCfg = Field(default_factory=BallCfg)


class TrackingCfg(BaseModel):
    tracker: str = "bytetrack.yaml"
    persist: bool = True


class CourtCfg(BaseModel):
    enabled: bool = False
    weights: str = "models/court/court_keypoints.pt"
    refresh_every: int = 30


class LogicCfg(BaseModel):
    enabled: bool = False


class OcrCfg(BaseModel):
    enabled: bool = False
    region: RegionCfg | None = None
    refresh_every: int = 15


class HighlightsCfg(BaseModel):
    enabled: bool = False
    pre_seconds: float = 6.0
    post_seconds: float = 3.0
    output_dir: str = "output/highlights"


class VizCfg(BaseModel):
    show_window: bool = True
    draw_fps: bool = True
    draw_tracks: bool = True
    window_name: str = "Tennis Analyzer"


# --------------------------------------------------------------------------- #
# Root model
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    source: SourceCfg = Field(default_factory=SourceCfg)
    capture: CaptureCfg
    device: str = "cuda:0"
    half_precision: bool = True
    detection: DetectionCfg = Field(default_factory=DetectionCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    court: CourtCfg = Field(default_factory=CourtCfg)
    logic: LogicCfg = Field(default_factory=LogicCfg)
    ocr: OcrCfg = Field(default_factory=OcrCfg)
    highlights: HighlightsCfg = Field(default_factory=HighlightsCfg)
    viz: VizCfg = Field(default_factory=VizCfg)


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    """Read and validate the YAML config into a typed :class:`AppConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
