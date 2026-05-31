"""Build the configured frame source (screen or file)."""

from __future__ import annotations

from .base import FrameSource
from .screen_capture import ScreenCapture
from .video_file import VideoFileSource


def build_source(cfg) -> FrameSource:
    """Return the :class:`FrameSource` selected by ``cfg.source`` (CLI may override first)."""
    if cfg.source.type == "file":
        f = cfg.source.file
        source: FrameSource = VideoFileSource(path=f.path, loop=f.loop, realtime=f.realtime)
    else:
        source = ScreenCapture(
            monitor=cfg.capture.monitor,
            region=cfg.capture.region.model_dump(),
            target_fps=cfg.capture.target_fps,
        )

    if getattr(cfg, "prefetch", True):
        from .threaded import ThreadedFrameSource

        source = ThreadedFrameSource(source)
    return source
