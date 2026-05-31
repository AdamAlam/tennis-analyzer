"""Frame sources: live screen capture and video files behind a common interface."""

from .base import FrameMeta, FrameSource
from .factory import build_source
from .screen_capture import ScreenCapture
from .video_file import VideoFileSource

__all__ = ["FrameMeta", "FrameSource", "build_source", "ScreenCapture", "VideoFileSource"]
