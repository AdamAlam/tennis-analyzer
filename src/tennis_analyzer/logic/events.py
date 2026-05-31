"""Event types emitted by the game-logic layer (Step 6).

These dataclasses are concrete (not stubs) so other modules can import and type against them
before the detection logic itself is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Event:
    """Base class for a timestamped match event."""

    frame_index: int
    t_seconds: float


@dataclass
class Bounce(Event):
    """A ball bounce, with its court-space location and in/out verdict."""

    x_img: float
    y_img: float
    x_court_m: float | None = None
    y_court_m: float | None = None
    inside: bool | None = None


@dataclass
class Serve(Event):
    """Start of a point: a serve, attributed to a player track id."""

    server_track_id: int | None = None


@dataclass
class PointWon(Event):
    """End of a point: winner + rally length (number of shots)."""

    winner_track_id: int | None = None
    rally_length: int = 0
    reason: str = ""  # e.g. "out", "double-bounce", "net"
