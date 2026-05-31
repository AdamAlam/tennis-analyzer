"""Match state machine (Step 6) - STUB.

Target behavior
----------------
A finite state machine over detected bounces, player positions, and ball contacts that tracks:
- serve detection (ball originates near a baseline; server identified by track id),
- rally status + length (count of player-ball contacts / direction reversals),
- point end + winner (ball out, double bounce, or no contact for N frames),
optionally cross-checked against scoreboard OCR (Step 7).

States: IDLE -> SERVE -> RALLY -> POINT_OVER -> IDLE.
"""

from __future__ import annotations

from .events import Event


class MatchState:
    def __init__(self):
        raise NotImplementedError(
            "MatchState is a Step-6 stub. Implement the serve/rally/point state machine here."
        )

    def step(self, *args, **kwargs) -> list[Event]:  # pragma: no cover - stub
        """Advance the state machine one frame and return any emitted events."""
        raise NotImplementedError
