"""Match state machine (Step 6).

A finite state machine over detected bounces, ball presence, and player positions that tracks
serves, rally length, and point ends:

    IDLE -> SERVE -> RALLY -> POINT_OVER -> IDLE

Heuristics (best-effort, since broadcast geometry is noisy):
- A serve starts when the ball reappears after the court has been idle.
- Each bounce extends the rally; a bounce outside the lines, a double bounce on one side, or
  the ball disappearing for too long ends the point.
- The winner is attributed to the player on the *opposite* side from the fault when possible.
"""

from __future__ import annotations

from enum import Enum

from .events import Bounce, Event, PointWon, Serve

# Net line in reference pixels (see court.reference); splits top/bottom halves.
_NET_Y = 1748.0


class _State(Enum):
    IDLE = "idle"
    SERVE = "serve"
    RALLY = "rally"
    POINT_OVER = "point_over"


def _court_side(y_court: float) -> str:
    """'far' (top half, smaller y) or 'near' (bottom half)."""
    return "far" if y_court < _NET_Y else "near"


class MatchState:
    def __init__(self, ball_lost_frames: int = 45, min_serve_gap: int = 30):
        self.ball_lost_frames = ball_lost_frames
        self.min_serve_gap = min_serve_gap

        self.state = _State.IDLE
        self.rally_length = 0
        self.server_id: int | None = None
        self.points_far = 0
        self.points_near = 0

        self._frames_without_ball = 0
        self._last_bounce_side: str | None = None
        self._point_over_frame = -(10**9)

    # ------------------------------------------------------------------ #
    def _players_by_side(self, players, homography) -> dict[str, int | None]:
        """Map 'far'/'near' -> a player track id using projected foot positions."""
        sides: dict[str, int | None] = {"far": None, "near": None}
        if not players or homography is None:
            return sides
        for det in players:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.xyxy
            foot = (0.5 * (x1 + x2), y2)  # bottom-center of the box
            try:
                _, y_court = homography.to_court(foot)
            except Exception:
                continue
            sides[_court_side(y_court)] = det.track_id
        return sides

    def _end_point(self, frame_index: int, t_seconds: float, winner_side: str | None,
                   reason: str, players, homography) -> PointWon:
        sides = self._players_by_side(players, homography)
        winner_id = sides.get(winner_side) if winner_side else None
        if winner_side == "far":
            self.points_far += 1
        elif winner_side == "near":
            self.points_near += 1
        self.state = _State.POINT_OVER
        self._point_over_frame = frame_index
        return PointWon(
            frame_index=frame_index,
            t_seconds=t_seconds,
            winner_track_id=winner_id,
            rally_length=self.rally_length,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    def step(self, frame_index: int, t_seconds: float, ball_xy, bounce: Bounce | None = None,
             players=None, homography=None, singles: bool = True) -> list[Event]:
        """Advance the FSM one frame and return any emitted events."""
        events: list[Event] = []

        if ball_xy is None:
            self._frames_without_ball += 1
        else:
            self._frames_without_ball = 0

        # --- POINT_OVER: brief settle, then return to IDLE ---------------- #
        if self.state == _State.POINT_OVER:
            if self._frames_without_ball > 5 or (frame_index - self._point_over_frame) > self.min_serve_gap:
                self.state = _State.IDLE
                self.rally_length = 0
                self._last_bounce_side = None
            return events

        # --- IDLE: wait for the ball to (re)appear -> serve --------------- #
        if self.state == _State.IDLE:
            if ball_xy is not None:
                sides = self._players_by_side(players, homography)
                # Server is whoever is on the side the ball is currently nearer to.
                self.server_id = None
                if homography is not None:
                    try:
                        _, y_court = homography.to_court(ball_xy)
                        self.server_id = sides.get(_court_side(y_court))
                    except Exception:
                        pass
                self.state = _State.SERVE
                self.rally_length = 0
                events.append(Serve(frame_index=frame_index, t_seconds=t_seconds,
                                    server_track_id=self.server_id))
            return events

        # --- SERVE -> RALLY on the first bounce --------------------------- #
        if self.state == _State.SERVE:
            if bounce is not None:
                self.state = _State.RALLY
                self.rally_length = 1
                self._last_bounce_side = self._bounce_side(bounce)
                if bounce.inside is False:
                    events.append(self._end_point(frame_index, t_seconds,
                                                  self._opposite(self._last_bounce_side),
                                                  "service-fault", players, homography))
            elif self._frames_without_ball > self.ball_lost_frames:
                self.state = _State.IDLE
            return events

        # --- RALLY -------------------------------------------------------- #
        if self.state == _State.RALLY:
            if bounce is not None:
                side = self._bounce_side(bounce)
                self.rally_length += 1
                if bounce.inside is False:
                    events.append(self._end_point(frame_index, t_seconds,
                                                  self._opposite(side), "out",
                                                  players, homography))
                elif side is not None and side == self._last_bounce_side:
                    # Two bounces same side without being returned -> point to the hitter.
                    events.append(self._end_point(frame_index, t_seconds,
                                                  self._opposite(side), "double-bounce",
                                                  players, homography))
                self._last_bounce_side = side
            elif self._frames_without_ball > self.ball_lost_frames:
                events.append(self._end_point(frame_index, t_seconds, None,
                                              "ball-lost", players, homography))
            return events

        return events

    # ------------------------------------------------------------------ #
    @staticmethod
    def _bounce_side(bounce: Bounce) -> str | None:
        if bounce.y_court_m is None:
            return None
        # y_court_m is meters from the top baseline; net is ~11.885 m down.
        return "far" if bounce.y_court_m < 11.885 else "near"

    @staticmethod
    def _opposite(side: str | None) -> str | None:
        if side == "far":
            return "near"
        if side == "near":
            return "far"
        return None

    @property
    def score(self) -> dict:
        return {
            "state": self.state.value,
            "rally": self.rally_length,
            "points_far": self.points_far,
            "points_near": self.points_near,
            "server": self.server_id,
        }
