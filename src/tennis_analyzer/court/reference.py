"""Reference geometry for the tennis court (Step 4).

The pretrained TennisCourtDetector predicts 14 keypoints in a fixed order. Their canonical
positions live in a top-down reference frame (taken from the model's
``court_reference.py``), which we treat as our "court space". Mapping detected image
keypoints to these reference points gives a homography we can use for in/out tests, player
position stats, and a 2D minimap.

The reference frame is in pixels; 1 reference pixel ~= 0.01 m (the doubles court spans
286..1379 in x = 10.97 m, and the baselines span 561..2935 in y = 23.77 m).
"""

from __future__ import annotations

import numpy as np

# --- Named court lines in reference (top-down) pixel coordinates ----------------- #
BASELINE_TOP = ((286, 561), (1379, 561))
BASELINE_BOTTOM = ((286, 2935), (1379, 2935))
NET = ((286, 1748), (1379, 1748))
LEFT_DOUBLES_LINE = ((286, 561), (286, 2935))
RIGHT_DOUBLES_LINE = ((1379, 561), (1379, 2935))
LEFT_SINGLES_LINE = ((423, 561), (423, 2935))
RIGHT_SINGLES_LINE = ((1242, 561), (1242, 2935))
MIDDLE_LINE = ((832, 1110), (832, 2386))
TOP_SERVICE_LINE = ((423, 1110), (1242, 1110))
BOTTOM_SERVICE_LINE = ((423, 2386), (1242, 2386))

# The 14 keypoints, in the exact order the model outputs them (channels 0..13).
KEYPOINTS: np.ndarray = np.array(
    [
        *BASELINE_TOP,            # 0, 1
        *BASELINE_BOTTOM,         # 2, 3
        *LEFT_SINGLES_LINE,       # 4, 5
        *RIGHT_SINGLES_LINE,      # 6, 7
        *TOP_SERVICE_LINE,        # 8, 9
        *BOTTOM_SERVICE_LINE,     # 10, 11
        *MIDDLE_LINE,             # 12, 13
    ],
    dtype=np.float32,
)

# Court bounds in reference pixels (used for in/out tests).
X_DOUBLES_LEFT, X_DOUBLES_RIGHT = 286.0, 1379.0
X_SINGLES_LEFT, X_SINGLES_RIGHT = 423.0, 1242.0
Y_TOP, Y_BOTTOM = 561.0, 2935.0

# Drawn extent of the reference court (for the minimap canvas).
REF_WIDTH = 1665   # a little padding around 1379
REF_HEIGHT = 3500  # a little padding around 2935

METERS_PER_REF_PX = 0.01001


def reference_lines() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All court lines as ``(p1, p2)`` pairs in reference coordinates (for minimap drawing)."""
    return [
        BASELINE_TOP, BASELINE_BOTTOM, NET,
        LEFT_DOUBLES_LINE, RIGHT_DOUBLES_LINE,
        LEFT_SINGLES_LINE, RIGHT_SINGLES_LINE,
        TOP_SERVICE_LINE, BOTTOM_SERVICE_LINE, MIDDLE_LINE,
    ]
