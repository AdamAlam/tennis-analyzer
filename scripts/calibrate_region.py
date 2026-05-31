"""Interactively pick the screen-capture region for the court.

Grabs the primary monitor, lets you drag a rectangle, and prints the top/left/width/height to
paste into config/config.yaml under ``capture.region``.

    python scripts/calibrate_region.py
    python scripts/calibrate_region.py --monitor 2
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from mss import mss


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick the screen capture region")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    args = ap.parse_args()

    with mss() as sct:
        mon = sct.monitors[args.monitor]
        shot = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA -> BGR

    print("Drag a box over the court, then press ENTER/SPACE (or C to cancel).")
    # selectROI returns (x, y, w, h) relative to the captured monitor image.
    x, y, w, h = cv2.selectROI("Select court region", shot, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("No region selected.")
        return

    # Offset by the monitor origin so the region is in absolute screen coordinates.
    top = int(mon["top"]) + int(y)
    left = int(mon["left"]) + int(x)

    print("\nPaste this into config/config.yaml under capture.region:\n")
    print("  region:")
    print(f"    top: {top}")
    print(f"    left: {left}")
    print(f"    width: {int(w)}")
    print(f"    height: {int(h)}")
    print(f"\n  (monitor: {args.monitor})")


if __name__ == "__main__":
    main()
