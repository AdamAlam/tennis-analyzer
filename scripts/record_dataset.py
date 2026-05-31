"""Dump frames from the configured source for labeling (dataset creation).

Saves every Nth frame as a JPEG into an output folder, ready to import into Roboflow/CVAT.
Works for both screen capture and video files (reuses the project config + source factory).

    python scripts/record_dataset.py --every 10 --out data/frames
    python scripts/record_dataset.py --file data/match.mp4 --every 5 --max 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running as a plain script (add project root to sys.path).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tennis_analyzer.capture.factory import build_source  # noqa: E402
from src.tennis_analyzer.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump frames for labeling")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", choices=["screen", "file"])
    ap.add_argument("--file", help="video path; implies --source file")
    ap.add_argument("--every", type=int, default=10, help="save every Nth frame")
    ap.add_argument("--max", type=int, default=0, help="stop after this many saved frames (0 = no limit)")
    ap.add_argument("--out", default="data/frames", help="output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.file:
        cfg.source.type = "file"
        cfg.source.file.path = args.file
        cfg.source.file.realtime = False  # decode as fast as possible when dumping
    elif args.source:
        cfg.source.type = args.source

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(cfg)
    saved = 0
    print(f"Saving every {args.every} frame(s) to {out_dir} (q in the preview to stop).")
    try:
        for idx, frame in enumerate(source):
            if idx % args.every == 0:
                path = out_dir / f"frame_{idx:07d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                if args.max and saved >= args.max:
                    break

            cv2.imshow("record_dataset (q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved} frames to {out_dir}.")


if __name__ == "__main__":
    main()
