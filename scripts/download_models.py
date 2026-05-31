"""Fetch / locate the model weights used by the analyzer.

- YOLO11n: auto-downloaded by Ultralytics into models/yolo/.
- TrackNet (ball) and TennisCourtDetector (court): print where to place the weights, since
  these come from third-party repos without a stable pip distribution.

    python scripts/download_models.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "models" / "yolo"
TRACKNET_DIR = ROOT / "models" / "tracknet"
COURT_DIR = ROOT / "models" / "court"


def download_yolo() -> None:
    """Download yolo11n.pt via Ultralytics and place it in models/yolo/."""
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    target = YOLO_DIR / "yolo11n.pt"
    if target.exists():
        print(f"[yolo] already present: {target}")
        return

    from ultralytics import YOLO

    print("[yolo] downloading yolo11n.pt ...")
    YOLO("yolo11n.pt")  # triggers the download into the ultralytics cache / CWD
    # Move the downloaded weight next to the project if it landed in the CWD.
    cwd_weight = Path("yolo11n.pt")
    if cwd_weight.exists():
        shutil.move(str(cwd_weight), str(target))
        print(f"[yolo] saved to {target}")
    else:
        print(
            "[yolo] downloaded into the Ultralytics cache. If models/yolo/yolo11n.pt is "
            "missing, copy it there or set detection.yolo.weights to the cached path."
        )


def print_manual_instructions() -> None:
    TRACKNET_DIR.mkdir(parents=True, exist_ok=True)
    COURT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "\n[tracknet] (Step 5) Download pretrained tennis TrackNet weights and place at:\n"
        f"    {TRACKNET_DIR / 'tracknet_tennis.pt'}\n"
        "    Source: a TrackNetV2/V3 tennis repo (search 'TrackNet tennis pretrained').\n"
        "\n[court] (Step 4) Download TennisCourtDetector keypoint weights and place at:\n"
        f"    {COURT_DIR / 'court_keypoints.pt'}\n"
        "    Source: github.com/yastrebksv/TennisCourtDetector (pretrained model).\n"
    )


def main() -> None:
    download_yolo()
    print_manual_instructions()


if __name__ == "__main__":
    main()
