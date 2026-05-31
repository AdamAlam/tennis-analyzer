"""Fetch / locate the model weights used by the analyzer.

- YOLO11n: auto-downloaded by Ultralytics into models/yolo/.
- TrackNet (ball) and TennisCourtDetector (court): downloaded from Google Drive via gdown
  (the published yastrebksv weights). If gdown/Drive fails, manual-placement instructions are
  printed instead.

    python scripts/download_models.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "models" / "yolo"
TRACKNET_DIR = ROOT / "models" / "tracknet"
COURT_DIR = ROOT / "models" / "court"

# Published pretrained weights (Google Drive file ids).
#   Ball:  github.com/yastrebksv/TrackNet            (TrackNet, 3-frame -> heatmap)
#   Court: github.com/yastrebksv/TennisCourtDetector (14 keypoints + center)
TRACKNET_GDRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
COURT_GDRIVE_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"


def _gdrive_download(file_id: str, target: Path) -> bool:
    """Download a Google Drive file to ``target``; return True on success."""
    if target.exists():
        print(f"[weights] already present: {target}")
        return True
    try:
        import gdown
    except ImportError:
        print("[weights] gdown not installed; run `pip install -r requirements.txt`.")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = gdown.download(id=file_id, output=str(target), quiet=False)
        return out is not None and target.exists()
    except Exception as exc:  # network/quota/permission issues
        print(f"[weights] download failed for {target.name}: {exc}")
        return False


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


def download_tracknet() -> None:
    target = TRACKNET_DIR / "tracknet_tennis.pt"
    if not _gdrive_download(TRACKNET_GDRIVE_ID, target):
        print(
            "[tracknet] Manual: download the pretrained tennis TrackNet weights from\n"
            f"    https://drive.google.com/file/d/{TRACKNET_GDRIVE_ID}/view\n"
            f"    and place the file at: {target}\n"
            "    (source: github.com/yastrebksv/TrackNet)\n"
        )


def download_court() -> None:
    target = COURT_DIR / "court_keypoints.pt"
    if not _gdrive_download(COURT_GDRIVE_ID, target):
        print(
            "[court] Manual: download the TennisCourtDetector keypoint weights from\n"
            f"    https://drive.google.com/file/d/{COURT_GDRIVE_ID}/view\n"
            f"    and place the file at: {target}\n"
            "    (source: github.com/yastrebksv/TennisCourtDetector)\n"
        )


def main() -> None:
    download_yolo()
    download_tracknet()
    download_court()


if __name__ == "__main__":
    main()
