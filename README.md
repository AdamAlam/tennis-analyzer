# Tennis Match Analyzer

Real-time tennis analysis from a **live screen region** or a **video file**. Detects players,
the ball, and the court; derives game logic (in/out, rallies, serves, points); optional
scoreboard OCR and auto-highlights.

Built for an **RTX 5090 (Blackwell)** + Ryzen 7 7800X3D. Modular, swappable components.

---

## 1. Environment setup (do this first)

> The RTX 5090 is compute capability **sm_120**. You **must** use a CUDA 12.8 (`cu128`) PyTorch
> build — older wheels fail with `no kernel image` or silently fall back to CPU.

> This repo's `.venv` already has **Python 3.13.5** with **torch 2.11.0+cu128** and
> **torchvision 0.26.0+cu128** installed (Blackwell-ready), plus `mss`, `pyyaml`, `scipy`. If
> that matches your setup, just run step 2 to add the rest. Python 3.11–3.13 all work as long as
> the torch wheel is `cu128`.

```powershell
# (Only if creating a fresh env) Python 3.11, 3.12, or 3.13
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) Torch FIRST, from the cu128 index (already present in this repo's .venv)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Then the rest (installs ultralytics, opencv, pydantic, rich, filterpy, easyocr)
pip install -r requirements.txt

# 3) Verify the GPU is actually used
python scripts/check_cuda.py
```

Expected: `cuda available: True`, device `NVIDIA GeForce RTX 5090`, capability `(12, 0)`.

---

## 2. Get the models

```powershell
python scripts/download_models.py
```

This auto-downloads `yolo11n.pt` (via Ultralytics) plus the pretrained **TrackNet** (ball) and
**TennisCourtDetector** (court) weights from Google Drive (via `gdown`). If a Drive download
fails (quota/permissions), the script prints the direct links so you can place the files
manually under `models/tracknet/` and `models/court/`.

---

## 3. Pick your input

**Screen capture** — define the court region once:

```powershell
python scripts/calibrate_region.py   # drag a box over the court, paste the printed region into config/config.yaml
```

**Video file** — no calibration needed:

```powershell
python main.py --file data/match.mp4
```

---

## 4. Run the analyzer

```powershell
python main.py                       # uses config/config.yaml (source.type)
python main.py --source screen       # force live screen capture
python main.py --file data/match.mp4 # force a video file (real-time paced)
```

Press **`q`** in the preview window to quit.

---

## 5. Development roadmap

| Step | Feature | Status |
|------|---------|--------|
| 1 | Source (screen/file) + YOLO detection + overlay | ✅ implemented |
| 2 | Multi-object tracking (ByteTrack, stable player IDs) | ✅ implemented (`tracking.enabled`) |
| 3 | Tennis fine-tuning (ball + players) | dataset guide below |
| 4 | Court keypoints + homography (in/out geometry) | ✅ implemented (`court.enabled` + weights) |
| 5 | TrackNet ball model (3-frame heatmap) | ✅ implemented (`ball.backend: tracknet` + weights) |
| 6 | Game logic (bounce, serve, rally, point winner) | ✅ implemented (`logic.enabled`) |
| 7 | Scoreboard OCR (EasyOCR) | ✅ implemented (`ocr.enabled`) |
| 8 | Highlights + player position stats | ✅ implemented (`highlights.enabled`) |

All stages are implemented and wired into `pipeline.py` behind their `config.yaml` flags.
Player tracking works out of the box with the stock YOLO weights. The TrackNet ball model
(Step 5) and court detector (Step 4) need their pretrained `.pt` files - run
`python scripts/download_models.py` to fetch them, then set `detection.ball.backend: tracknet`
and `court.enabled: true`. Game logic (Step 6) needs the court homography, so enable the court
first.

---

## 6. Project layout

```
config/        config.yaml (single source of truth)
models/        weights (gitignored): yolo/ tracknet/ court/
scripts/       check_cuda, calibrate_region, download_models, record_dataset
src/tennis_analyzer/
  capture/     FrameSource: screen_capture + video_file (+ factory)
  detection/   YOLO detector + ball/ (yolo_ball MVP, tracknet stub)
  tracking/    ByteTrack wrapper (Step 2)
  court/       keypoint detector + homography (Step 4)
  logic/       events, bounce, match_state (Step 6)
  ocr/         scoreboard reader (Step 7)
  highlights/  ring-buffer clip recorder (Step 8)
  viz/         overlay drawing
  pipeline.py  orchestrates the per-frame loop
main.py        entrypoint
```

---

## 7. Datasets & fine-tuning the ball

The tiny, motion-blurred ball is the hardest part. Strategy:

- **Players:** stock YOLO11 `person` works immediately.
- **Ball:** prefer the temporal **TrackNet** model (Step 5) over a single-frame detector.
- **Fine-tune YOLO** with your own broadcast frames for best results:
  1. `python scripts/record_dataset.py` to dump frames from your feed.
  2. Label in **Roboflow** or **CVAT** — classes `player`, `ball` (tight tiny boxes, even when blurred).
  3. Train at high resolution (small objects need pixels, not depth):
     ```
     yolo detect train model=yolo11n.pt data=tennis.yaml imgsz=1280 epochs=100 batch=16
     ```
- **Public data:** Roboflow Universe tennis ball/player datasets; the TrackNet tennis dataset
  (Tsai et al.) for ball-center labels.

Tips: higher `imgsz` beats a bigger model for the ball; add Kalman smoothing to fill misses;
restrict YOLO `classes` once fine-tuned to cut false positives.
