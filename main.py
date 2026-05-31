"""Tennis Match Analyzer - entrypoint.

Examples
--------
    python main.py                          # use config/config.yaml (source.type)
    python main.py --source screen          # force live screen capture
    python main.py --file data/match.mp4    # force a video file (real-time paced)
"""

from __future__ import annotations

import argparse

from src.tennis_analyzer.capture.factory import build_source
from src.tennis_analyzer.config import load_config
from src.tennis_analyzer.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Real-time tennis match analyzer")
    ap.add_argument("--config", default="config/config.yaml", help="path to config YAML")
    ap.add_argument("--source", choices=["screen", "file"], help="override source type")
    ap.add_argument("--file", help="video path to analyze; implies --source file")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides for ad-hoc runs.
    if args.file:
        cfg.source.type = "file"
        cfg.source.file.path = args.file
    elif args.source:
        cfg.source.type = args.source

    source = build_source(cfg)
    Pipeline(cfg, source).run()


if __name__ == "__main__":
    main()
