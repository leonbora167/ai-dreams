#!/usr/bin/env python3
"""CLI entry point for Decoupled Stateless-Service Sequential MTMC Tracker.

Architectural Highlights:
- Stateless inference services: RF-DETR / YOLO detector & OSNet / Pose feature extraction
  (designed for direct deployment on Triton Inference Server).
- Stateful tracking sessions: per-camera tracker instances (OC-SORT / BoT-SORT), Kalman filter
  states, and reservoir crop buffers.
- Stateful global gallery: cumulative cross-camera re-identification memory.
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import load_config, discover_videos
from src.pipeline import SequentialPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Decoupled Stateless-Service Sequential MTMC Tracker"
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--videos", type=Path, default=None,
        help="Path to directory containing camera video files (overrides config)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Target output directory (overrides config)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.output_dir:
        cfg['system']['output_dir'] = str(args.output_dir)

    video_dir = args.videos or Path(cfg['system']['video_dir'])
    if not video_dir.exists():
        print(f"Error: Video directory '{video_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    videos = discover_videos(video_dir)
    if not videos:
        print(f"Error: No video files found in '{video_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"==================================================================")
    print(f" Starting Decoupled Stateless MTMC Tracking Pipeline")
    print(f"   Config:     {args.config}")
    print(f"   Detector:   {cfg.get('detector', {}).get('name', 'rfdetr')}")
    print(f"   Tracker:    {cfg.get('tracker', {}).get('name', 'ocsort')}")
    print(f"   Pose Feat:  {cfg.get('features', {}).get('pose', {}).get('enabled', False)}")
    print(f"   Cameras:    {len(videos)} video streams in {video_dir}")
    print(f"   Output:     {cfg['system']['output_dir']}")
    print(f"==================================================================\n")

    pipeline = SequentialPipeline(cfg, videos)
    pipeline.run()


if __name__ == "__main__":
    main()

