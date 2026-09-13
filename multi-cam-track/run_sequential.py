#!/usr/bin/env python3
"""Entry point for sequential (camera-by-camera) MTMC tracking and per-camera visualization.

Processes camera videos one-by-one:
- Detects and tracks on Camera 1 completely, extracts ReID features, assigns IDs, renders cam01_tracked.mp4.
- Then processes Camera 2 completely, matches against Camera 1, renders cam02_tracked.mp4.
- And so on for all remaining cameras.
"""
import argparse
from pathlib import Path
from datetime import datetime

from src.config import load_config, discover_videos, print_resolved_config
from src.sequential_pipeline import SequentialPipeline


def main():
    p = argparse.ArgumentParser(description="Run MTMC tracking camera-by-camera sequentially.")
    p.add_argument('--config', default='configs/pets2009.yaml', help='Path to configuration YAML file')
    p.add_argument('--videos', default=None, help='Override input video directory')
    p.add_argument('--output-dir', default=None, help='Override output directory')
    p.add_argument('--dry-run', action='store_true', help='Show resolved config without running inference')
    p.add_argument('--quiet', action='store_true', help='Suppress logging output')
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.videos:
        cfg['system']['video_dir'] = args.videos
    if args.output_dir:
        cfg['system']['output_dir'] = args.output_dir

    videos = discover_videos(cfg)

    def log(message):
        if not args.quiet:
            print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)

    log(f"Loaded config: {args.config}")
    log(f"Compute device: {cfg['detector'].get('device_name', cfg['detector']['device'])} ({cfg['detector']['device']})")
    log(f"Discovered {len(videos)} camera video(s) in {cfg['system']['video_dir']}")

    if args.dry_run:
        print_resolved_config(cfg, videos)
        return

    pipeline = SequentialPipeline(cfg, videos, log=log)
    pipeline.run()


if __name__ == '__main__':
    main()
