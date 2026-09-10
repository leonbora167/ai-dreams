"""Entry point for the offline MTMC tracker POC."""
import argparse
from pathlib import Path
from datetime import datetime

from src.config import load_config, discover_videos, print_resolved_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/poc.yaml')
    p.add_argument('--videos', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    p.add_argument('--incremental', action='store_true', help='Process all camera videos frame-by-frame as live streams')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.videos:
        cfg['system']['video_dir'] = args.videos
    videos = discover_videos(cfg)
    def log(message):
        if not args.quiet: print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)
    log(f'Loaded config: {args.config}')
    log(f'Discovered {len(videos)} camera video(s) in {cfg["system"]["video_dir"]}')
    if args.dry_run:
        print_resolved_config(cfg, videos)
        return
    if args.incremental:
        from src.streaming_pipeline import StreamingPipeline
        StreamingPipeline(cfg, videos, log=log).run()
    else:
        from src.pipeline import Pipeline
        Pipeline(cfg, videos, log=log).run()


if __name__ == '__main__':
    main()
