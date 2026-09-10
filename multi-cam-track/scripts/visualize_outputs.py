"""Render saved tracklets/global IDs without rerunning detection or ReID."""
import argparse
from pathlib import Path
import json
import sys

# When a file inside scripts/ is executed directly, Python may not include the
# repository root on sys.path. Add it explicitly so `from src...` is reliable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, discover_videos
from src.tracklets import load_tracklets
from src.visualization import render


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/epfl_4p.yaml')
    p.add_argument('--videos', default=None)
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.videos: cfg['system']['video_dir'] = args.videos
    if args.output_dir:
        cfg['system']['output_dir'] = args.output_dir
        cfg['output']['video'] = str(Path(args.output_dir) / 'visualization' / 'multi_camera.mp4')
    root = Path(cfg['system']['output_dir'])
    tracks = []
    for path in sorted((root / 'tracks').glob('*.json')):
        tracks.extend(load_tracklets(path))
    mapping = {(t.camera_id, t.local_id): t.global_id for t in tracks if t.global_id is not None}
    if not mapping:
        mapping_file = root / 'global_id_map.json'
        if mapping_file.exists():
            mapping = {(parts[0], int(parts[1])): v for k, v in json.loads(mapping_file.read_text()).items() for parts in [k.split(':', 1)]}
    videos = discover_videos(cfg)
    print(f'Rendering {len(tracks)} saved tracklets from {len(videos)} videos')
    render(videos, mapping, tracks, cfg)
    print(f'Wrote {cfg["output"]["video"]}')


if __name__ == '__main__':
    main()
