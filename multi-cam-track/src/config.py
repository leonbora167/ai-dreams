from pathlib import Path
import copy
import yaml

DEFAULTS = {
    'system': {'video_dir': './data/videos', 'output_dir': './outputs', 'display_resolution': [1280, 720]},
    'cameras': {'auto_discover': True, 'topology': {'adjacency': {}, 'max_transition_seconds': 30, 'min_transition_seconds': 1}},
    'detector': {'model': 'yolo11n.pt', 'device': 'cuda', 'confidence': .35, 'classes': [0]},
    'tracker': {'type': 'bytetrack', 'track_high_thresh': .5, 'track_low_thresh': .1, 'new_track_thresh': .6, 'track_buffer': 30},
    'features': {
        'reid': {'enabled': True, 'model': 'osnet_x1_0', 'input_size': [256, 128], 'weight': .6},
        'color': {'enabled': True, 'weight': .15}, 'motion': {'enabled': False, 'weight': .1},
        'temporal': {'enabled': True, 'weight': .15}, 'topology': {'enabled': True},
        'quality': {'enabled': True, 'min_confidence': .5, 'min_crop_pixels': [40, 90]},
        'geometry': {'enabled': False}, 'pose': {'enabled': False}},
    'association': {'similarity_threshold': .75, 'clustering_method': 'average'},
    'output': {'video': 'outputs/visualization/multi_camera.mp4', 'save_tracklets': True},
}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): _merge(a[k], v)
        else: a[k] = v


def load_config(path):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path, encoding='utf-8') as f: _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def discover_videos(cfg):
    root = Path(cfg['system']['video_dir'])
    paths = sorted(root.glob('*.mp4'))
    return {p.stem: p for p in paths}


def print_resolved_config(cfg, videos):
    print(yaml.safe_dump(cfg, sort_keys=False))
    print('Discovered cameras:', ', '.join(f'{k}={v}' for k, v in videos.items()) or '(none)')
