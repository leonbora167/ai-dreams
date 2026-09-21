from pathlib import Path
import copy
import yaml

from .device import resolve_device, get_device_name

DEFAULTS = {
    'system': {'video_dir': './data/videos', 'output_dir': './outputs', 'display_resolution': [1280, 720]},
    'cameras': {'auto_discover': True, 'topology': {'adjacency': {}, 'max_transition_seconds': 30, 'min_transition_seconds': 1}},
    'detector': {'model': 'yolo11n.pt', 'device': 'auto', 'confidence': .35, 'classes': [0]},
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
    raw_device = cfg.get('detector', {}).get('device', 'auto')
    cfg['detector']['device'] = resolve_device(raw_device)
    cfg['detector']['device_name'] = get_device_name(cfg['detector']['device'])
    return cfg


import re


def discover_videos(cfg_or_path):
    if isinstance(cfg_or_path, dict):
        root = Path(cfg_or_path['system']['video_dir'])
    else:
        root = Path(cfg_or_path)
    paths = list(root.glob('*.mp4'))
    
    # Natural sort key: handles both 'cam01, cam02' and 'cam1, cam2, ..., cam10' correctly
    def natural_key(p):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]
    
    paths.sort(key=natural_key)
    return {p.stem: p for p in paths}


def print_resolved_config(cfg, videos):
    print(yaml.safe_dump(cfg, sort_keys=False))
    print('Discovered cameras:', ', '.join(f'{k}={v}' for k, v in videos.items()) or '(none)')
