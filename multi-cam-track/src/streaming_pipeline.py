"""Incremental multi-camera pipeline for live-like processing of video files.

Each input is treated as an independent stream. The outer loop advances one
frame per camera at a time, so no camera must finish before the others start.
Tracklets are finalized after a configurable inactivity window, features are
saved immediately, and the global gallery is updated when a tracklet closes.
"""
from collections import defaultdict
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

from .tracklets import Tracklet, save_tracklets
from .features import FeatureManager
from .visualization import render


class StreamingPipeline:
    def __init__(self, cfg, videos, log=print):
        self.cfg, self.videos, self.log = cfg, videos, log
        self.out = Path(cfg['system']['output_dir'])
        self.track_dir = self.out / 'tracks'
        self.feature_dir = self.out / 'features'
        self.observation_dir = self.out / 'observations'
        self.track_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        from ultralytics import YOLO
        names = list(self.videos)
        caps = {c: cv2.VideoCapture(str(self.videos[c])) for c in names}
        models = {c: YOLO(self.cfg['detector']['model']) for c in names}
        fps = {c: (caps[c].get(cv2.CAP_PROP_FPS) or 30.0) for c in names}
        active = {c: {} for c in names}
        finished = []
        descriptors = []
        mapping = {}
        gallery = []
        manager = FeatureManager(self.cfg)
        frame_no = {c: 0 for c in names}
        exhausted = set()
        observation_files = {c: (self.observation_dir / f'{c}.jsonl').open('w', encoding='utf-8') for c in names}
        total_frames = sum(int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT) or 0) for c in names)
        progress = tqdm(total=total_frames or None, desc='incremental streams', unit='frame')
        self.log(f'Streaming mode: {len(names)} camera(s), interleaving frames')

        try:
            while len(exhausted) < len(names):
                for camera in names:
                    if camera in exhausted: continue
                    ok, frame = caps[camera].read()
                    if not ok:
                        exhausted.add(camera)
                        self._finalize_all(camera, active[camera], manager, finished, descriptors, gallery, mapping, fps[camera])
                        self.log(f'  {camera}: stream ended; finalized all active tracklets')
                        continue
                    progress.update(1)
                    current = frame_no[camera]; frame_no[camera] += 1
                    result = models[camera].track(
                        source=frame, persist=True, tracker='bytetrack.yaml',
                        conf=self.cfg['detector']['confidence'], classes=self.cfg['detector']['classes'],
                        device=self.cfg['detector']['device'], verbose=False)[0]
                    seen = set()
                    if result.boxes is not None and result.boxes.id is not None:
                        for box, tid, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):
                            tid = int(tid); seen.add(tid)
                            item = active[camera].setdefault(tid, _WorkingTrack(camera, tid, fps[camera], str(self.videos[camera])))
                            item.add(current, box, float(conf), frame)
                            observation_files[camera].write(json.dumps({
                                'camera_id': camera, 'frame_id': current,
                                'timestamp_sec': round(current / fps[camera], 4),
                                'local_track_id': tid, 'bbox_xyxy': [round(float(x), 2) for x in box],
                                'confidence': round(float(conf), 5), 'global_id': None
                            }) + '\n')
                    timeout = int(self.cfg['tracker'].get('track_buffer', 30))
                    for tid, item in list(active[camera].items()):
                        if current - item.last_frame > timeout and tid not in seen:
                            active[camera].pop(tid)
                            self._finalize(item, manager, finished, descriptors, gallery, mapping)
                    if current and current % 500 == 0:
                        self.log(f'  {camera}: processed {current} frames; {len(finished)} tracklets finalized')
        finally:
            for cap in caps.values(): cap.release()
            for file in observation_files.values(): file.close()
            progress.close()
        for camera in names:
            self._finalize_all(camera, active[camera], manager, finished, descriptors, gallery, mapping, fps[camera])
        self._renumber_ids_by_first_appearance(finished, mapping)
        for camera in names:
            save_tracklets(self.track_dir / f'{camera}.json', [t for t in finished if t.camera_id == camera])
        (self.out / 'global_id_map.json').write_text(json.dumps({
            f'{camera}:{local}': gid for (camera, local), gid in mapping.items()
        }, indent=2), encoding='utf-8')
        for camera in names:
            observations = self.observation_dir / f'{camera}.jsonl'
            if not observations.exists(): continue
            rows = []
            for line in observations.read_text(encoding='utf-8').splitlines():
                if not line: continue
                row = json.loads(line)
                row['global_id'] = mapping.get((row['camera_id'], int(row['local_track_id'])))
                rows.append(json.dumps(row))
            observations.write_text(('\n'.join(rows) + '\n') if rows else '', encoding='utf-8')
        self.log(f'Association complete: {len(finished)} tracklets, {len(set(mapping.values()))} global identities')
        self.log('Rendering stitched output after all streams finished')
        render(self.videos, mapping, finished, self.cfg)
        self.log(f'Complete. Visualization: {self.cfg["output"]["video"]}')

    def _renumber_ids_by_first_appearance(self, tracks, mapping):
        """Make displayed IDs chronological instead of finalization-order IDs."""
        first_seen = {}
        for track in tracks:
            first_seen.setdefault(track.global_id, track.start_time)
        remap = {old: new for new, (old, _) in enumerate(sorted(first_seen.items(), key=lambda x: x[1]), 1)}
        for track in tracks:
            track.global_id = remap[track.global_id]
            mapping[(track.camera_id, track.local_id)] = track.global_id
            path = self.feature_dir / f'{track.camera_id}_{track.local_id}.npz'
            if path.exists():
                data = dict(np.load(path, allow_pickle=False)); data['global_id'] = np.asarray(track.global_id)
                np.savez_compressed(path, **data)

    def _finalize_all(self, camera, active, manager, finished, descriptors, gallery, mapping, fps):
        for item in list(active.values()):
            self._finalize(item, manager, finished, descriptors, gallery, mapping)
        active.clear()

    def _finalize(self, item, manager, finished, descriptors, gallery, mapping):
        track = item.to_tracklet()
        descriptor = manager.extract_crops(item.crops, track)
        gid = _assign_global(track, descriptor, gallery, self.cfg)
        track.global_id = gid
        feature_path = self.feature_dir / f'{track.camera_id}_{track.local_id}.npz'
        np.savez_compressed(
            feature_path,
            **{k: v for k, v in descriptor.items() if not k.startswith('_')},
            camera_id=np.asarray(track.camera_id),
            local_track_id=np.asarray(track.local_id),
            global_id=np.asarray(gid),
        )
        mapping[(track.camera_id, track.local_id)] = gid
        gallery.append((track, descriptor, gid)); finished.append(track); descriptors.append(descriptor)


class _WorkingTrack:
    def __init__(self, camera_id, local_id, fps, video_path):
        self.camera_id, self.local_id, self.fps, self.video_path = camera_id, local_id, fps, video_path
        self.frames, self.boxes, self.confidences, self.crops = [], [], [], []
        self.last_frame = -1

    def add(self, frame_no, box, confidence, frame):
        self.frames.append(frame_no); self.boxes.append(box); self.confidences.append(confidence); self.last_frame = frame_no
        x1, y1, x2, y2 = map(int, box)
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if crop.size:
            # Keep representative crops from the whole tracklet, not only its beginning.
            self.crop_seen = getattr(self, 'crop_seen', 0) + 1
            if len(self.crops) < 24:
                self.crops.append(crop.copy())
            else:
                slot = (self.crop_seen * 1103515245 + 12345) % self.crop_seen
                if slot < 24: self.crops[slot] = crop.copy()

    def to_tracklet(self):
        return Tracklet(self.camera_id, self.local_id, self.frames, self.boxes, self.confidences, self.fps, self.video_path)


def _similarity(a, b):
    values, weights = [], []
    for name, spec in a['_cfg']['features'].items():
        if not spec.get('enabled') or name in ('topology','temporal','quality','geometry','pose'): continue
        if name in a and name in b:
            x, y = a[name], b[name]
            values.append(float(np.dot(x, y) / ((np.linalg.norm(x)*np.linalg.norm(y))+1e-8)))
            weights.append(spec.get('weight', 0))
    return float(np.average(values, weights=weights)) if values else -1.


def _assign_global(track, descriptor, gallery, cfg):
    if descriptor.get('_skip'): return max([g for _,_,g in gallery], default=0) + 1
    descriptor['_cfg'] = cfg
    best, best_score = None, -1.
    adjacency = cfg['cameras']['topology'].get('adjacency', {})
    # A global identity may legitimately appear in multiple cameras at once,
    # but two simultaneous local tracks in the same camera cannot be the same
    # person. Reserve those IDs before greedy matching.
    occupied_here = {
        gid for old_track, _, gid in gallery
        if old_track.camera_id == track.camera_id
        and max(old_track.start_time, track.start_time) <= min(old_track.end_time, track.end_time)
    }
    for old_track, old_desc, gid in gallery:
        if old_track.camera_id == track.camera_id: continue
        if gid in occupied_here: continue
        if cfg['features'].get('topology', {}).get('enabled') and track.camera_id not in adjacency.get(old_track.camera_id, []): continue
        gap = max(track.start_time-old_track.end_time, old_track.start_time-track.end_time)
        if cfg['features'].get('temporal', {}).get('enabled'):
            topo = cfg['cameras']['topology']
            if gap < topo.get('min_transition_seconds', 1) or gap > topo.get('max_transition_seconds', 30): continue
        old_desc['_cfg'] = cfg
        score = _similarity(descriptor, old_desc)
        if score > best_score: best, best_score = gid, score
    if best is not None and best_score >= cfg['association']['similarity_threshold']: return best
    return max([g for _,_,g in gallery], default=0) + 1
