"""Modular Sequential Multi-Camera (MTMC) Tracking Pipeline.

Features:
- Pluggable Detectors (YOLO11, RT-DETR, custom)
- Pluggable Trackers (ByteTrack, custom)
- Pluggable Feature Extractors (OSNet ReID, Color histograms, custom)
- Sequential camera-by-camera processing with continuous cumulative gallery memory
- Chronological arrival-order ID assignment (Frame 0/earliest starts at G1 [L1], G2 [L2]...)
- Individual per-camera review videos + synchronized multi-camera stitched grid
"""
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

from .tracklets import Tracklet, save_tracklets
from .detectors import get_detector
from .trackers import get_tracker
from .features import FeatureManager
from .visualization import render_single_camera, stitch_videos


class _WorkingTrack:
    """Buffer for collecting boxes, frames, and reservoir crops for a single tracklet."""
    def __init__(self, camera_id, local_id, fps, video_path):
        self.camera_id = camera_id
        self.local_id = local_id
        self.fps = fps
        self.video_path = video_path
        self.frames = []
        self.boxes = []
        self.confidences = []
        self.crops = []
        self.start_frame = None
        self.last_frame = -1
        self.crop_seen = 0

    def add(self, frame_no, box, confidence, frame):
        if self.start_frame is None:
            self.start_frame = frame_no
        self.frames.append(frame_no)
        self.boxes.append(box)
        self.confidences.append(confidence)
        self.last_frame = frame_no

        x1, y1, x2, y2 = map(int, box)
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if crop.size:
            self.crop_seen += 1
            if len(self.crops) < 24:
                self.crops.append(crop.copy())
            else:
                slot = (self.crop_seen * 1103515245 + 12345) % self.crop_seen
                if slot < 24:
                    self.crops[slot] = crop.copy()

    def to_tracklet(self) -> Tracklet:
        return Tracklet(
            self.camera_id,
            self.local_id,
            self.frames,
            self.boxes,
            self.confidences,
            self.fps,
            self.video_path
        )


def _assign_global_id(track, descriptor, gallery, cfg, feature_manager):
    """Match a finalized tracklet against the cumulative gallery (intra-camera & cross-camera)."""
    if descriptor.get('_skip') or not gallery:
        return max([g for _, _, g in gallery], default=0) + 1

    best_gid = None
    best_score = -1.0
    adjacency = cfg.get('cameras', {}).get('topology', {}).get('adjacency', {})

    # Overlap constraint: identify GIDs already active in this camera at overlapping times
    occupied_here = {
        gid for old_track, _, gid in gallery
        if old_track.camera_id == track.camera_id
        and max(old_track.start_time, track.start_time) <= min(old_track.end_time, track.end_time)
    }

    for old_track, old_desc, gid in gallery:
        # Cannot assign an ID if that ID is already occupied in this camera at the same time
        if gid in occupied_here:
            continue

        # Topology constraint (only applies across different cameras if topology enabled)
        if old_track.camera_id != track.camera_id and cfg.get('features', {}).get('topology', {}).get('enabled'):
            if track.camera_id not in adjacency.get(old_track.camera_id, []):
                continue

        # Temporal window constraint (within max_transition_seconds)
        gap = max(0.0, track.start_time - old_track.end_time, old_track.start_time - track.end_time)
        if cfg.get('features', {}).get('temporal', {}).get('enabled'):
            topo = cfg.get('cameras', {}).get('topology', {})
            min_gap = topo.get('min_transition_seconds', 0)
            max_gap = topo.get('max_transition_seconds', 60)
            if gap < min_gap or gap > max_gap:
                continue

        score = feature_manager.compute_similarity(descriptor, old_desc)
        if score > best_score:
            best_gid, best_score = gid, score

    sim_thresh = cfg.get('association', {}).get('similarity_threshold', 0.70)
    if best_gid is not None and best_score >= sim_thresh:
        return best_gid

    return max([g for _, _, g in gallery], default=0) + 1


class SequentialPipeline:
    """Orchestrates camera-by-camera sequential processing with continuous cumulative gallery memory."""

    def __init__(self, cfg: dict, videos: dict, log=print):
        self.cfg = cfg
        self.videos = videos
        self.log = log
        self.out = Path(cfg['system']['output_dir'])
        self.track_dir = self.out / 'tracks'
        self.feature_dir = self.out / 'features'
        self.observation_dir = self.out / 'observations'
        self.viz_dir = self.out / 'visualization'

        self.track_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        detector = get_detector(self.cfg)
        feature_manager = FeatureManager(self.cfg)
        names = list(self.videos)

        gallery = []
        all_finished = []
        tracks_by_camera = {}
        mapping = {}
        individual_video_paths = {}

        self.log(f"Sequential Pipeline: Processing {len(names)} camera stream(s) one-by-one")

        # Stage 1: Process each camera video sequentially
        for cam_idx, camera in enumerate(names, 1):
            video_path = Path(self.videos[camera])
            self.log(f"\n[{cam_idx}/{len(names)}] Processing camera: {camera} ({video_path.name})")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.log(f"  Warning: Could not open {video_path}, skipping.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Instantiate modular tracker for this camera stream
            tracker = get_tracker(self.cfg, fps)

            active = {}
            raw_tracklets = []
            obs_file = (self.observation_dir / f'{camera}.jsonl').open('w', encoding='utf-8')
            timeout = int(self.cfg.get('tracker', {}).get('track_buffer', 30))

            frame_no = 0
            with tqdm(total=total_frames or None, desc=f'tracking {camera}', unit='frame') as pbar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Modular detection
                    detections = detector.detect(frame)

                    # Modular tracking
                    tracked = tracker.update(detections)
                    seen = set()

                    if len(tracked) > 0:
                        for box, tid, conf in zip(tracked.xyxy.tolist(), tracked.tracker_id.tolist(), tracked.confidence.tolist()):
                            tid = int(tid)
                            seen.add(tid)
                            item = active.setdefault(tid, _WorkingTrack(camera, tid, fps, str(video_path)))
                            item.add(frame_no, box, float(conf), frame)

                            obs_file.write(json.dumps({
                                'camera_id': camera,
                                'frame_id': frame_no,
                                'timestamp_sec': round(frame_no / fps, 4),
                                'local_track_id': tid,
                                'bbox_xyxy': [round(float(x), 2) for x in box],
                                'confidence': round(float(conf), 5),
                                'global_id': None
                            }) + '\n')

                    for tid, item in list(active.items()):
                        if (frame_no - item.last_frame) > timeout and tid not in seen:
                            active.pop(tid)
                            raw_tracklets.append(item)

                    frame_no += 1
                    pbar.update(1)

            cap.release()
            obs_file.close()

            for item in active.values():
                raw_tracklets.append(item)
            active.clear()

            # Order tracklets by first appearance frame within this camera
            raw_tracklets.sort(key=lambda t: (t.start_frame if t.start_frame is not None else 0, t.local_id))

            # Match against cumulative gallery and assign global IDs in appearance order
            this_cam_finished = []
            for item in raw_tracklets:
                track = item.to_tracklet()
                descriptor = feature_manager.extract_crops(item.crops, track)
                gid = _assign_global_id(track, descriptor, gallery, self.cfg, feature_manager)
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
                gallery.append((track, descriptor, gid))
                this_cam_finished.append(track)
                all_finished.append(track)

            tracks_by_camera[camera] = this_cam_finished
            self.log(f"  {camera}: finalized {len(this_cam_finished)} tracklet(s)")

            # Save tracklets for this camera
            save_tracklets(self.track_dir / f'{camera}.json', this_cam_finished)

            # Update observation log with assigned global IDs
            obs_path = self.observation_dir / f'{camera}.jsonl'
            if obs_path.exists():
                lines = obs_path.read_text(encoding='utf-8').splitlines()
                updated = []
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row['global_id'] = mapping.get((row['camera_id'], int(row['local_track_id'])))
                    updated.append(json.dumps(row))
                obs_path.write_text('\n'.join(updated) + '\n', encoding='utf-8')

            # Render individual tracked video for this camera immediately
            cam_video_out = self.viz_dir / f"{camera}_tracked.mp4"
            render_single_camera(
                video_path=video_path,
                camera_id=camera,
                tracks=this_cam_finished,
                mapping=mapping,
                output_path=cam_video_out,
                display_resolution=None,
                show_local_id=True
            )
            individual_video_paths[camera] = cam_video_out
            self.log(f"  Saved: {cam_video_out}")

        # Stage 2: Stitch individual videos into a single synchronized review video
        stitched_out = self.viz_dir / "stitched_review.mp4"
        disp_res = tuple(self.cfg['system'].get('display_resolution', [1280, 720]))
        self.log(f"\nStitching {len(names)} camera videos into a single synchronized video -> {stitched_out.name}...")
        stitch_videos(individual_video_paths, stitched_out, display_resolution=disp_res)
        self.log(f"  Saved stitched video: {stitched_out}")

        # Stage 3: Save global ID lookup map
        gid_map_path = self.out / 'global_id_map.json'
        gid_map_path.write_text(json.dumps({
            f"{c}:{tid}": gid for (c, tid), gid in mapping.items()
        }, indent=2), encoding='utf-8')

        total_gids = len(set(mapping.values()))
        self.log(f"\nSequential processing and stitching complete!")
        self.log(f"Total tracklets: {len(all_finished)}")
        self.log(f"Total global IDs: {total_gids}")
        self.log(f"Output files located at: {self.out.resolve()}")

