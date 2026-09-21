"""Decoupled Stateless-Service Sequential MTMC Tracking Pipeline.

Orchestrates:
1. Stateless Detector Service (RF-DETR / YOLO / Triton Client)
2. Stateful Camera Sessions (OC-SORT / BoT-SORT tracker instances per camera)
3. Stateless Feature Service (OSNet ReID / Pose / Color embeddings, Triton-ready)
4. Stateful Global Gallery (Cumulative cross-camera identity memory)
"""
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from tqdm import tqdm

from .services import create_detector_service, StatelessFeatureService
from .state import CameraTrackerSession, GlobalGallery
from .tracklets import save_tracklets
from .visualization import render_single_camera, stitch_videos


class SequentialPipeline:
    """Sequential pipeline with decoupled stateless inference services and stateful tracking sessions."""

    def __init__(self, cfg: dict, videos: dict, log=print):
        self.cfg = cfg
        self.videos = videos
        self.log = log
        self.out = Path(cfg["system"]["output_dir"])
        self.track_dir = self.out / "tracks"
        self.feature_dir = self.out / "features"
        self.observation_dir = self.out / "observations"
        self.viz_dir = self.out / "visualization"

        self.track_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. Instantiate pure stateless inference services
        self.detector_service = create_detector_service(cfg)
        self.feature_service = StatelessFeatureService(cfg)

        # 2. Instantiate stateful cross-camera global gallery
        self.global_gallery = GlobalGallery(cfg, self.feature_service)

    def run(self):
        names = list(self.videos)
        self.log(f"Stateless-Decoupled Pipeline: Processing {len(names)} camera stream(s) sequentially")

        all_finished_tracklets = []
        tracks_by_camera = {}
        mapping = {}
        individual_video_paths = {}

        # Process each camera stream sequentially
        for cam_idx, camera in enumerate(names, 1):
            video_path = Path(self.videos[camera])
            self.log(f"\n[{cam_idx}/{len(names)}] Processing camera: {camera} ({video_path.name})")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.log(f"  Warning: Could not open {video_path}, skipping.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Stateful camera session for this stream (holding dedicated tracker instance)
            session = CameraTrackerSession(camera, self.cfg, fps, str(video_path))
            observations = []

            frame_no = 0
            with tqdm(total=total_frames or None, desc=f"tracking {camera}", unit="frame") as pbar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # 1. Stateless detection
                    detections = self.detector_service.detect(frame)

                    # 2. Stateful tracking step
                    frame_obs = session.process_frame(frame_no, frame, detections)
                    observations.extend(frame_obs)

                    frame_no += 1
                    pbar.update(1)

            cap.release()
            cam_tracklets = session.flush()
            # Sort tracklets chronologically by appearance
            cam_tracklets.sort(key=lambda t: t.start_frame)
            self.log(f"  Finalized {len(cam_tracklets)} tracklets on {camera}")

            # 3. Stateless feature extraction & stateful gallery association
            cam_desc_dict = {}
            for t in cam_tracklets:
                avg_conf = float(np.mean(t.confidences)) if t.confidences else 1.0
                desc = self.feature_service.extract_descriptors(t.crops, avg_conf=avg_conf)
                cam_desc_dict[t.track_id] = desc

                # Assign global identity
                gid = self.global_gallery.assign_identity(t, desc)
                t.global_id = gid
                mapping[(camera, t.track_id)] = gid

            all_finished_tracklets.extend(cam_tracklets)
            tracks_by_camera[camera] = cam_tracklets

            # 4. Save camera observations with assigned global IDs
            obs_file = (self.observation_dir / f"{camera}.jsonl").open("w", encoding="utf-8")
            for obs in observations:
                obs["global_id"] = mapping.get((camera, obs["local_track_id"]))
                obs_file.write(json.dumps(obs) + "\n")
            obs_file.close()

            # 5. Save tracklets JSON
            save_tracklets(self.track_dir / f"{camera}.json", cam_tracklets)

            # 6. Save extracted embeddings (.npz)
            save_arrays = {}
            for tid, d in cam_desc_dict.items():
                for feat_name, arr in d.items():
                    if isinstance(arr, np.ndarray):
                        save_arrays[f"track_{tid}_{feat_name}"] = arr
            if save_arrays:
                np.savez_compressed(self.feature_dir / f"{camera}_features.npz", **save_arrays)

            # 7. Render tracked video for this camera
            cam_video_out = self.viz_dir / f"{camera}_tracked.mp4"
            self.log(f"  Rendering tracked visualization -> {cam_video_out.name}...")
            disp_res = tuple(self.cfg.get("system", {}).get("display_resolution", [1280, 720]))
            render_single_camera(
                video_path=video_path,
                camera_id=camera,
                tracks=cam_tracklets,
                mapping=mapping,
                output_path=cam_video_out,
                display_resolution=disp_res,
            )
            individual_video_paths[camera] = str(cam_video_out)
            self.log(f"  Saved: {cam_video_out}")

        # Summary JSON
        summary = {
            "num_cameras": len(names),
            "total_tracklets": len(all_finished_tracklets),
            "total_global_ids": self.global_gallery.total_identities,
            "mapping": {f"{c}:{tid}": gid for (c, tid), gid in mapping.items()},
        }
        with (self.out / "sequential_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 8. Synchronized grid stitching
        if len(individual_video_paths) >= 2:
            stitched_out = self.viz_dir / "stitched_review.mp4"
            self.log(f"\nStitching {len(individual_video_paths)} camera videos into synchronized grid -> stitched_review.mp4...")
            stitch_videos(
                video_paths=individual_video_paths,
                output_path=stitched_out,
                display_resolution=tuple(self.cfg.get("system", {}).get("display_resolution", [1280, 720]))
            )
            self.log(f"  Saved stitched video: {stitched_out}")

        self.log(f"\nSequential processing complete!")
        self.log(f"Total tracklets: {len(all_finished_tracklets)}")
        self.log(f"Total global IDs: {self.global_gallery.total_identities}")
        self.log(f"Output files located at: {self.out}")
