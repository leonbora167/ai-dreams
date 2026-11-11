import argparse
from pathlib import Path
import cv2
import numpy as np

from detector import PersonDetector
from tracker import AuditoriumTracker
from embedder import LocalEmbedder
from face_module import FaceModule
from pose_module import PoseAnalyzer
from vector_store import AuditoriumVectorDB
from reid_manager import OnlineReIDManager
from download_sample import download_sample


def run_pipeline(
    video_path: str = "data/raw_videos/sample.mp4",
    display: bool = False,
    max_frames: int | None = None,
    recreate_db: bool = True,
    reid_threshold: float = 0.72,
    face_threshold: float = 0.55,
    scene_interval_sec: float = 3.0,
    siglip_model: str = "google/siglip-base-patch16-384"
) -> None:
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Video file not found at {video_file}. Attempting to download sample video...")
        video_file = download_sample(video_file)

    crops_dir = Path("data/crops")
    scenes_dir = Path("data/scenes")
    crops_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing models and modules...")
    detector = PersonDetector()
    tracker = AuditoriumTracker(track_activation_threshold=24)
    embedder = LocalEmbedder(siglip_id=siglip_model)
    face_module = FaceModule()
    pose_analyzer = PoseAnalyzer()
    vector_db = AuditoriumVectorDB(recreate=recreate_db, siglip_dim=embedder.siglip_dim)
    reid_manager = OnlineReIDManager(
        vector_db=vector_db,
        reid_sim_threshold=reid_threshold,
        face_sim_threshold=face_threshold
    )

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_step_frames = max(1, int(round(fps * scene_interval_sec)))

    print(f"Processing video: {video_file.name} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Online Re-ID Active: MSMT17 OSNet-AIN (th={reid_threshold:.2f}) + ArcFace (th={face_threshold:.2f})")
    print(f"Scene Keyframe Sampling: Every {scene_interval_sec:.1f}s ({scene_step_frames} frames) | SigLIP: {siglip_model}\n")

    frame_id = 0
    scene_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            timestamp = frame_id / fps

            # 1. Detection & Tracking
            bboxes, _ = detector.detect(frame)
            active_tracks, newly_confirmed = tracker.update(
                detections=bboxes,
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp
            )

            # 2. Periodic full-frame Scene Keyframe Indexing (for global counting / wide Q&A)
            if frame_id == 1 or (frame_id % scene_step_frames == 0):
                scene_path = scenes_dir / f"scene_{frame_id:06d}.jpg"
                cv2.imwrite(str(scene_path), frame)
                scene_pose = pose_analyzer.analyze_scene(frame)
                scene_siglip = embedder.extract_multimodal_image(frame)
                vector_db.insert_scene_frame({
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp,
                    "frame_path": str(scene_path),
                    "person_count": scene_pose["person_count"],
                    "standing_count": scene_pose["standing_count"],
                    "sitting_count": scene_pose["sitting_count"],
                    "unknown_count": scene_pose["unknown_count"],
                    "siglip_vector": scene_siglip,
                    "video_source": str(video_file)
                })
                scene_count += 1

            # 3. Update active on-screen tracks for mutual-exclusion checks
            reid_manager.update_frame_state(active_tracks, timestamp=timestamp, frame_id=frame_id)

            # 4. Process newly confirmed tracks
            for track in newly_confirmed:
                local_id = track["track_id"]
                best_crop = track["best_crop"]
                top_crops = track.get("top_crops", [best_crop])

                crop_path = crops_dir / f"crop_local_{local_id}.jpg"
                cv2.imwrite(str(crop_path), best_crop)

                # Multimodal SigLIP (384px) & MSMT17 Body Re-ID
                siglip_vec = embedder.extract_multimodal_image(best_crop)
                body_vec = embedder.extract_body_reid(best_crop)

                # Multi-frame face scan across top 5 sharpest crops in gating window
                face_vec, face_crop = face_module.extract_best_face_from_crops(top_crops)
                face_crop_path = ""
                has_face = False
                if face_vec is not None and face_crop is not None:
                    has_face = True
                    face_file = crops_dir / f"face_local_{local_id}.jpg"
                    cv2.imwrite(str(face_file), face_crop)
                    face_crop_path = str(face_file)

                # Classify standing vs sitting pose
                pose_label = pose_analyzer.classify_crop(best_crop)

                # Resolve Global Identity
                global_id, is_reappearance, _ = reid_manager.resolve_track_id(
                    local_id=local_id,
                    body_vec=body_vec,
                    siglip_vec=siglip_vec,
                    face_vec=face_vec,
                    has_face=has_face,
                    pose_label=pose_label,
                    start_sec=track["start_sec"],
                    start_frame=track["start_frame"],
                    current_sec=timestamp,
                    current_frame=frame_id,
                    best_crop_path=str(crop_path),
                    face_crop_path=face_crop_path,
                    video_source=str(video_file)
                )

                status_str = f"Re-appearance -> Global #{global_id}" if is_reappearance else f"New Global #{global_id}"
                print(
                    f"[Frame {frame_id:04d} / {timestamp:05.1f}s] Confirmed Local #{local_id} -> "
                    f"{status_str} | Pose: {pose_label} | Face: {has_face}"
                )

            if display:
                for act in active_tracks:
                    box = act["bbox"]
                    lid = act["track_id"]
                    label = reid_manager.get_display_label(lid)
                    is_confirmed = reid_manager.get_global_id(lid) is not None
                    color = (0, 255, 0) if is_confirmed else (0, 165, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(
                        frame, label, (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                    )
                cv2.imshow("Auditorium Global Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Early stop requested by user.")
                    break

            if max_frames and frame_id >= max_frames:
                print(f"Reached maximum frame limit ({max_frames}). Stopping.")
                break

            if frame_id % 50 == 0:
                print(f"Processed {frame_id}/{total_frames} frames ({timestamp:.1f}s)...")

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    print("\nVideo processing finished. Syncing final intervals to LanceDB...")
    reid_manager.finalize()

    stats = vector_db.get_structured_stats()
    print(f"Pipeline complete! Indexed {stats['total_unique_tracks']} Global Persons & {scene_count} Scene Keyframes.")
    print(
        f"Pose Summary (Tracks) -> Standing: {stats['standing_tracks']} | "
        f"Sitting: {stats['sitting_tracks']} | Unknown: {stats['unknown_pose_tracks']} | "
        f"Faces Captured: {stats['tracks_with_faces']}"
    )
    for gid, entry in reid_manager.gallery.items():
        intervals_str = ", ".join([f"{inv['start_sec']:.1f}s-{inv['end_sec']:.1f}s" for inv in entry["intervals"]])
        print(f"  - Global Person #{gid} ({entry['pose_label']}): {len(entry['intervals'])} interval(s) [{intervals_str}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditorium Tracker & VSS Pipeline with Online Re-ID & Scene Indexing")
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--display", action="store_true", help="Display visual tracking window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--reid-threshold", type=float, default=0.72, help="OSNet-AIN MSMT17 cosine similarity threshold")
    parser.add_argument("--face-threshold", type=float, default=0.55, help="ArcFace cosine similarity threshold")
    parser.add_argument("--scene-interval", type=float, default=3.0, help="Seconds between scene keyframe captures")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384", help="SigLIP Hugging Face model ID")
    parser.add_argument("--no-recreate", action="store_true", help="Append to existing DB rather than recreating table")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        display=args.display,
        max_frames=args.max_frames,
        recreate_db=not args.no_recreate,
        reid_threshold=args.reid_threshold,
        face_threshold=args.face_threshold,
        scene_interval_sec=args.scene_interval,
        siglip_model=args.siglip_model
    )
