import argparse
import os
from pathlib import Path
import cv2
import numpy as np

from detector import PersonDetector
from tracker import AuditoriumTracker
from embedder import LocalEmbedder
from face_module import FaceModule
from vector_store import AuditoriumVectorDB
from reid_manager import OnlineReIDManager
from download_sample import download_sample

def run_pipeline(
    video_path: str = "data/raw_videos/sample.mp4",
    display: bool = False,
    max_frames: int | None = None,
    recreate_db: bool = True,
    reid_threshold: float = 0.75,
    face_threshold: float = 0.60
) -> None:
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Video file not found at {video_file}. Attempting to download sample video...")
        video_file = download_sample(video_file)

    crops_dir = Path("data/crops")
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing models and modules...")
    detector = PersonDetector()
    tracker = AuditoriumTracker(track_activation_threshold=24)
    embedder = LocalEmbedder()
    face_module = FaceModule()
    vector_db = AuditoriumVectorDB(recreate=recreate_db)
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
    print(f"Processing video: {video_file.name} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Online Re-ID Active: Body threshold = {reid_threshold:.2f}, Face threshold = {face_threshold:.2f}\n")

    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            timestamp = frame_id / fps

            # 1. Detection
            bboxes, _ = detector.detect(frame)

            # 2. Tracking & 24-frame gating
            active_tracks, newly_confirmed = tracker.update(
                detections=bboxes,
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp
            )

            # 3. Update active on-screen tracks for mutual-exclusion spatiotemporal checks
            reid_manager.update_frame_state(active_tracks, timestamp=timestamp, frame_id=frame_id)

            # 4. For newly confirmed tracks: extract embeddings and query Re-ID gallery
            for track in newly_confirmed:
                local_id = track['track_id']
                best_crop = track['best_crop']

                # Save best body crop
                crop_path = crops_dir / f"crop_local_{local_id}.jpg"
                cv2.imwrite(str(crop_path), best_crop)

                # Extract multimodal (SigLIP) & body Re-ID (OSNet)
                siglip_vec = embedder.extract_multimodal_image(best_crop)
                body_vec = embedder.extract_body_reid(best_crop)

                # Extract face embedding & crop opportunistically
                face_vec, face_crop = face_module.extract_face(best_crop)
                face_crop_path = ""
                has_face = False
                if face_vec is not None and face_crop is not None:
                    has_face = True
                    face_file = crops_dir / f"face_local_{local_id}.jpg"
                    cv2.imwrite(str(face_file), face_crop)
                    face_crop_path = str(face_file)

                # Resolve Global Identity (Online Re-ID search)
                global_id, is_reappearance, match_info = reid_manager.resolve_track_id(
                    local_id=local_id,
                    body_vec=body_vec,
                    siglip_vec=siglip_vec,
                    face_vec=face_vec,
                    has_face=has_face,
                    start_sec=track['start_sec'],
                    start_frame=track['start_frame'],
                    current_sec=timestamp,
                    current_frame=frame_id,
                    best_crop_path=str(crop_path),
                    face_crop_path=face_crop_path,
                    video_source=str(video_file)
                )

                status_str = f"Re-appearance -> Global #{global_id}" if is_reappearance else f"New Global #{global_id}"
                print(f"[Frame {frame_id:04d} / {timestamp:05.1f}s] Confirmed Local #{local_id} -> {status_str}")

            # 5. Optional display with Global ID labels
            if display:
                for act in active_tracks:
                    box = act['bbox']
                    lid = act['track_id']
                    label = reid_manager.get_display_label(lid)
                    is_confirmed = (reid_manager.get_global_id(lid) is not None)
                    color = (0, 255, 0) if is_confirmed else (0, 165, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )
                cv2.imshow("Auditorium Global Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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

    # 6. Video ended: final sync of all interval durations into LanceDB
    print("\nVideo processing finished. Syncing final intervals to LanceDB...")
    reid_manager.finalize()

    total_global_persons = len(reid_manager.gallery)
    print(f"Pipeline complete! Total unique Global Persons indexed: {total_global_persons}")
    for gid, entry in reid_manager.gallery.items():
        intervals_str = ", ".join([f"{inv['start_sec']:.1f}s-{inv['end_sec']:.1f}s" for inv in entry['intervals']])
        print(f"  - Global Person #{gid}: {len(entry['intervals'])} appearance interval(s) [{intervals_str}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditorium Tracker & VSS Pipeline with Online Re-ID")
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--display", action="store_true", help="Display visual tracking window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--reid-threshold", type=float, default=0.75, help="OSNet body cosine similarity threshold (default: 0.75)")
    parser.add_argument("--face-threshold", type=float, default=0.60, help="ArcFace cosine similarity threshold (default: 0.60)")
    parser.add_argument("--no-recreate", action="store_true", help="Append to existing DB rather than recreating table")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        display=args.display,
        max_frames=args.max_frames,
        recreate_db=not args.no_recreate,
        reid_threshold=args.reid_threshold,
        face_threshold=args.face_threshold
    )
