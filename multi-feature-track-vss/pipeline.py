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
from download_sample import download_sample

def run_pipeline(
    video_path: str = "data/raw_videos/sample.mp4",
    display: bool = False,
    max_frames: int | None = None
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
    vector_db = AuditoriumVectorDB()

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_file.name} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    frame_id = 0
    confirmed_track_ids = set()

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

            # 3. For newly confirmed tracks: extract embeddings and index
            for track in newly_confirmed:
                track_id = track['track_id']
                confirmed_track_ids.add(track_id)
                best_crop = track['best_crop']

                # Save best body crop
                crop_path = crops_dir / f"crop_{track_id}.jpg"
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
                    face_file = crops_dir / f"face_{track_id}.jpg"
                    cv2.imwrite(str(face_file), face_crop)
                    face_crop_path = str(face_file)

                # Index in LanceDB
                record = {
                    "track_id": str(track_id),
                    "siglip_vector": siglip_vec,
                    "body_vector": body_vec,
                    "face_vector": face_vec,
                    "has_face": has_face,
                    "start_sec": track['start_sec'],
                    "end_sec": timestamp,
                    "start_frame": track['start_frame'],
                    "end_frame": frame_id,
                    "best_crop_path": str(crop_path),
                    "face_crop_path": face_crop_path,
                    "video_source": str(video_file)
                }
                vector_db.insert_track(record)
                print(f"[Frame {frame_id:04d}] Indexed Track #{track_id} (Face detected: {has_face})")

            # Optional display
            if display:
                for act in active_tracks:
                    box = act['bbox']
                    tid = act['track_id']
                    color = (0, 255, 0) if tid in confirmed_track_ids else (0, 165, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(
                        frame,
                        f"ID: {tid}",
                        (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                cv2.imshow("Auditorium Tracker", frame)
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

    # 4. Video ended: update final end_sec and end_frame for all indexed tracks
    print("Video processing finished. Updating final durations in LanceDB...")
    for tid in confirmed_track_ids:
        summary = tracker.get_track_summary(tid)
        if summary:
            vector_db.update_track_duration(
                track_id=str(tid),
                end_sec=summary['end_sec'],
                end_frame=summary['end_frame']
            )

    print(f"Pipeline complete! Total indexed tracks: {len(confirmed_track_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditorium Tracker & VSS Pipeline")
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--display", action="store_true", help="Display visual tracking window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    run_pipeline(video_path=args.video, display=args.display, max_frames=args.max_frames)

