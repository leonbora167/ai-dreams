import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

from embedder import LocalEmbedder
from face_module import FaceModule
from vector_store import AuditoriumVectorDB


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def print_results(results: list[dict], search_type: str) -> None:
    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS ({search_type.upper()})")
    print("=" * 80)
    if not results:
        print("No matching tracklets found.")
        print("=" * 80 + "\n")
        return

    for rank, res in enumerate(results, start=1):
        track_id = res.get("track_id", "Unknown")
        distance = res.get("_distance", None)
        score_str = f"Cosine Dist: {distance:.4f}" if distance is not None else "N/A"
        pose_label = res.get("pose_label", "unknown")

        start_sec = res.get("start_sec", 0.0)
        end_sec = res.get("end_sec", 0.0)

        intervals_str = ""
        intervals_json = res.get("intervals_json", "")
        if intervals_json:
            try:
                inv_list = json.loads(intervals_json)
                if isinstance(inv_list, list) and len(inv_list) > 1:
                    parts = [
                        f"[{format_timestamp(iv['start_sec'])} - {format_timestamp(iv['end_sec'])}]"
                        for iv in inv_list
                    ]
                    intervals_str = " | Intervals: " + ", ".join(parts)
            except Exception:
                pass

        time_str = f"{format_timestamp(start_sec)} - {format_timestamp(end_sec)} ({start_sec:.1f}s - {end_sec:.1f}s){intervals_str}"
        start_f = res.get("start_frame", 0)
        end_f = res.get("end_frame", 0)
        best_crop = res.get("best_crop_path", "")
        has_face = res.get("has_face", False)
        face_crop = res.get("face_crop_path", "None")

        print(f"[{rank}] Global Track ID: #{track_id} | Pose: {pose_label} | {score_str}")
        print(f"    Timeline    : {time_str} (Frames: {start_f} -> {end_f})")
        print(f"    Body Crop   : {best_crop}")
        print(f"    Face Crop   : {face_crop if has_face else 'No face detected'}")
        print("-" * 80)
    print()


def print_scene_results(results: list[dict], search_type: str) -> None:
    print("\n" + "=" * 80)
    print(f"SCENE KEYFRAME RESULTS ({search_type.upper()})")
    print("=" * 80)
    if not results:
        print("No matching scene keyframes found.")
        print("=" * 80 + "\n")
        return

    for rank, res in enumerate(results, start=1):
        distance = res.get("_distance", None)
        score_str = f"Cosine Dist: {distance:.4f}" if distance is not None else "N/A"
        ts = res.get("timestamp_sec", 0.0)
        fid = res.get("frame_id", 0)
        p_cnt = res.get("person_count", 0)
        st_cnt = res.get("standing_count", 0)
        si_cnt = res.get("sitting_count", 0)
        fpath = res.get("frame_path", "")

        print(f"[{rank}] Scene Frame #{fid} ({format_timestamp(ts)} / {ts:.1f}s) | {score_str}")
        print(f"    Counts      : Total={p_cnt}, Standing={st_cnt}, Sitting={si_cnt}")
        print(f"    Image Path  : {fpath}")
        print("-" * 80)
    print()


def extract_video_embedding(video_path: Path, embedder: LocalEmbedder, num_samples: int = 5) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return [0.0] * embedder.siglip_dim
    step = max(1, total_frames // num_samples)
    vectors = []
    for f_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            vec = embedder.extract_multimodal_image(frame)
            vectors.append(vec)
    cap.release()
    if not vectors:
        return [0.0] * embedder.siglip_dim
    avg_vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm > 0:
        avg_vec = avg_vec / norm
    return avg_vec.tolist()


def main():
    parser = argparse.ArgumentParser(description="Auditorium Video Semantic Search CLI")
    parser.add_argument("--text", type=str, default=None, help="Search by natural language query")
    parser.add_argument("--image", type=str, default=None, help="Reference person image path")
    parser.add_argument("--video", type=str, default=None, help="Reference video clip path")
    parser.add_argument("--face", type=str, default=None, help="Search by reference face image (ArcFace)")
    parser.add_argument("--mode", type=str, choices=["reid", "multimodal"], default="reid",
                        help="Image search mode: 'reid' (MSMT17 OSNet-AIN) or 'multimodal' (SigLIP)")
    parser.add_argument("--pose", type=str, choices=["standing", "sitting"], default=None,
                        help="Optional pose filter for text search")
    parser.add_argument("--scenes", action="store_true",
                        help="Search sampled full auditorium scene keyframes instead of individual person crops")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384",
                        help="SigLIP model ID (must match indexed vector dimension)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    if not (args.text or args.image or args.face or args.video):
        parser.print_help()
        sys.exit(1)

    embedder = LocalEmbedder(siglip_id=args.siglip_model) if (args.text or args.image or args.video) else None
    siglip_dim = embedder.siglip_dim if embedder is not None else 768
    vector_db = AuditoriumVectorDB(siglip_dim=siglip_dim)

    if args.scenes and args.text:
        print(f"Searching full scene keyframes for: \"{args.text}\"...")
        text_vec = embedder.extract_multimodal_text(args.text)
        results = vector_db.search_scene_frames(text_vec, top_k=args.top_k)
        print_scene_results(results, f"Scene Keyframes: '{args.text}'")
        return

    if args.text and args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        img = cv2.imread(str(img_path))
        t_vec = np.array(embedder.extract_multimodal_text(args.text), dtype=np.float32)
        i_vec = np.array(embedder.extract_multimodal_image(img), dtype=np.float32)
        fused = (t_vec + i_vec) / 2.0
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        results = vector_db.search_by_text_vector(fused.tolist(), top_k=args.top_k, pose_filter=args.pose)
        print_results(results, f"Combined: Text ('{args.text}') + Image ('{img_path.name}')")

    elif args.text and args.video:
        vid_path = Path(args.video)
        if not vid_path.exists():
            print(f"Error: Video not found: {vid_path}", file=sys.stderr)
            sys.exit(1)
        t_vec = np.array(embedder.extract_multimodal_text(args.text), dtype=np.float32)
        v_vec = np.array(extract_video_embedding(vid_path, embedder), dtype=np.float32)
        fused = (t_vec + v_vec) / 2.0
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        results = vector_db.search_by_text_vector(fused.tolist(), top_k=args.top_k, pose_filter=args.pose)
        print_results(results, f"Combined: Text ('{args.text}') + Video ('{vid_path.name}')")

    elif args.text:
        print(f"Extracting SigLIP text embedding for: \"{args.text}\"...")
        text_vec = embedder.extract_multimodal_text(args.text)
        results = vector_db.search_by_text_vector(text_vec, top_k=args.top_k, pose_filter=args.pose)
        print_results(results, f"Text: '{args.text}'")

    elif args.video:
        vid_path = Path(args.video)
        if not vid_path.exists():
            print(f"Error: Video not found: {vid_path}", file=sys.stderr)
            sys.exit(1)
        vid_vec = extract_video_embedding(vid_path, embedder)
        results = vector_db.search_by_text_vector(vid_vec, top_k=args.top_k, pose_filter=args.pose)
        print_results(results, f"Video Clip: '{vid_path.name}'")

    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Unable to read image: {img_path}", file=sys.stderr)
            sys.exit(1)

        if args.mode == "multimodal":
            print(f"Extracting SigLIP multimodal image embedding for: {img_path.name}...")
            img_vec = embedder.extract_multimodal_image(img)
            results = vector_db.search_by_text_vector(img_vec, top_k=args.top_k, pose_filter=args.pose)
            print_results(results, f"Multimodal Image: '{img_path.name}'")
        else:
            print(f"Extracting MSMT17 OSNet-AIN Re-ID embedding for: {img_path.name}...")
            body_vec = embedder.extract_body_reid(img)
            results = vector_db.search_by_body_vector(body_vec, top_k=args.top_k)
            print_results(results, f"Body Re-ID Image: '{img_path.name}'")

    elif args.face:
        face_path = Path(args.face)
        if not face_path.exists():
            print(f"Error: Face image not found: {face_path}", file=sys.stderr)
            sys.exit(1)
        face_img = cv2.imread(str(face_path))
        if face_img is None:
            print(f"Error: Unable to read image: {face_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Extracting ArcFace embedding for: {face_path.name}...")
        face_module = FaceModule()
        face_vec = face_module.extract_face_from_crop(face_img)
        if face_vec is None:
            print("Error: Could not extract face embedding from the provided reference image.", file=sys.stderr)
            sys.exit(1)

        results = vector_db.search_by_face_vector(face_vec, top_k=args.top_k)
        print_results(results, f"Face Image: '{face_path.name}'")


if __name__ == "__main__":
    main()
