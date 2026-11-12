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
from object_detector import ObjectDetector
from object_tracker import ObjectTracker
from video_summarizer import VideoEventSummarizer


def _resolve_owner_label(raw_owner: str, reid_manager: OnlineReIDManager) -> tuple[str, str]:
    """Resolves temporary `local:<id>` owner tags to Global Track IDs once confirmed."""
    if raw_owner and raw_owner.startswith("local:"):
        try:
            lid = int(raw_owner.split(":", 1)[1])
            gid = reid_manager.get_global_id(lid)
            if gid is not None:
                entry = reid_manager.gallery.get(str(gid), {})
                return str(gid), str(entry.get("best_crop_path", ""))
        except Exception:
            pass
    if raw_owner and raw_owner != "Unknown":
        entry = reid_manager.gallery.get(str(raw_owner), {})
        return str(raw_owner), str(entry.get("best_crop_path", ""))
    return "Unknown", ""


def _bbox_iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def run_pipeline(
    video_path: str = "data/raw_videos/sample.mp4",
    display: bool = False,
    max_frames: int | None = None,
    recreate_db: bool = True,
    reid_threshold: float = 0.72,
    face_threshold: float = 0.55,
    scene_interval_sec: float = 3.0,
    siglip_model: str = "google/siglip-base-patch16-384",
    track_objects: str | None = None,
    abandon_seconds: float = 5.0,
    multi_angle_reid: bool = False,
    motion_owner_weight: bool = False,
    tile_objects: bool = False,
    summarize: bool = False,
    summary_interval: float = 8.0,
    vlm_model: str = "OpenGVLab/InternVL3-2B-hf",
    attn_impl: str = "sdpa",
    quantize: str = "none",
    endpoint_url: str | None = None,
    model_name: str = "nvidia/nemotron-4-340b",
    api_key: str | None = None,
    db_path: str = "data/lancedb",
) -> None:
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Video file not found at {video_file}. Attempting to download sample video...")
        video_file = download_sample(video_file)

    crops_dir = Path("data/crops")
    scenes_dir = Path("data/scenes")
    obj_crops_dir = Path("data/object_crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    if track_objects:
        obj_crops_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing models and modules...")
    detector = PersonDetector()
    tracker = AuditoriumTracker(track_activation_threshold=24)
    embedder = LocalEmbedder(siglip_id=siglip_model)
    face_module = FaceModule()
    pose_analyzer = PoseAnalyzer()
    vector_db = AuditoriumVectorDB(db_path=db_path, recreate=recreate_db, siglip_dim=embedder.siglip_dim)
    reid_manager = OnlineReIDManager(
        vector_db=vector_db,
        reid_sim_threshold=reid_threshold,
        face_sim_threshold=face_threshold,
        multi_angle_reid=multi_angle_reid,
    )

    object_detector = None
    object_tracker = None
    if track_objects:
        target_classes = [c.strip() for c in track_objects.split(",") if c.strip()]
        if target_classes:
            object_detector = ObjectDetector(target_classes=target_classes)

    summarizer = (
        VideoEventSummarizer(
            summary_interval=summary_interval,
            sample_step_sec=max(1.5, summary_interval / 4.0),
            scenes_dir=str(scenes_dir),
        )
        if summarize
        else None
    )

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_step_frames = max(1, int(round(fps * scene_interval_sec)))
    roi_recheck_step = max(1, int(round(fps * 0.5)))
    multi_angle_step = max(1, int(round(fps * 1.0)))

    if object_detector is not None:
        object_tracker = ObjectTracker(
            track_activation_threshold=6,
            abandon_seconds=abandon_seconds,
            fps=fps,
            motion_owner_weight=motion_owner_weight,
        )

    print(f"Processing video: {video_file.name} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(
        f"Online Re-ID Active: MSMT17 OSNet-AIN (th={reid_threshold:.2f}, multi_angle={multi_angle_reid}) "
        f"+ ArcFace (th={face_threshold:.2f})"
    )
    if object_tracker is not None:
        print(
            f"Object Tracker Active: abandon={abandon_seconds:.1f}s | tile_objects={tile_objects} "
            f"| motion_owner_weight={motion_owner_weight}"
        )
    print(f"Scene Keyframe Sampling: Every {scene_interval_sec:.1f}s ({scene_step_frames} frames) | SigLIP: {siglip_model}\n")

    frame_id = 0
    scene_count = 0
    indexed_obj_ids: set[str] = set()
    obj_crop_paths: dict[str, str] = {}
    abandoned_frame_paths: dict[str, str] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            timestamp = frame_id / fps

            # 1. Person Detection & Tracking (unchanged)
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

            # Opt-in Multi-Angle OSNet Gallery expansion for isolated confirmed tracks
            if multi_angle_reid and (frame_id % multi_angle_step == 0) and active_tracks:
                h_f, w_f = frame.shape[:2]
                for act in active_tracks:
                    lid = act["track_id"]
                    if reid_manager.get_global_id(lid) is None:
                        continue
                    abox = act["bbox"]
                    is_isolated = not any(
                        other["track_id"] != lid and _bbox_iou(abox, other["bbox"]) > 0.15
                        for other in active_tracks
                    )
                    if not is_isolated:
                        continue
                    x1 = max(0, int(abox[0]))
                    y1 = max(0, int(abox[1]))
                    x2 = min(w_f, int(abox[2]))
                    y2 = min(h_f, int(abox[3]))
                    if (x2 - x1) < 20 or (y2 - y1) < 40:
                        continue
                    acrop = frame[y1:y2, x1:x2]
                    if acrop.size == 0:
                        continue
                    avec = embedder.extract_body_reid(acrop)
                    reid_manager.maybe_add_angle_vector(lid, avec, is_isolated=True)

            discrete_events: list[str] = []

            # 4. Process newly confirmed person tracks
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
                discrete_events.append(f"At {timestamp:.1f}s confirmed [ID:{global_id}] ({pose_label})")

            # 5. Separate Open-Vocab Object Detection, Ownership Binding & Abandonment Check
            active_objs: list[dict] = []
            new_abandoned: list[dict] = []
            if object_detector is not None and object_tracker is not None:
                person_candidates = []
                for act in active_tracks:
                    lid = act["track_id"]
                    gid = reid_manager.get_global_id(lid)
                    pid = str(gid) if gid is not None else f"local:{lid}"
                    cpath = reid_manager.gallery.get(str(gid), {}).get("best_crop_path", "") if gid is not None else ""
                    person_candidates.append({
                        "person_id": pid,
                        "bbox": act["bbox"],
                        "crop_path": cpath,
                    })

                if tile_objects:
                    obj_bboxes, obj_labels = object_detector.detect_tiled(frame)
                else:
                    obj_bboxes, obj_labels = object_detector.detect(frame)

                if frame_id % roi_recheck_step == 0:
                    lost_rois = object_tracker.get_recently_lost_rois(timestamp)
                    for roi in lost_rois:
                        roi_boxes, roi_lbls = object_detector.detect_roi(frame, roi)
                        if len(roi_boxes) == 0:
                            continue
                        merged_boxes = list(obj_bboxes) if len(obj_bboxes) > 0 else []
                        merged_labels = list(obj_labels)
                        for rb, rl in zip(roi_boxes, roi_lbls):
                            if not any(_bbox_iou(rb, eb) >= 0.45 for eb in merged_boxes):
                                merged_boxes.append(rb)
                                merged_labels.append(rl)
                        obj_bboxes = np.array(merged_boxes, dtype=np.float32) if merged_boxes else np.empty((0, 4), dtype=np.float32)
                        obj_labels = merged_labels

                active_objs, new_objs, new_abandoned = object_tracker.update(
                    detections=obj_bboxes,
                    labels=obj_labels,
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    person_candidates=person_candidates,
                )

                for obj in new_objs:
                    oid = str(obj["object_id"])
                    best_obj_crop = obj["best_crop"]
                    ocrop_path = obj_crops_dir / f"obj_{oid}.jpg"
                    cv2.imwrite(str(ocrop_path), best_obj_crop)
                    obj_crop_paths[oid] = str(ocrop_path)

                    owner_id, owner_cpath = _resolve_owner_label(obj["owner_track_id"], reid_manager)
                    obj_siglip = embedder.extract_multimodal_image(best_obj_crop)

                    ab_fpath = ""
                    if obj["status"] == "abandoned" and object_tracker.active_buffers[int(oid)].get("abandoned_frame") is not None:
                        af = scenes_dir / f"abandoned_obj_{oid}_f{frame_id}.jpg"
                        cv2.imwrite(str(af), object_tracker.active_buffers[int(oid)]["abandoned_frame"])
                        ab_fpath = str(af)
                        abandoned_frame_paths[oid] = ab_fpath

                    vector_db.insert_object({
                        "object_id": oid,
                        "label": obj["label"],
                        "siglip_vector": obj_siglip,
                        "owner_track_id": owner_id,
                        "owner_crop_path": owner_cpath,
                        "status": obj["status"],
                        "abandoned_since_sec": obj["abandoned_since_sec"],
                        "abandoned_frame_path": ab_fpath,
                        "start_sec": obj["start_sec"],
                        "end_sec": obj["end_sec"],
                        "start_frame": obj["start_frame"],
                        "end_frame": obj["end_frame"],
                        "best_crop_path": str(ocrop_path),
                        "video_source": str(video_file),
                    })
                    indexed_obj_ids.add(oid)
                    print(
                        f"[Frame {frame_id:04d} / {timestamp:05.1f}s] Indexed Object [OBJ:{oid}] ({obj['label']}) "
                        f"| Bound Owner: Person #{owner_id} | Status: {obj['status']}"
                    )

                for ab in new_abandoned:
                    oid = str(ab["object_id"])
                    owner_id, owner_cpath = _resolve_owner_label(ab["owner_track_id"], reid_manager)
                    ab_path = scenes_dir / f"abandoned_obj_{oid}_f{frame_id}.jpg"
                    if ab.get("abandoned_frame") is not None:
                        cv2.imwrite(str(ab_path), ab["abandoned_frame"])
                        abandoned_frame_paths[oid] = str(ab_path)

                    print(
                        f"\n[ALERT - ABANDONED OBJECT] [OBJ:{oid}] ({ab['label']}) left unattended since "
                        f"{ab['abandoned_since_sec']:.1f}s (detected @ {timestamp:.1f}s) | "
                        f"Likely Owner: Person #{owner_id}\n"
                    )
                    discrete_events.append(
                        f"ALERT at {timestamp:.1f}s: [OBJ:{oid}] ({ab['label']}) ABANDONED "
                        f"(unattended since {ab['abandoned_since_sec']:.1f}s), likely owner [ID:{owner_id}]"
                    )
                    if oid in indexed_obj_ids:
                        vector_db.update_object(
                            object_id=oid,
                            owner_track_id=owner_id,
                            owner_crop_path=owner_cpath,
                            status="abandoned",
                            abandoned_since_sec=ab["abandoned_since_sec"],
                            abandoned_frame_path=abandoned_frame_paths.get(oid, ""),
                            end_sec=timestamp,
                            end_frame=frame_id,
                        )

            # 6. Feed Phase 2 Spatial-Windowed Video Event Summarizer
            if summarizer is not None:
                sum_persons = []
                for act in active_tracks:
                    lid = act["track_id"]
                    gid = reid_manager.get_global_id(lid)
                    pid = str(gid) if gid is not None else f"local:{lid}"
                    pose = reid_manager.gallery.get(str(gid), {}).get("pose_label", "unknown") if gid is not None else "unknown"
                    sum_persons.append({
                        "person_id": pid,
                        "bbox": act["bbox"],
                        "pose": pose,
                    })
                sum_objects = []
                for obj in active_objs:
                    owner_id, _ = _resolve_owner_label(obj["owner_track_id"], reid_manager)
                    sum_objects.append({**obj, "owner_track_id": owner_id})

                flush_reason = "abandoned_object" if new_abandoned else None
                summarizer.observe_frame(
                    frame=frame,
                    timestamp=timestamp,
                    person_items=sum_persons,
                    object_items=sum_objects,
                    discrete_events=discrete_events,
                    force_flush_reason=flush_reason,
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
                for obj in active_objs:
                    obox = [int(v) for v in obj["bbox"]]
                    ocolor = (0, 0, 255) if obj["status"] == "abandoned" else (0, 165, 255)
                    olbl = f"OBJ:{obj['object_id']} {obj['label']} ({obj['status']})"
                    cv2.rectangle(frame, (obox[0], obox[1]), (obox[2], obox[3]), ocolor, 2)
                    cv2.putText(
                        frame, olbl, (obox[0], max(15, obox[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocolor, 2
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

    # Sync final object ownership, statuses, and timelines
    if object_tracker is not None:
        final_objs = object_tracker.get_all_indexed_objects()
        for obj in final_objs:
            oid = obj["object_id"]
            if oid not in indexed_obj_ids:
                continue
            buf = object_tracker.active_buffers.get(int(oid), {})
            # Resolve every score entry in case carrier became confirmed after carrying
            resolved_scores: dict[str, float] = {}
            for raw_pid, sc in buf.get("owner_scores", {}).items():
                r_pid, _ = _resolve_owner_label(raw_pid, reid_manager)
                if r_pid != "Unknown":
                    resolved_scores[r_pid] = resolved_scores.get(r_pid, 0.0) + sc
            if resolved_scores:
                best_pid = max(resolved_scores.items(), key=lambda kv: kv[1])[0]
                best_cpath = reid_manager.gallery.get(best_pid, {}).get("best_crop_path", "")
            else:
                best_pid, best_cpath = _resolve_owner_label(obj["owner_track_id"], reid_manager)

            if obj["status"] in ("abandoned", "unattended_at_end") and oid not in abandoned_frame_paths and obj.get("abandoned_frame") is not None:
                af = scenes_dir / f"abandoned_obj_{oid}_final.jpg"
                cv2.imwrite(str(af), obj["abandoned_frame"])
                abandoned_frame_paths[oid] = str(af)

            vector_db.update_object(
                object_id=oid,
                owner_track_id=best_pid,
                owner_crop_path=best_cpath,
                status=obj["status"],
                abandoned_since_sec=obj["abandoned_since_sec"],
                abandoned_frame_path=abandoned_frame_paths.get(oid, ""),
                end_sec=obj["end_sec"],
                end_frame=obj["end_frame"],
            )

    gallery_snapshot = [
        (gid, entry["pose_label"], list(entry["intervals"]))
        for gid, entry in reid_manager.gallery.items()
    ]

    # Free detection & embedder VRAM/commit before loading local VLM for Phase 2 chunk summarization
    event_count = 0
    if summarizer is not None:
        local_to_global = {
            f"local:{lid}": str(gid)
            for lid, gid in reid_manager.local_to_global.items()
        }
        summarizer.resolve_local_ids(local_to_global)
        import gc
        import torch
        del detector
        if object_detector is not None:
            del object_detector
        del face_module
        del pose_analyzer
        del reid_manager
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from vlm_qa import HuggingFaceLocalBackend, OpenAICompatibleRemoteBackend
        if endpoint_url:
            vlm_backend = OpenAICompatibleRemoteBackend(
                endpoint_url=endpoint_url,
                model_name=model_name,
                api_key=api_key,
            )
        else:
            vlm_backend = HuggingFaceLocalBackend(
                model_id=vlm_model,
                attn_impl=attn_impl,
                quantize=quantize,
            )
        summarized_chunks = summarizer.summarize_all(vlm_backend=vlm_backend)

        # Free VLM before re-loading SigLIP to embed summary texts into LanceDB
        del vlm_backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if summarized_chunks:
            text_embedder = LocalEmbedder(siglip_id=siglip_model)
            for item in summarized_chunks:
                vec = text_embedder.extract_multimodal_text(item["summary_text"])
                vector_db.insert_video_event({
                    **item,
                    "siglip_vector": vec,
                    "video_source": str(video_file),
                })
                event_count += 1
            del text_embedder
            gc.collect()

    stats = vector_db.get_structured_stats()
    all_objs = vector_db.get_all_objects()
    print(
        f"Pipeline complete! Indexed {stats['total_unique_tracks']} Global Persons, "
        f"{len(all_objs)} Tracked Objects, {scene_count} Scene Keyframes, {event_count} Video Event Chunks."
    )
    print(
        f"Pose Summary (Tracks) -> Standing: {stats['standing_tracks']} | "
        f"Sitting: {stats['sitting_tracks']} | Unknown: {stats['unknown_pose_tracks']} | "
        f"Faces Captured: {stats['tracks_with_faces']}"
    )
    for gid, pose_lbl, invs in gallery_snapshot:
        intervals_str = ", ".join([f"{inv['start_sec']:.1f}s-{inv['end_sec']:.1f}s" for inv in invs])
        print(f"  - Global Person #{gid} ({pose_lbl}): {len(invs)} interval(s) [{intervals_str}]")
    for obj in all_objs:
        print(
            f"  - Tracked Object [OBJ:{obj['object_id']}] ({obj['label']}): "
            f"Status={obj['status']} | Owner=Person #{obj['owner_track_id']} "
            f"| Visible={obj['start_sec']:.1f}s-{obj['end_sec']:.1f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditorium Tracker & VSS Pipeline with Online Re-ID, Object Ownership & Summarization")
    parser.add_argument("--video", type=str, default="data/raw_videos/sample.mp4", help="Path to input video")
    parser.add_argument("--display", action="store_true", help="Display visual tracking window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--reid-threshold", type=float, default=0.72, help="OSNet-AIN MSMT17 cosine similarity threshold")
    parser.add_argument("--face-threshold", type=float, default=0.55, help="ArcFace cosine similarity threshold")
    parser.add_argument("--scene-interval", type=float, default=3.0, help="Seconds between scene keyframe captures")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384", help="SigLIP Hugging Face model ID")
    parser.add_argument("--db-path", type=str, default="data/lancedb", help="LanceDB directory path")
    parser.add_argument("--no-recreate", action="store_true", help="Append to existing DB rather than recreating table")

    # Phase 1: Open-Vocab Object Tracking & Abandonment
    parser.add_argument("--track-objects", type=str, default=None,
                        help="Comma-separated YOLO-World prompts, e.g. 'bag, backpack, handbag, suitcase, water bottle'")
    parser.add_argument("--abandon-seconds", type=float, default=5.0,
                        help="Seconds stationary & unattended before flagging an object as abandoned")
    parser.add_argument("--tile-objects", action="store_true",
                        help="Run 2x2 tiled YOLO-World detection for tiny objects in wide 4K frames (opt-in)")
    parser.add_argument("--multi-angle-reid", action="store_true",
                        help="Store up to 5 diverse-angle OSNet vectors per isolated track for Re-ID (opt-in)")
    parser.add_argument("--motion-owner-weight", action="store_true",
                        help="Apply 2.5x owner proximity weight while object is actively moving (opt-in)")

    # Phase 2: Spatial-Windowed Video Summarization
    parser.add_argument("--summarize", action="store_true",
                        help="Enable Phase 2 spatial-windowed VLM video event summarization")
    parser.add_argument("--summary-interval", type=float, default=8.0,
                        help="Target seconds per video event summary chunk")
    parser.add_argument("--vlm-model", type=str, default="OpenGVLab/InternVL3-2B-hf",
                        help="HF VLM model ID for --summarize (e.g. OpenGVLab/InternVL3-2B-hf or HuggingFaceTB/SmolVLM2-2.2B-Instruct)")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "sdp", "eager"],
                        help="Attention implementation for VLM during --summarize")
    parser.add_argument("--quantize", type=str, default="none", choices=["none", "4bit", "8bit"],
                        help="Optional quantization for local VLM during --summarize")
    parser.add_argument("--endpoint-url", type=str, default=None,
                        help="Optional remote OpenAI-compatible VLM URL for --summarize")
    parser.add_argument("--model-name", type=str, default="nvidia/nemotron-4-340b",
                        help="Model name identifier for remote --summarize endpoint")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Bearer API key for remote --summarize endpoint")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        display=args.display,
        max_frames=args.max_frames,
        recreate_db=not args.no_recreate,
        reid_threshold=args.reid_threshold,
        face_threshold=args.face_threshold,
        scene_interval_sec=args.scene_interval,
        siglip_model=args.siglip_model,
        track_objects=args.track_objects,
        abandon_seconds=args.abandon_seconds,
        multi_angle_reid=args.multi_angle_reid,
        motion_owner_weight=args.motion_owner_weight,
        tile_objects=args.tile_objects,
        summarize=args.summarize,
        summary_interval=args.summary_interval,
        vlm_model=args.vlm_model,
        attn_impl=args.attn_impl,
        quantize=args.quantize,
        endpoint_url=args.endpoint_url,
        model_name=args.model_name,
        api_key=args.api_key,
        db_path=args.db_path,
    )
