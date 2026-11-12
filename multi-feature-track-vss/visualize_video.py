"""
Standalone Annotated Video Visualizer (zero changes to existing pipeline modules).

Reads an input video, runs the existing detection, tracking, Re-ID, and optional
open-vocabulary object/abandonment modules, and writes an annotated MP4 video
showing Global Track IDs, bounding boxes, and abandonment alerts.
"""

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
from object_detector import ObjectDetector
from object_tracker import ObjectTracker


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


def _id_color(key: str) -> tuple[int, int, int]:
    """Deterministic BGR color for a given track/object ID."""
    palette = [
        (80, 220, 100),   # green
        (255, 190, 50),   # sky blue
        (220, 130, 255),  # pink-violet
        (60, 210, 245),   # yellow
        (255, 140, 80),   # royal blue
        (160, 240, 200),  # mint
        (200, 180, 255),  # rose
        (240, 220, 110),  # cyan
    ]
    h = sum(ord(c) for c in str(key))
    return palette[h % len(palette)]


def _draw_label_box(
    frame: np.ndarray,
    bbox: list[float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.55,
) -> None:
    h, w = frame.shape[:2]
    x1 = max(0, min(int(round(bbox[0])), w - 1))
    y1 = max(0, min(int(round(bbox[1])), h - 1))
    x2 = max(0, min(int(round(bbox[2])), w - 1))
    y2 = max(0, min(int(round(bbox[3])), h - 1))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    pad = 4
    top = max(0, y1 - th - pad * 2)
    bot = top + th + pad * 2
    right = min(w - 1, x1 + tw + pad * 2)
    cv2.rectangle(frame, (x1, top), (right, bot), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + pad, bot - pad - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


def _resolve_owner(raw_owner: str, reid_manager: OnlineReIDManager) -> str:
    if raw_owner.startswith("local:"):
        lid = int(raw_owner.split(":", 1)[1])
        gid = reid_manager.get_global_id(lid)
        if gid is not None:
            return f"ID:{gid}"
        return raw_owner
    if raw_owner and raw_owner != "Unknown":
        return f"ID:{raw_owner}"
    return "Unknown"


def render_annotated_video(
    video_path: str,
    output_path: str = "data/annotated_output.mp4",
    max_frames: int | None = None,
    reid_threshold: float = 0.72,
    face_threshold: float = 0.55,
    siglip_model: str = "google/siglip-base-patch16-384",
    track_objects: str | None = None,
    abandon_seconds: float = 5.0,
    tile_objects: bool = False,
    multi_angle_reid: bool = False,
    motion_owner_weight: bool = False,
    show_persons: bool = True,
    show_objects: bool = True,
    show_abandonment: bool = True,
    show_hud: bool = True,
    only_abandoned: bool = False,
    summary_slate: bool = True,
    display: bool = False,
    db_path: str = "data/lancedb_viz",
) -> Path:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    crops_dir = Path("data/viz_crops")
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing models for standalone visualization...")
    detector = PersonDetector()
    tracker = AuditoriumTracker(track_activation_threshold=24)
    embedder = LocalEmbedder(siglip_id=siglip_model)
    face_module = FaceModule()
    pose_analyzer = PoseAnalyzer()
    vector_db = AuditoriumVectorDB(db_path=db_path, recreate=True, siglip_dim=embedder.siglip_dim)
    reid_manager = OnlineReIDManager(
        vector_db=vector_db,
        reid_sim_threshold=reid_threshold,
        face_sim_threshold=face_threshold,
        multi_angle_reid=multi_angle_reid,
    )

    object_detector = None
    if track_objects:
        target_classes = [c.strip() for c in track_objects.split(",") if c.strip()]
        if target_classes:
            object_detector = ObjectDetector(target_classes=target_classes)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    object_tracker = (
        ObjectTracker(
            track_activation_threshold=6,
            abandon_seconds=abandon_seconds,
            fps=fps,
            motion_owner_weight=motion_owner_weight,
        )
        if object_detector is not None
        else None
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for {out_file}")

    roi_recheck_step = max(1, int(round(fps * 0.5)))
    multi_angle_step = max(1, int(round(fps * 1.0)))
    font_scale = max(0.5, min(1.0, width / 1920.0 * 0.75))
    box_thick = max(2, int(round(width / 1280.0 * 2)))

    frame_id = 0
    last_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            timestamp = frame_id / fps
            vis = frame.copy()
            last_frame = frame

            # 1. Person Detection & Tracking
            bboxes, _ = detector.detect(frame)
            active_tracks, newly_confirmed = tracker.update(
                detections=bboxes,
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            reid_manager.update_frame_state(active_tracks, timestamp=timestamp, frame_id=frame_id)

            if multi_angle_reid and (frame_id % multi_angle_step == 0) and active_tracks:
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
                    x1, y1 = max(0, int(abox[0])), max(0, int(abox[1]))
                    x2, y2 = min(width, int(abox[2])), min(height, int(abox[3]))
                    if (x2 - x1) >= 20 and (y2 - y1) >= 40:
                        reid_manager.maybe_add_angle_vector(
                            lid, embedder.extract_body_reid(frame[y1:y2, x1:x2]), is_isolated=True
                        )

            for track in newly_confirmed:
                local_id = track["track_id"]
                best_crop = track["best_crop"]
                top_crops = track.get("top_crops", [best_crop])
                crop_path = crops_dir / f"viz_local_{local_id}.jpg"
                cv2.imwrite(str(crop_path), best_crop)

                siglip_vec = embedder.extract_multimodal_image(best_crop)
                body_vec = embedder.extract_body_reid(best_crop)
                face_vec, _ = face_module.extract_best_face_from_crops(top_crops)
                pose_label = pose_analyzer.classify_crop(best_crop)
                reid_manager.resolve_track_id(
                    local_id=local_id,
                    body_vec=body_vec,
                    siglip_vec=siglip_vec,
                    face_vec=face_vec,
                    has_face=face_vec is not None,
                    pose_label=pose_label,
                    start_sec=track["start_sec"],
                    start_frame=track["start_frame"],
                    current_sec=timestamp,
                    current_frame=frame_id,
                    best_crop_path=str(crop_path),
                    face_crop_path="",
                    video_source=str(video_file),
                )

            # 2. Object Detection & Abandonment Tracking
            active_objs: list[dict] = []
            if object_detector is not None and object_tracker is not None:
                person_candidates = []
                for act in active_tracks:
                    lid = act["track_id"]
                    gid = reid_manager.get_global_id(lid)
                    pid = str(gid) if gid is not None else f"local:{lid}"
                    person_candidates.append({
                        "person_id": pid,
                        "bbox": act["bbox"],
                        "crop_path": "",
                    })

                if tile_objects:
                    obj_bboxes, obj_labels = object_detector.detect_tiled(frame)
                else:
                    obj_bboxes, obj_labels = object_detector.detect(frame)

                if frame_id % roi_recheck_step == 0:
                    for roi in object_tracker.get_recently_lost_rois(timestamp):
                        roi_boxes, roi_lbls = object_detector.detect_roi(frame, roi)
                        if len(roi_boxes) > 0:
                            merged_boxes = list(obj_bboxes) if len(obj_bboxes) > 0 else []
                            merged_labels = list(obj_labels)
                            for rb, rl in zip(roi_boxes, roi_lbls):
                                if not any(_bbox_iou(rb, eb) >= 0.45 for eb in merged_boxes):
                                    merged_boxes.append(rb)
                                    merged_labels.append(rl)
                            obj_bboxes = (
                                np.array(merged_boxes, dtype=np.float32)
                                if merged_boxes
                                else np.empty((0, 4), dtype=np.float32)
                            )
                            obj_labels = merged_labels

                active_objs, _, _ = object_tracker.update(
                    detections=obj_bboxes,
                    labels=obj_labels,
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    person_candidates=person_candidates,
                )

            abandoned_objs = [o for o in active_objs if o["status"] == "abandoned"]
            abandoned_owner_keys = {
                _resolve_owner(o["owner_track_id"], reid_manager) for o in abandoned_objs
            }

            # 3. Draw Person Boxes & Track IDs
            if show_persons:
                for trk in active_tracks:
                    lid = trk["track_id"]
                    gid = reid_manager.get_global_id(lid)
                    pid_label = f"ID:{gid}" if gid is not None else f"local:{lid}"
                    if only_abandoned and pid_label not in abandoned_owner_keys:
                        continue
                    pose = (
                        reid_manager.gallery.get(str(gid), {}).get("pose_label", "")
                        if gid is not None
                        else ""
                    )
                    lbl = f"{pid_label} ({pose})" if pose else pid_label
                    color = (0, 215, 255) if pid_label in abandoned_owner_keys else _id_color(pid_label)
                    _draw_label_box(vis, trk["bbox"], lbl, color, thickness=box_thick, font_scale=font_scale)

            # 4. Draw Object Boxes & Abandonment Status
            if show_objects and active_objs:
                for obj in active_objs:
                    st = obj["status"]
                    if only_abandoned and st != "abandoned":
                        continue
                    owner_str = _resolve_owner(obj["owner_track_id"], reid_manager)
                    if st == "abandoned" and show_abandonment:
                        color = (0, 0, 255)
                        since = obj.get("abandoned_since_sec", -1.0)
                        lbl = f"ABANDONED [OBJ:{obj['object_id']}] {obj['label']} | Owner:{owner_str} (@{since:.1f}s)"
                        thick = box_thick + 2
                    elif st == "stationary":
                        color = (0, 165, 255)
                        lbl = f"[OBJ:{obj['object_id']}] {obj['label']} | Owner:{owner_str} (stationary)"
                        thick = box_thick
                    else:
                        color = (60, 230, 180)
                        lbl = f"[OBJ:{obj['object_id']}] {obj['label']} | Owner:{owner_str}"
                        thick = box_thick
                    _draw_label_box(vis, obj["bbox"], lbl, color, thickness=thick, font_scale=font_scale)

            # 5. Abandonment Alert Banner
            if show_abandonment and abandoned_objs:
                banner_h = max(44, int(height * 0.055))
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, height - banner_h), (width, height), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)
                alert_parts = [
                    f"[OBJ:{o['object_id']}] {o['label']} (Owner {_resolve_owner(o['owner_track_id'], reid_manager)}, since {o['abandoned_since_sec']:.1f}s)"
                    for o in abandoned_objs
                ]
                alert_text = "ABANDONMENT ALERT: " + " | ".join(alert_parts)
                cv2.putText(
                    vis,
                    alert_text,
                    (16, height - int(banner_h * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 1.05,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 6. Top HUD
            if show_hud:
                hud_h = max(38, int(height * 0.048))
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (width, hud_h), (20, 20, 25), -1)
                cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
                hud_txt = (
                    f"Frame {frame_id}/{total_frames} | {_fmt_time(timestamp)} ({timestamp:.1f}s) | "
                    f"Persons: {len(active_tracks)} | Objects: {len(active_objs)} | "
                    f"Alerts: {len(abandoned_objs)}"
                )
                cv2.putText(
                    vis,
                    hud_txt,
                    (16, int(hud_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.95,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(vis)
            if display:
                cv2.imshow("Annotated Track Visualizer", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frame_id >= max_frames:
                break
            if frame_id % 100 == 0:
                print(f"Rendered {frame_id}/{total_frames} frames ({timestamp:.1f}s)...")

        # 7. Append 1.5s End-of-Clip Summary Slate (captures late/unattended_at_end alerts too)
        if summary_slate and last_frame is not None:
            slate = (last_frame.astype(np.float32) * 0.28).astype(np.uint8)
            y_cur = max(70, int(height * 0.12))
            line_step = max(36, int(height * 0.048))
            cv2.putText(
                slate,
                f"END-OF-CLIP SUMMARY ({video_file.name})",
                (40, y_cur),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.35,
                (120, 220, 255),
                2,
                cv2.LINE_AA,
            )
            y_cur += line_step + 10
            cv2.putText(
                slate,
                f"Total Global Persons: {len(reid_manager.gallery)}",
                (40, y_cur),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            y_cur += line_step

            if object_tracker is not None:
                final_objs = object_tracker.get_all_indexed_objects()
                for obj in final_objs:
                    st = obj["status"]
                    owner_str = _resolve_owner(obj["owner_track_id"], reid_manager)
                    color = (
                        (90, 90, 255)
                        if st == "abandoned"
                        else ((80, 200, 255) if st == "unattended_at_end" else (140, 235, 160))
                    )
                    line = (
                        f"[OBJ:{obj['object_id']}] {obj['label']} | Status: {st.upper()} | "
                        f"Owner: {owner_str} | Visible: {obj['start_sec']:.1f}s-{obj['end_sec']:.1f}s"
                    )
                    if obj.get("abandoned_since_sec", -1.0) >= 0:
                        line += f" | Unattended since {obj['abandoned_since_sec']:.1f}s"
                    cv2.putText(
                        slate,
                        line,
                        (40, min(height - 30, y_cur)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    y_cur += line_step

            for _ in range(max(1, int(round(fps * 1.5)))):
                writer.write(slate)

    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\nAnnotated video saved to: {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Track ID, Bounding Box & Abandonment Video Visualizer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="data/annotated_output.mp4", help="Output annotated MP4 path")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--track-objects", type=str, default=None, help="Comma-separated YOLO-World prompts")
    parser.add_argument("--abandon-seconds", type=float, default=5.0, help="Abandonment threshold in seconds")
    parser.add_argument("--reid-threshold", type=float, default=0.72, help="OSNet Re-ID cosine similarity threshold")
    parser.add_argument("--face-threshold", type=float, default=0.55, help="ArcFace cosine similarity threshold")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384", help="SigLIP model ID")
    parser.add_argument("--db-path", type=str, default="data/lancedb_viz", help="Scratch LanceDB path for visualizer")

    # Detection / Re-ID opt-in flags
    parser.add_argument("--tile-objects", action="store_true", help="Use 2x2 tiled YOLO-World detection")
    parser.add_argument("--multi-angle-reid", action="store_true", help="Enable multi-angle OSNet gallery")
    parser.add_argument("--motion-owner-weight", action="store_true", help="Enable 2.5x moving-object owner weight")

    # Visualization toggle flags
    parser.add_argument("--no-persons", action="store_true", help="Hide person bounding boxes and Global IDs")
    parser.add_argument("--no-objects", action="store_true", help="Hide object bounding boxes")
    parser.add_argument("--no-abandonment", action="store_true", help="Hide abandonment alert highlights and banners")
    parser.add_argument("--no-hud", action="store_true", help="Hide top frame/timestamp HUD bar")
    parser.add_argument("--only-abandoned", action="store_true", help="Draw only abandoned objects and their owners")
    parser.add_argument("--no-summary-slate", action="store_true", help="Do not append 1.5s end-of-clip summary slate")
    parser.add_argument("--display", action="store_true", help="Also show live preview window while rendering")
    args = parser.parse_args()

    render_annotated_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        reid_threshold=args.reid_threshold,
        face_threshold=args.face_threshold,
        siglip_model=args.siglip_model,
        track_objects=args.track_objects,
        abandon_seconds=args.abandon_seconds,
        tile_objects=args.tile_objects,
        multi_angle_reid=args.multi_angle_reid,
        motion_owner_weight=args.motion_owner_weight,
        show_persons=not args.no_persons,
        show_objects=not args.no_objects,
        show_abandonment=not args.no_abandonment,
        show_hud=not args.no_hud,
        only_abandoned=args.only_abandoned,
        summary_slate=not args.no_summary_slate,
        display=args.display,
        db_path=args.db_path,
    )
