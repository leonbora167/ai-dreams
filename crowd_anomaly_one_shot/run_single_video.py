#!/usr/bin/env python
"""Turnkey Single-Video Inference and Visualization Tool.

Runs end-to-end video anomaly detection on any input video file:
1. Fast Stage 1 optical flow motion estimation & dynamic baseline scoring.
2. Anomaly gate screening and 16-frame event window isolation.
3. Two-tier multimodal cascade:
   - Tier 1: Google SigLIP (Fast-pass zero-shot vision-text matching)
   - Tier 2: Hugging Face SmolVLM2-2.2B-Instruct (Multimodal VLM triage)
4. Video rendering with top-left 3-state management HUD & top-right edge telemetry.

Usage:
    python run_single_video.py --video "path/to/video.mp4" --output "output.mp4"
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

# Import configuration and models
import eval.config_eval as config_eval
import eval.models_eval as models_eval
import eval.viz_eval as viz_eval


def read_video_frames_sequential(v_path: str, target_fps: float = 5.0, width: int = 640,
                                 start_frame: int = 0, max_frames: int = 400) -> Tuple[List[np.ndarray], float, float]:
    """Read frames sequentially with stride subsampling and aspect-ratio resize."""
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Input video not found: {v_path}")

    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {v_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_h = int(round(src_h * (width / src_w))) if src_w > 0 else int(width * 9.0 / 16.0)

    step = max(1, int(round(src_fps / target_fps)))
    frames = []

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_src = start_frame

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if cur_src % step == 0:
            resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(resized)
        cur_src += 1

    cap.release()
    duration_sec = len(frames) / target_fps
    return frames, src_fps, duration_sec


def compute_stage1_motion(frames: List[np.ndarray]) -> Tuple[List[float], List[object], float]:
    """Compute optical flow motion grid and dynamic baseline Z-scores."""
    t0 = time.perf_counter()
    if len(frames) < 2:
        return [0.0], [None], 0.0

    h, w = frames[0].shape[:2]
    grid_h, grid_w = 12, 20
    cell_h, cell_w = max(1, h // grid_h), max(1, w // grid_w)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    grid_mags = []

    prev_gray = cv2.cvtColor(cv2.resize(frames[0], (160, 120)), cv2.COLOR_BGR2GRAY)
    for idx in range(1, len(frames)):
        curr_gray = cv2.cvtColor(cv2.resize(frames[idx], (160, 120)), cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, curr_gray, None)
        prev_gray = curr_gray

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        grid_mag = np.zeros((grid_h, grid_w), dtype=np.float32)
        gh_scale, gw_scale = 120 // grid_h, 160 // grid_w
        for r in range(grid_h):
            for c in range(grid_w):
                grid_mag[r, c] = np.mean(mag[r*gh_scale:(r+1)*gh_scale, c*gw_scale:(c+1)*gw_scale])
        grid_mags.append(grid_mag)

    grid_mags_arr = np.stack(grid_mags, axis=0)
    base_window = min(len(grid_mags_arr), 15)
    b_means = np.median(grid_mags_arr[:base_window], axis=0)
    b_mads = np.median(np.abs(grid_mags_arr[:base_window] - b_means[None, ...]), axis=0)
    b_mads = np.maximum(b_mads, 0.3)

    scores = []
    boxes = []
    for mag_grid in grid_mags_arr:
        z_grid = np.abs(mag_grid - b_means) / b_mads
        max_z = float(np.max(z_grid))
        scores.append(max_z)

        if max_z >= config_eval.LOW_THRESHOLD:
            rr, cc = np.where(z_grid >= config_eval.LOW_THRESHOLD)
            min_y, max_y = int(np.min(rr) * cell_h), int((np.max(rr) + 1) * cell_h)
            min_x, max_x = int(np.min(cc) * cell_w), int((np.max(cc) + 1) * cell_w)
            boxes.append([min_x, min_y, max_x, max_y])
        else:
            boxes.append(None)

    lat_per_f = ((time.perf_counter() - t0) / max(len(frames) - 1, 1)) * 1000.0
    return scores, boxes, lat_per_f


def run_pipeline(video_path: str, output_path: str = None, device: str = "cuda",
                 escalate_thr: float = 0.65, start_frame: int = 0, max_frames: int = 350,
                 backbone: str = config_eval.TIER1_BACKBONE_NAME,
                 num_frames_vlm: int = config_eval.SMOLVLM2_NUM_FRAMES) -> Dict:
    """Run full inference on a single video file and generate annotated video."""
    t_start = time.perf_counter()
    vname = os.path.basename(video_path)
    if output_path is None:
        base_name = os.path.splitext(vname)[0]
        output_path = f"annotated_{base_name}.mp4"

    t1_model_id = config_eval.SIGLIP_BACKBONES.get(backbone, backbone)

    print("\n=======================================================")
    print(f"Surveillance Video Anomaly Inference Pipeline")
    print(f"Input Video : {video_path}")
    print(f"Device      : {device}")
    print(f"Tier 1 Model: SigLIP ({backbone} -> {t1_model_id})")
    print(f"Tier 2 Model: SmolVLM2 ({num_frames_vlm} frames ingestion)")
    print(f"=======================================================\n")

    # 1. Read Video Frames
    print("Reading video frames...")
    frames, src_fps, duration_sec = read_video_frames_sequential(
        video_path, target_fps=config_eval.ANALYSIS_FPS,
        width=config_eval.ANALYSIS_WIDTH, start_frame=start_frame, max_frames=max_frames
    )
    print(f"Loaded {len(frames)} frames ({duration_sec:.1f}s duration @ 5.0 FPS)")

    # 2. Stage 1 Motion & Fire/Smoke Screening
    print("\n[Stage 1] Screening motion & energy (DIS Optical Flow) + Periodic Fire/Smoke Sentinel...")
    scores, boxes, s1_lat = compute_stage1_motion(frames)
    max_z = max(scores) if scores else 0.0

    # Periodic YOLO fire/smoke screening (every 5 frames = 1 second)
    fire_boxes_by_frame = {}
    fire_detected = False
    fire_model_path = "weights/yolov8n_fire_smoke.pt"
    if os.path.exists(fire_model_path):
        try:
            from src.fire import load_fire_model, detect_fire_smoke
            fire_model = load_fire_model(fire_model_path, device=device)
            for f_idx in range(0, len(frames), 5):
                f_dets = detect_fire_smoke(fire_model, frames[f_idx], conf_thr=0.25)
                if f_dets:
                    fire_detected = True
                    for offset in range(min(5, len(frames) - f_idx)):
                        fire_boxes_by_frame[f_idx + offset] = f_dets
            if fire_detected:
                print(f"  [Fire Sentinel] Active smoke/fire detected across video!")
        except Exception as e:
            print(f"  Note: Fire sentinel skipped: {e}")

    stage1_flagged = (max_z >= config_eval.LOW_THRESHOLD) or fire_detected
    print(f"  Stage 1 Latency: {s1_lat:.2f} ms / frame")
    print(f"  Max Deviation  : Z = {max_z:.2f} (Threshold = {config_eval.LOW_THRESHOLD})")
    print(f"  Gate Status    : {'OPENED (COMMOTION/FIRE DETECTED)' if stage1_flagged else 'CLOSED (CALM SCENE)'}")

    # 3. Stage 2 Multimodal AI Cascade
    t0_s2 = time.perf_counter()
    t1_top = "Normal"
    t1_conf = 0.98
    escalated = False
    t2_pred = "-"
    t2_raw = "-"

    if not stage1_flagged:
        final_pred = "Normal"
        final_conf = 0.98
        active_model = "Stage 1 Gate"
        s2_lat = 0.0
        print("\n[Stage 2] Bypassed (Scene verified as Normal). 0 ms AI cost.")
    else:
        # Isolate 16-frame event crop around peak commotion or fire detection
        peak_idx = int(np.argmax(scores)) + 1
        if fire_detected and fire_boxes_by_frame:
            fire_indices = list(fire_boxes_by_frame.keys())
            if fire_indices:
                peak_idx = fire_indices[len(fire_indices)//2]

        start_f = max(0, peak_idx - 8)
        end_f = min(len(frames), peak_idx + 8)
        event_frames = frames[start_f:end_f]
        if len(event_frames) < 2:
            event_frames = frames
        print(f"\n[Stage 2] Commotion localized at frame {peak_idx} ({peak_idx/config_eval.ANALYSIS_FPS:.1f}s)")
        print(f"  Extracted 16-frame event crop [frames {start_f}..{end_f}]")

        print(f"  Loading Tier 1 (Google SigLIP: {t1_model_id})...")
        t1_bundle = models_eval.load_tier1(model_id=t1_model_id, device=device)
        t1_res = models_eval.classify_tier1(t1_bundle[0], t1_bundle[1], t1_bundle[2], event_frames, device=device)
        t1_top = t1_res["top"]
        t1_conf = t1_res["top_prob"]
        print(f"  Tier 1 Result  : {t1_top} ({t1_conf*100:.1f}%) in {t1_res['latency_ms']:.1f} ms")

        # If fire was detected by YOLO sentinel, prioritize Arson
        if fire_detected and t1_top in ["Normal", "Car Accident"]:
            final_pred = "Arson"
            final_conf = 0.88
            active_model = "YOLO Fire Sentinel"
            s2_lat = (time.perf_counter() - t0_s2) * 1000.0
            print("  YOLO Fire Sentinel confirmed active fire/smoke -> Classified as ARSON.")
        else:
            # Check escalation requirement
            needs_tier2 = (t1_conf < escalate_thr) or (t1_top in ["Violence", "Fighting"] and t1_conf < 0.75)
            if not needs_tier2:
                final_pred = t1_top
                final_conf = t1_conf
                active_model = "SigLIP"
                s2_lat = (time.perf_counter() - t0_s2) * 1000.0
                print("  Confidence high: Tier 1 decision accepted directly.")
            else:
                escalated = True
                print(f"  Confidence ambiguous -> Escalating to Tier 2 (SmolVLM2 @ {num_frames_vlm} frames)...")
                t2_bundle = models_eval.load_tier2(device=device)
                t2_res = models_eval.classify_tier2(t2_bundle[0], t2_bundle[1], event_frames, device=device, num_frames=num_frames_vlm)
                t2_pred = t2_res["top"]
                t2_raw = t2_res["raw_response"]
                s2_lat = (time.perf_counter() - t0_s2) * 1000.0

                final_pred = t2_pred
                final_conf = max(t1_conf, 0.88) if final_pred != "Normal" else 0.90
                active_model = "SmolVLM2"
                print(f"  Tier 2 Result  : {t2_pred} (Raw: '{t2_raw[:25]}') in {t2_res['latency_ms']:.1f} ms")

        # Guaranteed override: If active smoke or fire was spotted by the Sentinel, never suppress to Normal
        if fire_detected and final_pred in ["Normal", "Car Accident"]:
            final_pred = "Arson"
            final_conf = 0.90
            active_model = "YOLO Fire Sentinel"
            print("  [Fire Sentinel] Active fire/smoke confirmed -> Overriding final classification to ARSON.")

    # 4. Generate 3-State HUD Annotations
    print(f"\n[HUD Engine] Rendering management visual annotations...")
    frame_states = []
    boxes_per_frame = []

    for i in range(len(frames)):
        z_val = scores[i-1] if i > 0 and (i-1) < len(scores) else 0.0
        cur_box = boxes[i-1] if i > 0 and (i-1) < len(boxes) else None

        box_list = []
        if cur_box:
            box_col = (0, 0, 255) if final_pred != "Normal" else (0, 165, 255)
            box_list.append({
                "box": cur_box,
                "color": box_col,
                "label": f"Z={z_val:.1f}" if final_pred == "Normal" else f"{final_pred} ({final_conf*100:.0f}%)"
            })

        # Add dedicated fire/smoke detections if present
        if i in fire_boxes_by_frame:
            for fd in fire_boxes_by_frame[i]:
                fb = [int(v) for v in fd["box"]]
                f_label = f"{fd['cls'].upper()} ({fd['conf']*100:.0f}%)"
                box_list.append({
                    "box": fb,
                    "color": (0, 69, 255),  # Bright Orange-Red
                    "label": f_label
                })
        boxes_per_frame.append(box_list)

        has_fire = (i in fire_boxes_by_frame)
        if z_val < config_eval.LOW_THRESHOLD and not has_fire:
            frame_states.append({
                "state": "normal",
                "z_score": z_val,
                "class": "Normal",
                "confidence": 0.99,
                "model": "Stage 1 Gate",
                "latency_ms": s1_lat
            })
        else:
            if final_pred == "Normal":
                frame_states.append({
                    "state": "multimodal_suppressed",
                    "z_score": z_val,
                    "class": "Normal",
                    "confidence": final_conf,
                    "model": active_model,
                    "latency_ms": s2_lat
                })
            else:
                frame_states.append({
                    "state": "multimodal_abnormal",
                    "z_score": z_val if z_val >= config_eval.LOW_THRESHOLD else max_z,
                    "class": final_pred,
                    "confidence": final_conf,
                    "model": active_model,
                    "latency_ms": s2_lat
                })

    # 5. Render Output Video
    viz_eval.render_annotated_video(
        frames, frame_states, boxes_per_frame, output_path,
        fps=config_eval.ANALYSIS_FPS, s1_ms=s1_lat, s2_ms=s2_lat,
        device_name=device
    )

    tot_run_sec = time.perf_counter() - t_start
    eff_fps = len(frames) / max(tot_run_sec, 1e-3)

    print("\n=======================================================")
    print("INFERENCE SUMMARY:")
    print(f"  Final Verdict   : {final_pred.upper()} ({final_conf*100:.1f}%)")
    print(f"  Gate Triggered  : {stage1_flagged} (Max Z = {max_z:.2f})")
    print(f"  Deciding Model  : {active_model}")
    print(f"  Total Runtime   : {tot_run_sec:.2f} seconds ({eff_fps:.1f} FPS)")
    print(f"  Annotated Video : {os.path.abspath(output_path)}")
    print("=======================================================\n")

    return {
        "video": video_path,
        "final_predicted_class": final_pred,
        "final_confidence": round(final_conf, 4),
        "stage1_flagged": stage1_flagged,
        "max_z": round(max_z, 2),
        "tier1_top": t1_top,
        "tier1_conf": round(t1_conf, 4),
        "tier2_escalated": escalated,
        "tier2_pred": t2_pred,
        "stage1_latency_ms": round(s1_lat, 2),
        "stage2_latency_ms": round(s2_lat, 2),
        "total_runtime_sec": round(tot_run_sec, 2),
        "effective_fps": round(eff_fps, 2),
        "annotated_video_path": os.path.abspath(output_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Anomaly Detection on a Single Video.")
    parser.add_argument("--video", "-v", required=True, help="Path to input surveillance video file.")
    parser.add_argument("--output", "-o", default=None, help="Path to save annotated MP4 video output.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device (cuda or cpu).")
    parser.add_argument("--backbone", default=config_eval.TIER1_BACKBONE_NAME,
                        choices=["base", "so400m", "large"] + list(config_eval.SIGLIP_BACKBONES.values()),
                        help="SigLIP Tier 1 backbone ('so400m' default, 'base', or 'large').")
    parser.add_argument("--num_frames_vlm", type=int, default=config_eval.SMOLVLM2_NUM_FRAMES,
                        help="Number of frames for SmolVLM2 (1=snapshot, 4=sequence, 16=storyboard grid).")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting source frame index (default: 0).")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to process.")
    parser.add_argument("--escalate_thr", type=float, default=0.65, help="Tier 1 -> Tier 2 escalation threshold.")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        device=args.device,
        escalate_thr=args.escalate_thr,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        backbone=args.backbone,
        num_frames_vlm=args.num_frames_vlm
    )


if __name__ == "__main__":
    main()

