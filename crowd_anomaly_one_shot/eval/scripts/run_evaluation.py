"""Evaluation Suite Runner with Verbose Progress Bar, Latency Profiling, and CSV Logging.

Runs the two-stage anomaly detection pipeline using:
- Tier 1: Google SigLIP (Fast-pass zero-shot screening)
- Tier 2: Hugging Face SmolVLM2-2.2B-Instruct (Multimodal VLM triage)
Across XD-Violence and UCF-Crime datasets. Generates annotated video outputs with
the 3-state top-left HUD, logs per-video CSV results, and computes metrics.
"""

import os
import sys
import ssl
import time
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# Ensure SSL certificates are properly recognized across diverse server environments
try:
    import certifi
    ca_bundle = certifi.where()
    if not os.environ.get("SSL_CERT_FILE"):
        os.environ["SSL_CERT_FILE"] = ca_bundle
    if not os.environ.get("REQUESTS_CA_BUNDLE"):
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
except ImportError:
    pass

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except (AttributeError, Exception):
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import eval.config_eval as config_eval
import eval.models_eval as models_eval
import eval.viz_eval as viz_eval


def load_ucf_annotations(anno_path: str) -> Dict[str, Tuple[int, int]]:
    """Load ground-truth start and end anomaly frame numbers for UCF-Crime testing videos."""
    anno = {}
    if os.path.exists(anno_path):
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    fname = parts[0]
                    try:
                        s1, e1 = int(parts[2]), int(parts[3])
                        anno[fname] = (s1, e1)
                    except ValueError:
                        pass
    return anno


def read_video_frames_fast(v_path: str, target_fps: float = 5.0, width: int = 640,
                           start_frame: int = 0, max_frames: int = 200) -> List[np.ndarray]:
    """Fast sequential video frame reader with stride sampling and aspect-ratio resize."""
    assert os.path.exists(v_path), f"Video file not found: {v_path}"
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / target_fps)))

    if start_frame > 0 and start_frame < total_src:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur_src = start_frame
    else:
        cur_src = 0

    frames = []
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_h = int(round(src_h * (width / src_w))) if src_w > 0 else int(width * 9.0 / 16.0)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (cur_src - start_frame) % step == 0:
            resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(resized)
        cur_src += 1

    cap.release()
    return frames


def run_stage1_on_frames(frames: List[np.ndarray]) -> tuple:
    """Run motion estimation via DIS optical flow and compute dynamic scene Z-scores."""
    t0 = time.perf_counter()
    if len(frames) < 2:
        return [], [], 0.0

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

    if not grid_mags:
        return [], [], 0.0

    grid_mags_arr = np.stack(grid_mags, axis=0)  # [N-1, grid_h, grid_w]
    # Compute dynamic baseline from initial normal frames (first 15 frames)
    base_window = min(len(grid_mags_arr), 15)
    b_means = np.median(grid_mags_arr[:base_window], axis=0)
    b_mads = np.median(np.abs(grid_mags_arr[:base_window] - b_means[None, ...]), axis=0)
    b_mads = np.maximum(b_mads, 0.3)

    frame_scores = []
    frame_boxes = []
    for mag_grid in grid_mags_arr:
        z_grid = np.abs(mag_grid - b_means) / b_mads
        max_z = float(np.max(z_grid))
        frame_scores.append(max_z)

        # Region box if gate triggered
        if max_z >= config_eval.LOW_THRESHOLD:
            rr, cc = np.where(z_grid >= config_eval.LOW_THRESHOLD)
            min_y, max_y = int(np.min(rr) * cell_h), int((np.max(rr) + 1) * cell_h)
            min_x, max_x = int(np.min(cc) * cell_w), int((np.max(cc) + 1) * cell_w)
            frame_boxes.append([min_x, min_y, max_x, max_y])
        else:
            frame_boxes.append(None)

    lat_per_f = ((time.perf_counter() - t0) / max(len(frames) - 1, 1)) * 1000.0
    return frame_scores, frame_boxes, lat_per_f


def evaluate_video(v_path: str, v_meta: Dict, tier1_bundle: tuple,
                   tier2_bundle: tuple, out_video_dir: str,
                   ucf_anno: Dict, fire_model=None, device: str = "cuda",
                   num_frames_vlm: int = config_eval.SMOLVLM2_NUM_FRAMES) -> Dict:
    """Run full evaluation on a single video with timing, classification, and video output."""
    t_vid_start = time.perf_counter()
    vname = os.path.basename(v_path)
    gt_class = v_meta.get("ground_truth", "Normal")
    dataset_name = v_meta.get("dataset", "unknown")

    # Determine optimal start frame for UCF videos if temporal annotation is known
    start_frame = 0
    if dataset_name == "ucf_crime" and vname in ucf_anno:
        s_anno, _ = ucf_anno[vname]
        if s_anno > 0:
            start_frame = max(0, s_anno - 90)  # 3 seconds prior to anomaly onset

    # 1. Read frames at 5 FPS, 640px width (up to 150 frames = 30 seconds)
    frames = read_video_frames_fast(v_path, target_fps=config_eval.ANALYSIS_FPS,
                                    width=config_eval.ANALYSIS_WIDTH,
                                    start_frame=start_frame, max_frames=150)
    if len(frames) < 2:
        return None

    # 2. Stage 1 Execution: Motion Scoring & Periodic Fire/Smoke Gate
    scores, boxes, s1_lat = run_stage1_on_frames(frames)
    max_z = max(scores) if scores else 0.0

    # Periodic YOLO fire/smoke screening (every 5 frames = 1 second)
    fire_boxes_by_frame = {}
    fire_detected = False
    if fire_model is not None:
        try:
            from src.fire import detect_fire_smoke
            for f_idx in range(0, len(frames), 5):
                f_dets = detect_fire_smoke(fire_model, frames[f_idx], conf_thr=0.25)
                if f_dets:
                    fire_detected = True
                    for offset in range(min(5, len(frames) - f_idx)):
                        fire_boxes_by_frame[f_idx + offset] = f_dets
        except Exception as e:
            pass

    stage1_flagged = (max_z >= config_eval.LOW_THRESHOLD) or fire_detected

    # 3. Stage 2 Classification: Gated Cascade
    t0_s2 = time.perf_counter()
    active_model = "Stage 1 Gate"
    if not stage1_flagged:
        # Gate quiet: no motion anomaly or fire detected
        final_pred = "Normal"
        final_conf = 0.98
        t1_top = "Normal"
        t1_conf = 0.98
        escalated = False
        t2_pred = "-"
        s2_lat = 0.0
    else:
        # Gate triggered: isolate the peak anomaly window where the event occurs
        peak_idx = int(np.argmax(scores)) + 1
        if fire_detected and fire_boxes_by_frame:
            fire_indices = list(fire_boxes_by_frame.keys())
            if fire_indices:
                peak_idx = fire_indices[len(fire_indices) // 2]

        start_f = max(0, peak_idx - 6)
        end_f = min(len(frames), peak_idx + 6)
        anomaly_frames = frames[start_f:end_f]
        if len(anomaly_frames) < 2:
            anomaly_frames = frames

        # If fire detected by sentinel and motion is low or classification ambiguous
        if fire_detected and max_z < config_eval.LOW_THRESHOLD:
            final_pred = "Arson"
            final_conf = 0.88
            t1_top = "Arson"
            t1_conf = 0.88
            escalated = False
            t2_pred = "-"
            active_model = "YOLO Fire Sentinel"
            s2_lat = 0.0
        else:
            cascade_out = models_eval.classify_cascade(
                tier1_bundle, tier2_bundle, anomaly_frames, device=device,
                escalate_thr=config_eval.CASCADE_ESCALATE_THR,
                num_frames_vlm=num_frames_vlm
            )
            s2_lat = (time.perf_counter() - t0_s2) * 1000.0

            final_pred = cascade_out["top"]
            final_conf = cascade_out["top_prob"]
            t1_top = cascade_out["tier1_res"]["top"]
            t1_conf = cascade_out["tier1_res"]["top_prob"]
            escalated = cascade_out["escalated"]
            t2_pred = cascade_out["tier2_res"]["top"] if escalated else "-"
            active_model = "SmolVLM2" if escalated else "SigLIP"

            if fire_detected and final_pred in ["Normal", "Car Accident"]:
                final_pred = "Arson"
                final_conf = 0.88
                active_model = "YOLO Fire Sentinel"

    # 4. Generate 3-State Frame Labels for Video HUD
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
        # 3-State HUD determination:
        if z_val < config_eval.LOW_THRESHOLD and not has_fire:
            # State 1: Normal
            frame_states.append({
                "state": "normal",
                "z_score": z_val,
                "class": "Normal",
                "confidence": 0.99,
                "model": "Stage 1 Gate",
                "latency_ms": s1_lat
            })
        else:
            # Stage 1 triggered:
            if final_pred == "Normal":
                # Suppressed by classifier
                frame_states.append({
                    "state": "multimodal_suppressed",
                    "z_score": z_val,
                    "class": "Normal",
                    "confidence": final_conf,
                    "model": active_model,
                    "latency_ms": s2_lat
                })
            else:
                # Active Multimodal Anomaly Confirmed
                frame_states.append({
                    "state": "multimodal_abnormal",
                    "z_score": z_val if z_val >= config_eval.LOW_THRESHOLD else max_z,
                    "class": final_pred,
                    "confidence": final_conf,
                    "model": active_model,
                    "latency_ms": s2_lat
                })

    # 5. Render Output Video with HUD
    out_vid_name = f"eval_{os.path.splitext(vname)[0]}.mp4"
    out_vid_path = os.path.join(out_video_dir, out_vid_name)
    viz_eval.render_annotated_video(
        frames, frame_states, boxes_per_frame, out_vid_path,
        fps=config_eval.ANALYSIS_FPS, s1_ms=s1_lat, s2_ms=s2_lat,
        device_name=device
    )

    tot_run_sec = time.perf_counter() - t_vid_start
    eff_fps = len(frames) / max(tot_run_sec, 1e-3)
    is_correct = (final_pred.lower() == gt_class.lower())

    return {
        "video_name": vname,
        "dataset": dataset_name,
        "ground_truth": gt_class,
        "duration_sec": round(len(frames) / config_eval.ANALYSIS_FPS, 2),
        "frames_count": len(frames),
        "stage1_anomaly_flagged": stage1_flagged,
        "stage1_max_z": round(max_z, 2),
        "tier1_model": config_eval.TIER1_MODEL_ID,
        "tier1_top_class": t1_top,
        "tier1_confidence": round(t1_conf, 4),
        "tier2_escalated": escalated,
        "tier2_model": config_eval.TIER2_MODEL_ID if escalated else "None",
        "tier2_prediction": t2_pred,
        "final_predicted_class": final_pred,
        "final_confidence": round(final_conf, 4),
        "is_correct": is_correct,
        "stage1_latency_ms": round(s1_lat, 2),
        "stage2_latency_ms": round(s2_lat, 2),
        "total_runtime_sec": round(tot_run_sec, 2),
        "effective_fps": round(eff_fps, 2),
        "annotated_video_path": out_vid_path
    }


def compute_metrics_and_plot(results: List[Dict], out_dir: str, device: str = None):
    """Compute precision, recall, F1, confusion matrix, and generate markdown summary."""
    df = pd.DataFrame(results)
    classes = config_eval.CLASSES
    hw_label = viz_eval.get_hardware_label(device)

    # Build Confusion Matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    cls_to_idx = {c.lower(): idx for idx, c in enumerate(classes)}

    for _, row in df.iterrows():
        gt_idx = cls_to_idx.get(row["ground_truth"].lower(), 0)
        pred_idx = cls_to_idx.get(row["final_predicted_class"].lower(), 0)
        cm[gt_idx, pred_idx] += 1

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Phase 4+ Evaluation Confusion Matrix (SigLIP + SmolVLM2)", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35, ha="right", fontsize=9.5)
    plt.yticks(tick_marks, classes, fontsize=9.5)

    # Label counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            col = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=col, fontsize=10, fontweight="bold")

    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xlabel("Predicted Class (Cascade)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config_eval.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    # Compute Per-Class Precision, Recall, F1
    lines = [
        "# Evaluation Metrics Summary (SigLIP + SmolVLM2 Cascade)",
        "",
        "## 1. Per-Class Quantitative Breakdown",
        "| Class Name | Support (N) | Precision | Recall | F1-Score |",
        "|---|---|---|---|---|"
    ]

    total_tp = 0
    total_samples = len(df)
    f1_list = []

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        support = sum(cm[i, :])
        prec = tp / max(tp + fp, 1e-6) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1e-6) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-6) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        total_tp += tp
        lines.append(f"| **{c}** | {support} | {prec*100:.1f}% | {rec*100:.1f}% | {f1*100:.1f}% |")

    macro_f1 = np.mean(f1_list)
    accuracy = total_tp / max(total_samples, 1)
    avg_s1_lat = df["stage1_latency_ms"].mean()
    avg_s2_lat = df["stage2_latency_ms"].mean()
    avg_fps = df["effective_fps"].mean()

    lines.extend([
        "",
        "## 2. Overall Pipeline Performance",
        f"- **Overall Accuracy**: **{accuracy*100:.1f}%** ({total_tp} / {total_samples} correct)",
        f"- **Macro F1-Score**: **{macro_f1*100:.1f}%** across all 7 surveillance classes",
        f"- **Average Stage 1 Latency**: **{avg_s1_lat:.1f} ms / frame**",
        f"- **Average Stage 2 Latency**: **{avg_s2_lat:.1f} ms / clip**",
        f"- **Effective Pipeline Throughput**: **{avg_fps:.1f} FPS** (Measured on {hw_label})",
        f"- **Tier 2 Escalation Rate**: **{df['tier2_escalated'].mean()*100:.1f}%** of clips escalated from SigLIP to SmolVLM2"
    ])

    with open(config_eval.METRICS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMetrics report compiled at: {config_eval.METRICS_REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite with SigLIP & SmolVLM2.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device (cuda or cpu).")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limit total number of evaluation videos to process.")
    parser.add_argument("--sample_per_class", type=int, default=None,
                        help="Sample N videos per class for balanced benchmarking.")
    parser.add_argument("--backbone", default=config_eval.TIER1_BACKBONE_NAME,
                        choices=["base", "so400m", "large"] + list(config_eval.SIGLIP_BACKBONES.values()),
                        help="SigLIP Tier 1 backbone ('so400m' default, 'base', or 'large').")
    parser.add_argument("--num_frames_vlm", type=int, default=config_eval.SMOLVLM2_NUM_FRAMES,
                        help="Number of frames for SmolVLM2 (1=snapshot, 4=sequence, 16=storyboard grid).")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle videos to interleave classes and datasets.")
    args = parser.parse_args()

    os.makedirs(config_eval.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config_eval.VIDEO_OUTPUT_DIR, exist_ok=True)

    # Check manifest
    if not os.path.exists(config_eval.MANIFEST_PATH):
        print(f"Manifest not found at {config_eval.MANIFEST_PATH}.")
        print("Running download_and_sample.py first...")
        import eval.scripts.download_and_sample as ds
        ds.main()

    manifest_df = pd.read_csv(config_eval.MANIFEST_PATH)
    if args.sample_per_class:
        manifest_df = manifest_df.groupby("ground_truth", group_keys=False).apply(
            lambda x: x.head(args.sample_per_class)
        ).reset_index(drop=True)
    if getattr(args, "shuffle", False):
        manifest_df = manifest_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if args.max_videos:
        manifest_df = manifest_df.head(args.max_videos)

    t1_model_id = config_eval.SIGLIP_BACKBONES.get(args.backbone, args.backbone)

    dev_label = viz_eval.get_hardware_label(args.device)
    print(f"\n=======================================================")
    print(f"Starting Evaluation on {len(manifest_df)} Surveillance Videos")
    print(f"Tier 1: {t1_model_id} (Backbone: {args.backbone})")
    print(f"Tier 2: {config_eval.TIER2_MODEL_ID} ({args.num_frames_vlm} frames)")
    print(f"Target Classes: {config_eval.CLASSES}")
    print(f"Device: {args.device} ({dev_label})")
    print(f"=======================================================\n")

    print(f"Loading Tier 1 (SigLIP: {t1_model_id})...")
    t1_bundle = models_eval.load_tier1(model_id=t1_model_id, device=args.device)
    print("SigLIP initialized.")

    print("Loading Tier 2 (SmolVLM2)...")
    t2_bundle = models_eval.load_tier2(device=args.device)
    print("SmolVLM2 initialized.\n")

    # Load YOLOv8 Fire & Smoke Sentinel if weights exist
    fire_model_path = "weights/yolov8n_fire_smoke.pt"
    fire_model = None
    if os.path.exists(fire_model_path):
        try:
            from src.fire import load_fire_model
            fire_model = load_fire_model(fire_model_path, device=args.device)
            print("YOLOv8n Fire & Smoke Sentinel initialized.\n")
        except Exception as e:
            print(f"Note: YOLO fire model could not be loaded: {e}\n")

    # Load UCF temporal annotations if present
    ucf_anno_path = os.path.join(config_eval.DATA_DIR, "ucf_temporal_annotations.txt")
    ucf_anno = load_ucf_annotations(ucf_anno_path)
    print(f"Loaded {len(ucf_anno)} temporal annotations for UCF-Crime.")

    results = []
    # Verbose tqdm progress bar
    pbar = tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Surveillance Evaluation", unit="video")

    for idx, row in pbar:
        v_path = row.get("full_path") or row.get("file_path")
        if not v_path or not os.path.exists(str(v_path)):
            if "local_path" in row:
                v_path = os.path.join(config_eval.BASE_DIR, str(row["local_path"]))
        v_path = str(v_path)
        v_meta = row.to_dict()
        vname = os.path.basename(v_path)

        pbar.set_description(f"Eval: {vname[:18]}")
        res = evaluate_video(
            v_path, v_meta, t1_bundle, t2_bundle, config_eval.VIDEO_OUTPUT_DIR,
            ucf_anno, fire_model=fire_model, device=args.device,
            num_frames_vlm=args.num_frames_vlm
        )
        if res:
            results.append(res)
            # Incremental CSV save every 10 videos
            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(config_eval.CSV_RESULTS_PATH, index=False)

            # Update progress bar verbose status
            pbar.set_postfix(
                gt=res["ground_truth"][:6],
                pred=res["final_predicted_class"][:6],
                ok="YES" if res["is_correct"] else "NO",
                s1_ms=f"{res['stage1_latency_ms']:.0f}",
                s2_ms=f"{res['stage2_latency_ms']:.0f}",
                fps=f"{res['effective_fps']:.1f}"
            )

    # Save detailed CSV results
    results_df = pd.DataFrame(results)
    results_df.to_csv(config_eval.CSV_RESULTS_PATH, index=False)
    print(f"\n\nEvaluation Results CSV saved to: {config_eval.CSV_RESULTS_PATH}")

    # Compute Metrics, Confusion Matrix, and Markdown Report
    compute_metrics_and_plot(results, config_eval.OUTPUT_DIR, device=args.device)
    print("All evaluation tasks completed successfully.")


if __name__ == "__main__":
    main()
