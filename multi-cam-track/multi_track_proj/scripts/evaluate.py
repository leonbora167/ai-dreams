"""Evaluation script for Multi-Target Multi-Camera (MTMC) tracking outputs.

Computes:
1. Ground truth benchmark metrics (MOTA, IDF1, Precision, Recall, ID switches)
   when GT annotations are provided.
2. Cross-camera structural metrics (global identity count, tracklet fragmentation,
   camera co-occurrence, concurrency, overlap integrity).
3. Visual diagnostics (Gantt identity timelines, occupancy plots, track length histograms).
4. Annotated review videos comparing GT and Tracker predictions.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_tracker_data(output_dir: Path):
    """Load tracks and observations from output directory."""
    tracks_dir = output_dir / 'tracks'
    obs_dir = output_dir / 'observations'

    tracks_by_camera = defaultdict(list)
    obs_by_camera_frame = defaultdict(lambda: defaultdict(list))
    all_global_ids = set()

    if tracks_dir.exists():
        for track_file in sorted(tracks_dir.glob('*.json')):
            cam_id = track_file.stem
            with open(track_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tracks_by_camera[cam_id] = data
                for t in data:
                    gid = t.get('global_id')
                    if gid is not None:
                        all_global_ids.add(gid)

    if obs_dir.exists():
        for obs_file in sorted(obs_dir.glob('*.jsonl')):
            cam_id = obs_file.stem
            with open(obs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    fid = int(record['frame_id'])
                    obs_by_camera_frame[cam_id][fid].append(record)
                    gid = record.get('global_id')
                    if gid is not None:
                        all_global_ids.add(gid)

    return tracks_by_camera, obs_by_camera_frame, all_global_ids


def load_ground_truth(gt_path: Path):
    """Load ground truth JSON format {frame_id: [{'id': ..., 'bbox_xyxy': [...]}]}."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Convert keys to int
    return {int(k): v for k, v in data.items()}


def evaluate_mot_metrics(gt_frames: dict, pred_frames: dict, iou_thresh: float = 0.5):
    """Compute MOT metrics (MOTA, MOTP, IDF1, Precision, Recall, IDSW)."""
    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    sum_iou = 0.0

    gt_to_pred_last_id = {}

    # For IDF1 trajectory-level bipartite matching
    gt_trajectory_frames = defaultdict(set)  # gt_id -> set of (frame, 'gt')
    pred_trajectory_frames = defaultdict(set)  # pred_id -> set of (frame, 'pred')
    matches_by_id = defaultdict(lambda: defaultdict(int))  # (gt_id, pred_id) -> count of matched frames

    for fid in tqdm(all_frames, desc="Evaluating MOT frames"):
        gts = gt_frames.get(fid, [])
        preds = pred_frames.get(fid, [])

        total_gt += len(gts)
        total_pred += len(preds)

        for g in gts:
            gt_trajectory_frames[g['id']].add(fid)
        for p in preds:
            pid = p.get('global_id') if p.get('global_id') is not None else p.get('local_track_id')
            pred_trajectory_frames[pid].add(fid)

        if not gts or not preds:
            if not gts:
                total_fp += len(preds)
            if not preds:
                total_fn += len(gts)
            continue

        cost_matrix = np.zeros((len(gts), len(preds)))
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                iou = compute_iou(g['bbox_xyxy'], p['bbox_xyxy'])
                cost_matrix[i, j] = 1.0 - iou if iou >= iou_thresh else 1.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - iou_thresh):
                matched_gt.add(r)
                matched_pred.add(c)
                total_tp += 1
                iou = 1.0 - cost_matrix[r, c]
                sum_iou += iou

                gid = gts[r]['id']
                pid = preds[c].get('global_id') if preds[c].get('global_id') is not None else preds[c].get('local_track_id')
                matches_by_id[gid][pid] += 1

                # ID switch detection
                if gid in gt_to_pred_last_id and gt_to_pred_last_id[gid] != pid:
                    total_idsw += 1
                gt_to_pred_last_id[gid] = pid

        total_fn += (len(gts) - len(matched_gt))
        total_fp += (len(preds) - len(matched_pred))

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    motp = (sum_iou / total_tp) if total_tp > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(1, total_gt)

    # Compute IDF1 via Hungarian matching on trajectory overlaps
    gt_ids = list(gt_trajectory_frames.keys())
    pred_ids = list(pred_trajectory_frames.keys())
    idtp = 0

    if gt_ids and pred_ids:
        overlap_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                overlap_matrix[i, j] = matches_by_id[gid].get(pid, 0)

        # Maximize overlap (minimize negative overlap)
        r_idx, c_idx = linear_sum_assignment(-overlap_matrix)
        for r, c in zip(r_idx, c_idx):
            idtp += int(overlap_matrix[r, c])

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    idp = idtp / max(1, idtp + idfp)
    idr = idtp / max(1, idtp + idfn)
    idf1 = 2 * idp * idr / max(1e-6, idp + idr)

    return {
        'total_gt_boxes': total_gt,
        'total_pred_boxes': total_pred,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'idsw': total_idsw,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'motp': round(motp, 4),
        'mota': round(mota, 4),
        'idf1': round(idf1, 4),
        'idp': round(idp, 4),
        'idr': round(idr, 4),
    }


def compute_structural_metrics(tracks_by_camera, obs_by_camera_frame, all_global_ids):
    """Compute dataset-wide multi-camera tracking structural metrics."""
    camera_ids = sorted(tracks_by_camera.keys())
    per_cam_stats = {}

    total_tracklets = 0
    all_durations = []

    # Map global_id -> cameras it appeared in and frame ranges
    global_id_lifespans = defaultdict(lambda: {'cameras': set(), 'frames': set(), 'boxes': 0})

    overlap_conflicts = 0

    for cam_id in camera_ids:
        cam_tracks = tracks_by_camera[cam_id]
        total_tracklets += len(cam_tracks)
        durations = [len(t.get('frames', [])) for t in cam_tracks]
        all_durations.extend(durations)

        per_cam_stats[cam_id] = {
            'tracklet_count': len(cam_tracks),
            'mean_duration_frames': round(float(np.mean(durations)), 1) if durations else 0,
            'median_duration_frames': round(float(np.median(durations)), 1) if durations else 0,
            'min_duration_frames': int(np.min(durations)) if durations else 0,
            'max_duration_frames': int(np.max(durations)) if durations else 0,
        }

        # Check for overlap integrity in observations
        for fid, records in obs_by_camera_frame[cam_id].items():
            gids = [r['global_id'] for r in records if r.get('global_id') is not None]
            if len(gids) != len(set(gids)):
                overlap_conflicts += 1

            for r in records:
                gid = r.get('global_id')
                if gid is not None:
                    global_id_lifespans[gid]['cameras'].add(cam_id)
                    global_id_lifespans[gid]['frames'].add(fid)
                    global_id_lifespans[gid]['boxes'] += 1

    # Camera distribution
    cam_coverage_counts = defaultdict(int)
    for gid, info in global_id_lifespans.items():
        c_count = len(info['cameras'])
        cam_coverage_counts[c_count] += 1

    return {
        'num_cameras': len(camera_ids),
        'total_tracklets': total_tracklets,
        'total_global_ids': len(all_global_ids),
        'overlap_conflicts': overlap_conflicts,
        'mean_tracklet_length': round(float(np.mean(all_durations)), 1) if all_durations else 0,
        'median_tracklet_length': round(float(np.median(all_durations)), 1) if all_durations else 0,
        'per_camera': per_cam_stats,
        'camera_coverage_counts': dict(cam_coverage_counts),
        'global_lifespans': global_id_lifespans,
    }


def generate_plots(results_dir: Path, structural_metrics: dict, obs_by_camera_frame: dict, mot_metrics: dict = None):
    """Generate diagnostic PNG plots."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Timeline / Gantt Chart of Global Identities
    lifespans = structural_metrics['global_lifespans']
    if lifespans:
        plt.figure(figsize=(12, max(6, len(lifespans) * 0.3)))
        sorted_gids = sorted(lifespans.keys(), key=lambda x: min(lifespans[x]['frames']) if lifespans[x]['frames'] else 0)

        for idx, gid in enumerate(sorted_gids):
            frames = sorted(lifespans[gid]['frames'])
            if not frames:
                continue
            cams = ", ".join(sorted(lifespans[gid]['cameras']))
            plt.plot([min(frames), max(frames)], [idx, idx], color='steelblue', linewidth=2.5, solid_capstyle='round')
            plt.scatter(frames, [idx] * len(frames), color='dodgerblue', s=3, alpha=0.5)
            plt.text(max(frames) + 5, idx, f"G{gid} ({cams})", verticalalignment='center', fontsize=8)

        plt.yticks(range(len(sorted_gids)), [f"G{gid}" for gid in sorted_gids], fontsize=8)
        plt.xlabel("Frame Number")
        plt.title("Active Global Identities Timeline across Cameras")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        timeline_path = results_dir / 'active_identities_timeline.png'
        plt.savefig(timeline_path, dpi=200)
        plt.close()

    # 2. Camera Occupancy over Time
    plt.figure(figsize=(12, 5))
    for cam_id, frames_dict in sorted(obs_by_camera_frame.items()):
        frames = sorted(frames_dict.keys())
        counts = [len(frames_dict[f]) for f in frames]
        plt.plot(frames, counts, label=cam_id, alpha=0.8, linewidth=1.5)

    plt.xlabel("Frame Number")
    plt.ylabel("Detected Persons Count")
    plt.title("Camera Occupancy (Detected Persons per Frame)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    occupancy_path = results_dir / 'camera_occupancy.png'
    plt.savefig(occupancy_path, dpi=200)
    plt.close()

    # 3. Tracklet Duration Histogram
    all_lengths = []
    for info in structural_metrics['per_camera'].values():
        pass
    for cam_id, tracks in obs_by_camera_frame.items():
        pass
    durations = []
    for gid, info in structural_metrics['global_lifespans'].items():
        durations.append(len(info['frames']))

    if durations:
        plt.figure(figsize=(8, 4.5))
        plt.hist(durations, bins=20, color='teal', edgecolor='black', alpha=0.7)
        plt.xlabel("Identity Lifespan (Active Frames)")
        plt.ylabel("Number of Global Identities")
        plt.title("Distribution of Global Identity Lifespans")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        hist_path = results_dir / 'track_length_distribution.png'
        plt.savefig(hist_path, dpi=200)
        plt.close()

    # 4. Summary Metric Chart
    plt.figure(figsize=(7, 4.5))
    if mot_metrics:
        labels = ['MOTA', 'IDF1', 'Precision', 'Recall', 'MOTP']
        values = [max(0, mot_metrics['mota']), mot_metrics['idf1'], mot_metrics['precision'], mot_metrics['recall'], mot_metrics['motp']]
        bars = plt.bar(labels, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b'], edgecolor='black', alpha=0.8)
        plt.ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        plt.title("Tracking Benchmark Metrics (IoU >= 0.5)")
    else:
        cov = structural_metrics['camera_coverage_counts']
        labels = [f"{k} Cam(s)" for k in sorted(cov.keys())]
        values = [cov[k] for k in sorted(cov.keys())]
        bars = plt.bar(labels, values, color='#1f77b4', edgecolor='black', alpha=0.8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval}", ha='center', va='bottom', fontsize=9)
        plt.title("Global Identities Cross-Camera Coverage")

    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    summary_bar_path = results_dir / 'metrics_summary_bar.png'
    plt.savefig(summary_bar_path, dpi=200)
    plt.close()


def export_csvs(results_dir: Path, structural: dict, mot: dict = None):
    """Save metrics to CSV files."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary metrics CSV
    summary_path = results_dir / 'summary_metrics.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['num_cameras', structural['num_cameras']])
        writer.writerow(['total_tracklets', structural['total_tracklets']])
        writer.writerow(['total_global_ids', structural['total_global_ids']])
        writer.writerow(['overlap_conflicts', structural['overlap_conflicts']])
        writer.writerow(['mean_tracklet_length_frames', structural['mean_tracklet_length']])
        writer.writerow(['median_tracklet_length_frames', structural['median_tracklet_length']])

        if mot:
            writer.writerow(['--- MOT Metrics ---', '---'])
            for k, v in mot.items():
                writer.writerow([k, v])

    # 2. Per-camera metrics CSV
    per_cam_path = results_dir / 'per_camera_metrics.csv'
    with open(per_cam_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Camera', 'Tracklets', 'Mean Duration', 'Median Duration', 'Min Duration', 'Max Duration'])
        for cam_id, stats in sorted(structural['per_camera'].items()):
            writer.writerow([
                cam_id,
                stats['tracklet_count'],
                stats['mean_duration_frames'],
                stats['median_duration_frames'],
                stats['min_duration_frames'],
                stats['max_duration_frames'],
            ])

    # 3. Global identities lifespan CSV
    identities_path = results_dir / 'global_identities.csv'
    with open(identities_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Global ID', 'Num Cameras', 'Cameras', 'Total Observations', 'Start Frame', 'End Frame', 'Duration Frames'])
        for gid, info in sorted(structural['global_lifespans'].items()):
            frames = sorted(info['frames'])
            start_f = min(frames) if frames else 0
            end_f = max(frames) if frames else 0
            dur = end_f - start_f + 1 if frames else 0
            writer.writerow([
                f"G{gid}",
                len(info['cameras']),
                ";".join(sorted(info['cameras'])),
                info['boxes'],
                start_f,
                end_f,
                dur,
            ])


def render_sample_video(video_path: Path, output_video_path: Path, pred_obs: dict, gt_obs: dict = None, max_frames: int = 350):
    """Render annotated sample video with GT and Tracker bounding boxes."""
    if not video_path.exists():
        print(f"Skipping video rendering: {video_path} not found.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping video rendering: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_render = min(total_frames, max_frames) if max_frames > 0 else total_frames

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    print(f"Rendering sample evaluation video ({frames_to_render} frames @ {fps} fps)...")
    for fid in tqdm(range(frames_to_render), desc=f"Rendering {output_video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw Ground Truth boxes (Blue)
        if gt_obs and fid in gt_obs:
            for g in gt_obs[fid]:
                x1, y1, x2, y2 = map(int, g['bbox_xyxy'])
                oid = g['id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                label = f"GT:{oid}"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)

        # Draw Prediction boxes (Green)
        if pred_obs and fid in pred_obs:
            for p in pred_obs[fid]:
                x1, y1, x2, y2 = map(int, p['bbox_xyxy'])
                gid = p.get('global_id')
                lid = p.get('local_track_id')
                color = (0, 255, 0) if gid is not None else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"G{gid} (L{lid})" if gid is not None else f"L{lid}"
                cv2.putText(frame, label, (x1, min(h - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay frame indicator
        header_text = f"Frame: {fid}"
        if gt_obs:
            header_text += " | Blue=GT | Green=Pred"
        else:
            header_text += " | Green=Pred (G=Global, L=Local)"
        cv2.putText(frame, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved review video: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTMC tracking outputs")
    parser.add_argument('--output-dir', required=True, help='Path to tracker output directory (containing tracks/ and observations/)')
    parser.add_argument('--gt-path', default=None, help='Optional path to ground truth JSON file')
    parser.add_argument('--gt-camera', default='cam01', help='Camera ID corresponding to ground truth (default: cam01)')
    parser.add_argument('--video-dir', default=None, help='Optional path to video directory for rendering review video')
    parser.add_argument('--results-dir', default=None, help='Directory to save evaluation results (default: <output-dir>/evaluation)')
    parser.add_argument('--save-video', action='store_true', help='Render sample review video with overlay boxes')
    parser.add_argument('--max-video-frames', type=int, default=350, help='Max frames to render in sample video (default: 350)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")

    results_dir = Path(args.results_dir) if args.results_dir else (out_dir / 'evaluation')
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"MTMC Tracker Evaluation")
    print(f"Output directory: {out_dir}")
    print(f"Results target:   {results_dir}")
    print(f"=======================================================\n")

    # 1. Load tracker data
    tracks_by_cam, obs_by_cam_frame, all_gids = load_tracker_data(out_dir)
    print(f"Loaded {sum(len(t) for t in tracks_by_cam.values())} tracklets across {len(tracks_by_cam)} cameras.")
    print(f"Identified {len(all_gids)} unique global identities.")

    # 2. Structural metrics
    structural_metrics = compute_structural_metrics(tracks_by_cam, obs_by_cam_frame, all_gids)

    # 3. Ground truth evaluation (if provided)
    mot_metrics = None
    gt_frames = None
    if args.gt_path:
        gt_path = Path(args.gt_path)
        if gt_path.exists():
            print(f"\nLoading ground truth for {args.gt_camera} from {gt_path}...")
            gt_frames = load_ground_truth(gt_path)
            pred_frames = obs_by_cam_frame.get(args.gt_camera, {})
            mot_metrics = evaluate_mot_metrics(gt_frames, pred_frames)
            print("\n--- Ground Truth Benchmark Results ---")
            for k, v in mot_metrics.items():
                print(f"  {k:20s}: {v}")
        else:
            print(f"Warning: Ground truth file not found at {gt_path}")

    # Print structural summary
    print("\n--- Multi-Camera Structural Summary ---")
    print(f"  Total Global IDs     : {structural_metrics['total_global_ids']}")
    print(f"  Total Tracklets      : {structural_metrics['total_tracklets']}")
    print(f"  Mean Tracklet Length : {structural_metrics['mean_tracklet_length']} frames")
    print(f"  Median Tracklet Lgth : {structural_metrics['median_tracklet_length']} frames")
    print(f"  Overlap Conflicts    : {structural_metrics['overlap_conflicts']}")
    print("  Cross-Camera Coverage:")
    for num_cams, count in sorted(structural_metrics['camera_coverage_counts'].items()):
        print(f"    Present in {num_cams} camera(s): {count} identities")

    # 4. Export CSVs
    export_csvs(results_dir, structural_metrics, mot_metrics)
    print(f"\nSaved CSV summaries to {results_dir}")

    # 5. Generate plots
    generate_plots(results_dir, structural_metrics, obs_by_cam_frame, mot_metrics)
    print(f"Saved diagnostic plots to {results_dir}")

    # 6. Render sample video
    if args.save_video and args.video_dir:
        vdir = Path(args.video_dir)
        target_cam = args.gt_camera if gt_frames else (sorted(tracks_by_cam.keys())[0] if tracks_by_cam else 'cam01')
        raw_video = vdir / f'{target_cam}.mp4'
        sample_out = results_dir / f'sample_review_{target_cam}.mp4'
        pred_obs = obs_by_cam_frame.get(target_cam, {})
        render_sample_video(raw_video, sample_out, pred_obs, gt_frames, max_frames=args.max_video_frames)

    print(f"\nEvaluation complete! Results stored at: {results_dir.resolve()}\n")


if __name__ == '__main__':
    main()

