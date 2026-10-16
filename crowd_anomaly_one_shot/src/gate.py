"""Recall-first gate and region event generation module for Phase 1D.

Extracts spatial anomaly regions via connected components, tracks events across time
with persistence and hysteresis, assigns two-level severity ('confirm' vs 'direct'),
and emits events conforming to the system contract.
"""

import cv2
import numpy as np


def threshold_mask(score_map: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean mask of grid cells exceeding threshold."""
    assert score_map.ndim == 2, f"Expected 2D score map, got {score_map.shape}"
    return (score_map >= threshold).astype(bool)


def find_regions(mask: np.ndarray, min_cells: int = 2) -> list:
    """Find contiguous connected component bounding boxes in grid coordinates [c1, r1, c2, r2].
    
    Uses cv2.connectedComponentsWithStats with 8-way connectivity.
    Regions with fewer than min_cells are filtered out to suppress single-cell noise.
    """
    assert mask.ndim == 2 and mask.dtype == bool, "mask must be 2D boolean array"
    u8_mask = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(u8_mask, connectivity=8)

    regions = []
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_cells:
            c1 = int(stats[label, cv2.CC_STAT_LEFT])
            r1 = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            regions.append([c1, r1, c1 + w, r1 + h])
    return regions


def grid_box_to_pixels(box: list, grid_h: int, grid_w: int, frame_shape: tuple) -> list:
    """Convert grid box [c1, r1, c2, r2] to pixel coordinates [x1, y1, x2, y2]."""
    h_frame, w_frame = frame_shape[:2]
    c1, r1, c2, r2 = box
    x1 = int(round(c1 * w_frame / grid_w))
    y1 = int(round(r1 * h_frame / grid_h))
    x2 = int(round(c2 * w_frame / grid_w))
    y2 = int(round(r2 * h_frame / grid_h))
    return [max(0, x1), max(0, y1), min(w_frame, x2), min(h_frame, y2)]


def box_iou(box_a: list, box_b: list) -> float:
    """Compute Intersection-over-Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


def merge_boxes(boxes_with_scores: list, iou_thresh: float = 0.25) -> list:
    """Merge overlapping boxes from multi-scale detections via spatial union, retaining max score.
    
    If two boxes overlap with IoU >= iou_thresh, they represent the same physical disturbance
    detected at different grid granularities and are united.
    """
    if not boxes_with_scores:
        return []
    # Sort descending by score
    boxes = sorted(boxes_with_scores, key=lambda x: x[1], reverse=True)
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        cur_box, cur_score = boxes[i][0][:], boxes[i][1]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and box_iou(cur_box, boxes[j][0]) >= iou_thresh:
                # Union of coordinates
                cur_box[0] = min(cur_box[0], boxes[j][0][0])
                cur_box[1] = min(cur_box[1], boxes[j][0][1])
                cur_box[2] = max(cur_box[2], boxes[j][0][2])
                cur_box[3] = max(cur_box[3], boxes[j][0][3])
                cur_score = max(cur_score, boxes[j][1])
                used[j] = True
        merged.append((cur_box, cur_score))
    return merged


def smooth_box(old_box: list, new_box: list, alpha: float = 0.7) -> list:
    """Exponential moving average on box coordinates [x1, y1, x2, y2] to suppress jitter.
    
    alpha=0.7 gives responsive tracking while preventing bounding box boundary flicker.
    """
    return [
        int(round(alpha * new_box[i] + (1.0 - alpha) * old_box[i]))
        for i in range(4)
    ]


def update_events(open_events: list, regions_this_frame: list, frame_idx: int, fps: float,
                  min_event_seconds: float = 1.5, max_gap_seconds: float = 1.0,
                  low_threshold: float = 3.5, high_threshold: float = 6.0,
                  next_event_id: int = 1) -> tuple:
    """Update event tracking: extend matching candidates, open new candidates, close finished events.
    
    Returns:
        (updated_open_events, finished_events, next_event_id)
    """
    min_event_frames = int(round(min_event_seconds * fps))
    max_gap_frames = int(round(max_gap_seconds * fps))

    finished_events = []
    active_open = []
    matched_regions = [False] * len(regions_this_frame)

    # 1. Match detected regions with open events
    for ev in open_events:
        best_iou, best_idx = 0.0, -1
        for idx, (rbox, rscore) in enumerate(regions_this_frame):
            if not matched_regions[idx]:
                iou = box_iou(ev["box"], rbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

        if best_iou >= 0.15:  # Spatial continuity threshold
            rbox, rscore = regions_this_frame[best_idx]
            matched_regions[best_idx] = True
            ev["end_frame"] = frame_idx
            ev["box"] = smooth_box(ev["box"], rbox)
            ev["score"] = max(ev["score"], float(rscore))
            ev["gap_frames"] = 0
            ev["active_frames"] += 1
            active_open.append(ev)
        else:
            ev["gap_frames"] += 1
            if ev["gap_frames"] <= max_gap_frames:
                active_open.append(ev)
            else:
                # Event ended due to gap hysteresis timeout
                if ev["active_frames"] >= min_event_frames:
                    level = "direct" if ev["score"] >= high_threshold else "confirm"
                    finished_events.append({
                        "id": ev["id"], "start_frame": ev["start_frame"], "end_frame": ev["end_frame"],
                        "box": ev["box"], "score": ev["score"], "source": "motion",
                        "level": level, "origin": "gate"
                    })

    # 2. Open new candidate tracks for unmatched regions
    for idx, (rbox, rscore) in enumerate(regions_this_frame):
        if not matched_regions[idx]:
            active_open.append({
                "id": next_event_id,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "box": rbox,
                "score": float(rscore),
                "gap_frames": 0,
                "active_frames": 1,
            })
            next_event_id += 1

    return active_open, finished_events, next_event_id


def finalize_events(open_events: list, fps: float, min_event_seconds: float = 1.5,
                    low_threshold: float = 3.5, high_threshold: float = 6.0) -> list:
    """Finalize any remaining open events at video end that meet min_event_seconds."""
    min_event_frames = int(round(min_event_seconds * fps))
    finalized = []
    for ev in open_events:
        if ev["active_frames"] >= min_event_frames:
            level = "direct" if ev["score"] >= high_threshold else "confirm"
            finalized.append({
                "id": ev["id"], "start_frame": ev["start_frame"], "end_frame": ev["end_frame"],
                "box": ev["box"], "score": ev["score"], "source": "motion",
                "level": level, "origin": "gate"
            })
    return finalized


# Module-level cache for visual embedding models
_EMBEDDING_MODELS = {}


def get_embedding_model(model_name: str = "mobilenet", device: str = None):
    """Load visual appearance feature extractor: 'mobilenet' (MobileNetV3) or 'clip' (OpenAI ViT-B/32)."""
    global _EMBEDDING_MODELS
    key = model_name.lower()
    if key not in _EMBEDDING_MODELS:
        import torch
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if key in ["clip", "clip-vit"]:
            from transformers import CLIPVisionModelWithProjection
            m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        else:
            import torchvision.models as models
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            m.classifier = torch.nn.Identity()
            m = m.to(dev).eval()
        _EMBEDDING_MODELS[key] = m
    return _EMBEDDING_MODELS[key]


def extract_crop_embedding(crop: np.ndarray, model=None, model_type: str = "mobilenet", device: str = None) -> np.ndarray:
    """Extract L2-normalized visual appearance embedding from a BGR crop (H, W, 3) uint8."""
    import torch
    if model is None:
        model = get_embedding_model(model_name=model_type, device=device)
    dev = next(model.parameters()).device

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (resized - mean) / std

    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        if model_type.lower() in ["clip", "clip-vit"] or hasattr(model, "vision_model"):
            out = model(pixel_values=tensor)
            feat = out.image_embeds.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            feat = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    norm_val = np.linalg.norm(feat)
    return feat / max(norm_val, 1e-6)


def appearance_distance(crop_embedding: np.ndarray, normal_embeddings: np.ndarray) -> float:
    """Compute visual novelty distance: 1.0 minus maximum cosine similarity to normal bank."""
    if normal_embeddings is None or len(normal_embeddings) == 0:
        return 1.0
    sims = np.dot(normal_embeddings, crop_embedding)
    max_sim = float(np.max(sims))
    return float(np.clip(1.0 - max_sim, 0.0, 2.0))


def fit_normal_embeddings(normal_frames: list, camera: str, model_type: str = "mobilenet", out_dir: str = "weights", max_samples: int = 40) -> np.ndarray:
    """Fit and save normal appearance embedding bank from normal footage crops."""
    import os
    model = get_embedding_model(model_name=model_type)
    embeddings = []
    step = max(1, len(normal_frames) // max_samples)

    for i in range(0, len(normal_frames), step):
        f = normal_frames[i]
        h, w = f.shape[:2]
        # Crop center / typical pedestrian walkway region
        cy, cx = h // 2, w // 2
        crop = f[max(0, cy - 100) : min(h, cy + 100), max(0, cx - 100) : min(w, cx + 100)]
        if crop.size > 0:
            emb = extract_crop_embedding(crop, model=model, model_type=model_type)
            embeddings.append(emb)

    arr = np.stack(embeddings, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"normal_emb_{camera}_{model_type}.npz")
    np.savez(save_path, embeddings=arr)
    # Also save primary copy
    np.savez(os.path.join(out_dir, f"normal_emb_{camera}.npz"), embeddings=arr)
    return arr


def confirm_event(event: dict, score_history: list, extra_checks: dict = None, fps: float = 5.0) -> tuple:
    """Second-gate validation for low-confidence 'confirm' events.
    
    Returns:
        (is_forwarded: bool, reason: str)
    """
    if extra_checks is None:
        extra_checks = {}

    dur_sec = (event["end_frame"] - event["start_frame"]) / fps
    min_confirm_sec = extra_checks.get("confirm_min_sec", 2.0)

    # 1. Persistence or rising trend check
    is_rising = False
    if len(score_history) >= 4:
        # Simple linear slope across last scores
        x = np.arange(len(score_history))
        slope = np.polyfit(x, score_history, 1)[0]
        is_rising = slope > 0.05

    if dur_sec < min_confirm_sec and not is_rising:
        return False, f"dropped: short duration ({dur_sec:.1f}s < {min_confirm_sec:.1f}s) and non-rising trend"

    # 2. Spatial neighbor support check
    neighbor_support = extra_checks.get("neighbor_support", True)
    if not neighbor_support:
        return False, "dropped: lack of spatial neighbor support (isolated cell artifact)"

    # 3. Optional visual appearance novelty check
    if extra_checks.get("use_appearance", False):
        app_dist = extra_checks.get("appearance_dist", 1.0)
        app_th = extra_checks.get("appearance_threshold", 0.25)
        if app_dist < app_th:
            return False, f"dropped: appearance matches normal crowd background (distance {app_dist:.3f} < {app_th:.2f})"

    # 4. Optional match schedule surge relaxation
    if extra_checks.get("use_schedule_relaxation", False):
        is_surge_time = extra_checks.get("is_expected_surge", False)
        surge_threshold = extra_checks.get("surge_confirm_threshold", 5.0)
        if is_surge_time and event["score"] < surge_threshold:
            return False, f"dropped: expected crowd surge window (score {event['score']:.1f} < {surge_threshold:.1f})"

    return True, f"confirmed: sustained activity ({dur_sec:.1f}s) and multi-cell spatial support"

