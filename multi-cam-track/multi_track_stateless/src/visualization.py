"""Per-camera visualization renderer and multi-camera video stitcher.

Renders:
1. Individual annotated review videos for each camera stream.
2. Stitched combined review video (horizontal side-by-side for 2 cameras, or grid for 4+).
"""
import math
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


# Distinct high-contrast color palette for global identities
PALETTE = [
    (0, 220, 0),     # Bright Green
    (255, 128, 0),   # Blue-Orange
    (0, 165, 255),   # Orange
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (0, 0, 255),     # Red
    (255, 191, 0),   # Deep Sky Blue
    (0, 255, 128),   # Spring Green
    (203, 192, 255), # Pink
    (128, 255, 0),   # Chartreuse
]


def _get_color(global_id):
    if isinstance(global_id, int) and global_id > 0:
        return PALETTE[(global_id - 1) % len(PALETTE)]
    return (200, 200, 200)


def render_single_camera(video_path: Path, camera_id: str, tracks, mapping, output_path: Path,
                         display_resolution=None, show_local_id=True):
    """Render an individual annotated video for a single camera stream.

    Args:
        video_path: Path to the raw camera video file.
        camera_id: Identifier string of the camera (e.g. 'cam01').
        tracks: List of Tracklet objects belonging to this camera.
        mapping: Dictionary mapping (camera_id, local_id) -> global_id.
        output_path: Path to save the annotated .mp4 video.
        display_resolution: Optional (width, height) tuple to resize output.
        show_local_id: If True, renders label as 'G1 [L3]'. If False, renders 'G1'.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_w, out_h = display_resolution if display_resolution else (orig_w, orig_h)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

    # Index tracklets by frame number for O(1) per-frame lookup
    boxes_by_frame = {}
    for t in tracks:
        gid = mapping.get((camera_id, t.local_id), t.global_id)
        for f_no, box in zip(t.frames, t.boxes):
            boxes_by_frame.setdefault(f_no, []).append((box, t.local_id, gid))

    frame_idx = 0
    with tqdm(total=total_frames or None, desc=f'rendering {camera_id}', unit='frame') as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Overlay bounding boxes and identities
            if frame_idx in boxes_by_frame:
                for box, local_id, gid in boxes_by_frame[frame_idx]:
                    x1, y1, x2, y2 = map(int, box)
                    color = _get_color(gid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    if show_local_id:
                        label = f"G{gid} [L{local_id}]" if gid is not None else f"L{local_id}"
                    else:
                        label = f"G{gid}" if gid is not None else f"L{local_id}"

                    (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                    ly = max(lh + baseline + 4, y1)
                    cv2.rectangle(frame, (x1, ly - lh - baseline - 4), (x1 + lw + 6, ly), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 3, ly - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            # Camera name overlay
            cv2.putText(frame, f"{camera_id} | Frame {frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{camera_id} | Frame {frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)

            if (out_w, out_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (out_w, out_h))

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


def stitch_videos(video_paths: dict, output_path: Path, display_resolution=(1280, 720)):
    """Stitch multiple individual tracked camera videos into a single synchronized video.

    For 2 cameras: side-by-side horizontal view [Cam 1 | Cam 2].
    For 3+ cameras: 2x2 grid view.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cams = list(video_paths.keys())
    caps = {c: cv2.VideoCapture(str(video_paths[c])) for c in cams}

    fps = caps[cams[0]].get(cv2.CAP_PROP_FPS) or 25.0
    frame_counts = [int(caps[c].get(cv2.CAP_PROP_FRAME_COUNT) or 0) for c in cams]
    total_frames = max(frame_counts) if frame_counts else None

    # Calculate grid layout
    num_cams = len(cams)
    if num_cams == 2:
        cols, rows = 2, 1
    else:
        cols = max(1, math.ceil(math.sqrt(num_cams)))
        rows = max(1, math.ceil(num_cams / cols))

    out_w, out_h = display_resolution
    tile_w = out_w // cols
    tile_h = out_h // rows

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

    with tqdm(total=total_frames, desc='stitching videos', unit='frame') as pbar:
        while True:
            tiles = []
            any_active = False

            for c in cams:
                ok, frame = caps[c].read()
                if ok:
                    any_active = True
                    resized = cv2.resize(frame, (tile_w, tile_h))
                else:
                    resized = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    cv2.putText(resized, f"{c} (ended)", (20, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
                tiles.append(resized)

            if not any_active:
                break

            # Pad empty slots in grid if any
            while len(tiles) < cols * rows:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

            # Assemble grid rows
            grid_rows = []
            for r in range(rows):
                row_tiles = tiles[r * cols : (r + 1) * cols]
                grid_rows.append(cv2.hconcat(row_tiles))
            canvas = cv2.vconcat(grid_rows)
            canvas = cv2.resize(canvas, (out_w, out_h))

            writer.write(canvas)
            pbar.update(1)

    for cap in caps.values():
        cap.release()
    writer.release()
