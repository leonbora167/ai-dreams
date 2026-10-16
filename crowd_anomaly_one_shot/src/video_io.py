"""Video I/O helper functions for reading, subsampling, and writing frames."""

import os
import cv2
import numpy as np


def get_video_info(path: str) -> dict:
    """Read video metadata from path and return width, height, fps, and frame_count."""
    assert os.path.exists(path), f"Video file not found: {path}"
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Failed to open video: {path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count}


def read_frames(path: str, target_fps: float, width: int, max_frames: int = None, start_sec: float = 0.0):
    """Yield (frame_index, timestamp_sec, frame) subsampled to target_fps and resized to width preserving aspect ratio.
    
    Subsampling method:
    Calculates target source frame index as round((start_sec + k / target_fps) * src_fps).
    For example, from 30 FPS to 5 FPS, step N = round(30 / 5) = 6, yielding every 6th frame.
    Output frame: np.ndarray (H_out, width, 3), dtype=uint8, BGR.
    """
    assert os.path.exists(path), f"Video file not found: {path}"
    info = get_video_info(path)
    src_fps = info["fps"] if info["fps"] > 0 else target_fps
    src_w, src_h = info["width"], info["height"]
    assert src_w > 0 and src_h > 0, f"Invalid video dimensions: {src_w}x{src_h}"

    new_h = int(round(src_h * (width / src_w)))
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Failed to open video: {path}"

    k = 0
    while True:
        if max_frames is not None and k >= max_frames:
            break

        timestamp_sec = start_sec + (k / target_fps)
        target_src_idx = int(round(timestamp_sec * src_fps))
        if target_src_idx >= info["frame_count"] and info["frame_count"] > 0:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_src_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        resized_frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
        assert resized_frame.shape == (new_h, width, 3), f"Unexpected shape {resized_frame.shape}"
        assert resized_frame.dtype == np.uint8, f"Unexpected dtype {resized_frame.dtype}"

        yield k, timestamp_sec, resized_frame
        k += 1

    cap.release()


def write_video(frames, path: str, fps: float):
    """Write an iterable or list of frames (H, W, 3 uint8 BGR) to an mp4 video file at path."""
    writer = None
    count = 0
    try:
        for frame in frames:
            assert isinstance(frame, np.ndarray), "Frame must be a numpy ndarray"
            assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
            assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"

            if writer is None:
                h, w = frame.shape[:2]
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
                assert writer.isOpened(), f"Failed to open VideoWriter for {path}"

            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.release()
    assert count > 0, f"No frames written to {path}"

