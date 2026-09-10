from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
from .tracklets import Tracklet, save_tracklets


def track_video(camera_id, video_path, cfg, output_path, log=print):
    """Run Ultralytics' ByteTrack and persist compact tracklets.

    The import/model are deliberately inside this function: disabled/dry-run
    workflows do not load detector dependencies or model weights.
    """
    from ultralytics import YOLO
    model = YOLO(cfg['detector']['model'])
    cap = cv2.VideoCapture(str(video_path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    tracks = defaultdict(lambda: [[], [], []])
    results = model.track(source=str(video_path), stream=True, persist=True,
                          tracker='bytetrack.yaml', conf=cfg['detector']['confidence'],
                          classes=cfg['detector']['classes'], device=cfg['detector']['device'], verbose=False)
    bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc=f'{camera_id} tracking', unit='frame')
    for frame_no, result in enumerate(results):
        bar.update(1)
        if frame_no and frame_no % 500 == 0:
            log(f'    {camera_id}: processed {frame_no} frames')
        boxes = result.boxes
        if boxes is None or boxes.id is None: continue
        for xyxy, tid, conf in zip(boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(), boxes.conf.cpu().tolist()):
            tracks[tid][0].append(frame_no); tracks[tid][1].append(xyxy); tracks[tid][2].append(float(conf))
    cap.release()
    bar.close()
    out = [Tracklet(camera_id, int(tid), *vals, float(fps), str(video_path)) for tid, vals in tracks.items()]
    save_tracklets(output_path, out)
    log(f'    {camera_id}: detector/tracker finished; cache written to {output_path}')
    return out
