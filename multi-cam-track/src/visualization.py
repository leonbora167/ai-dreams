from pathlib import Path
import cv2
import math
import numpy as np
from tqdm import tqdm


def render(videos, mapping, tracks, cfg):
    """Render a simple synchronized camera grid; skipped when no output videos exist."""
    if not videos: return
    # Keep rendering isolated so association can be used without GUI/display dependencies.
    out=Path(cfg['output']['video']); out.parent.mkdir(parents=True, exist_ok=True)
    camera_ids = list(videos)
    caps={c:cv2.VideoCapture(str(videos[c])) for c in camera_ids}
    size=tuple(cfg['system']['display_resolution']); writer=cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    by_cam={c:[t for t in tracks if t.camera_id==c] for c in camera_ids}
    columns = max(1, math.ceil(math.sqrt(len(camera_ids))))
    rows = max(1, math.ceil(len(camera_ids) / columns))
    tile_w, tile_h = size[0] // columns, size[1] // rows
    total_frames = []
    for path in videos.values():
        probe = cv2.VideoCapture(str(path))
        total_frames.append(int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        probe.release()
    total_frames = min(total_frames) if total_frames and all(total_frames) else None
    progress = tqdm(total=total_frames, desc='rendering visualization', unit='frame')
    while True:
        frames=[]; any_frame=False
        for cam in camera_ids:
            cap=caps[cam]
            ok,frame=cap.read()
            if ok:
                any_frame=True
                n=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                for t in by_cam[cam]:
                    if n in t.frames:
                        k=t.frames.index(n); x1,y1,x2,y2=map(int,t.boxes[k]); gid=mapping.get((cam,t.local_id),'?')
                        # Thin outlines remain readable after the camera grid is resized.
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),1, cv2.LINE_AA)
                        label = f'G{gid}'
                        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
                        ly = max(lh + baseline + 2, y1)
                        cv2.rectangle(frame, (x1, ly - lh - baseline - 2), (x1 + lw + 4, ly), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x1 + 2, ly - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, .48, (150, 255, 150), 1, cv2.LINE_AA)
                # Camera label without a filled banner; use a subtle outline for readability.
                cv2.putText(frame, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                frame = None
            if frame is not None:
                frame=cv2.resize(frame,(tile_w,tile_h)); frames.append(frame)
            else:
                blank = np.zeros((tile_h, tile_w, 3), dtype='uint8')
                cv2.putText(blank, f'{cam} (ended)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (180, 180, 180), 2)
                frames.append(blank)
        if not any_frame: break
        while len(frames) < rows * columns:
            frames.append(np.zeros((tile_h, tile_w, 3), dtype='uint8'))
        canvas = cv2.vconcat([cv2.hconcat(frames[i:i+columns]) for i in range(0, rows*columns, columns)])
        canvas=cv2.resize(canvas,size); writer.write(canvas)
        progress.update(1)
    writer.release()
    for c in caps.values(): c.release()
    progress.close()
