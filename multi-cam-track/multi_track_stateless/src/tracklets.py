from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    frames: list
    boxes: list
    confidences: list
    fps: float
    video_path: str
    global_id: int = None
    crops: list = None

    @property
    def track_id(self): return self.local_id
    @property
    def start_frame(self): return self.frames[0] if self.frames else 0
    @property
    def end_frame(self): return self.frames[-1] if self.frames else 0
    @property
    def start_time(self): return self.frames[0] / self.fps if self.frames else 0.
    @property
    def end_time(self): return self.frames[-1] / self.fps if self.frames else 0.
    @property
    def last_box(self): return np.asarray(self.boxes[-1], dtype=float)
    @property
    def first_box(self): return np.asarray(self.boxes[0], dtype=float)


def save_tracklets(path, tracks):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    clean = []
    for t in tracks:
        d = asdict(t)
        d.pop('crops', None)
        clean.append(d)
    path.write_text(json.dumps(clean, indent=2), encoding='utf-8')


def load_tracklets(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return [Tracklet(**x) for x in data]
