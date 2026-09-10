# How the tracker works

## Plain-language explanation

Imagine four security cameras watching the same place. The tracker does four
things:

1. It looks at each new frame and finds people.
2. It remembers where each person was in that camera and gives them a temporary
   local number, such as “camera 2, person 7”.
3. When that short local journey is stable enough, it creates a compact
   description of the person’s appearance (an embedding). This is saved so it
   can later be placed in a vector database.
4. It compares that description with descriptions from other cameras and gives
   matching journeys the same global number.

The final video is produced after processing finishes. It is only a review
output; it does not run the tracker again.

## What happens in incremental mode

The MP4 files are treated like live camera streams. The program takes one frame
from each camera in rotation:

```text
camera 1 frame 1 → camera 2 frame 1 → camera 3 frame 1 → camera 4 frame 1
camera 1 frame 2 → camera 2 frame 2 → camera 3 frame 2 → camera 4 frame 2
...
```

Detection and local tracking happen on every frame. Feature extraction is not
needed on every frame: the tracker keeps a small sample of good crops from each
local tracklet and extracts a pooled descriptor when that tracklet becomes
inactive. This reduces repeated work while retaining the appearance signal.

Global association is **not** performed every 500 frames. Messages such as
`processed 3500 frames; 345 tracklets finalized` are progress messages only.
Association happens whenever a local tracklet becomes inactive, and again for
any tracklets still active when a camera stream ends.

## Technical flow

```text
MP4/RTSP camera input
        │
        ▼
YOLO person detections (every frame)
        │
        ▼
Independent ByteTrack state per camera
        │
        ▼
Local tracklet: frame IDs + boxes + confidences
        │
        ├── representative crops → OSNet ReID embedding
        ├── representative crops → color descriptor
        └── trajectory/confidence → motion and quality values
        │
        ▼
Topology/temporal gates + weighted similarity
        │
        ▼
Incremental global identity gallery
        │
        ▼
JSON/NPZ outputs and final stitched review video
```

## Saved data

`observations/cam01.jsonl` contains one readable record per detected person per
frame. It includes camera ID, frame ID, timestamp, bounding box, confidence,
local track ID, and final global ID.

`tracks/cam01.json` contains the grouped local tracklets, including all frame
IDs and bounding boxes for each local track.

`global_id_map.json` maps `(camera_id, local_track_id)` to a global ID.

`features/cam01_12.npz` is a NumPy archive for one tracklet. Its main fields are
`reid` (normally a 512-dimensional float vector), optional `color` and
`motion` vectors, and metadata fields for camera ID, local track ID, and global
ID. The `reid` array and its metadata are the fields a future vector database
would normally store.

## Visualization layout

The renderer calculates a square grid automatically:

| Cameras | Layout |
|---:|:---|
| 1 | 1×1 |
| 2–4 | 2×2 |
| 5–9 | 3×3 |
| 10–16 | 4×4 |

Each tile is labeled with its camera ID. Render saved results without rerunning
inference:

```bash
python scripts/visualize_outputs.py --config configs/epfl_4p.yaml --videos ./data/videos
```
