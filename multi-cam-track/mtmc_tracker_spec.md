# Multi-Target Multi-Camera (MTMC) Tracking POC — Agent Build Spec

## Goal

Build a local, offline, config-driven Python POC that tracks people across
**non-overlapping** cameras and assigns each person a consistent `global_id`
across all camera feeds, using every discriminative signal available from
the video (not just ReID appearance). Single command to run:

```bash
python run.py --config configs/poc.yaml --videos ./data/videos
```

No Kafka, Redis, databases, REST APIs, Kubernetes, or distributed workers.
One Python process. This can evolve into production infra later — don't
build that now.

## Architecture

Two responsibilities, kept strictly separate:

```
Camera video → Detector → Single-camera tracker → Local tracklets   [per camera, independent]
Local tracklets → Feature extraction → Cross-camera association → Global IDs   [global stage]
```

Match **tracklets** (many pooled observations of one local track), not
individual frames. A tracklet is far more stable than a single crop.

```
                    VIDEO DIRECTORY
                          |
              +-----------+-----------+
              |           |           |
           Camera 1    Camera 2    Camera N
              |           |           |
        YOLO→ByteTrack  YOLO→ByteTrack  YOLO→ByteTrack
              |           |           |
          Tracklets    Tracklets    Tracklets
              +-----------+-----------+
                          |
                  Feature Manager
              (ReID | Color | Motion | Temporal | Topology | Quality)
                          |
                Candidate Generator (topology + temporal gating)
                          |
                Association Scorer (weighted fusion → similarity matrix)
                          |
                Global Identity Gallery (clustering)
                          |
                  Visualization → per-camera + combined grid video
```

## Feature modules (all toggleable, this is the core requirement)

Every module below must be independently switchable in YAML. A disabled
module must not load its model or run its computation — this keeps the POC
fast when you only want to test one signal at a time.

| Module | Signal | Why it matters for non-overlapping cameras |
|---|---|---|
| `reid` | Appearance embedding (OSNet, 512D) | Primary signal — works even with zero geometric overlap |
| `color` | Dominant clothing color histogram | Cheap, fast pre-filter / tie-breaker when ReID is ambiguous |
| `motion` | Trajectory direction/speed at camera exit-entry | Useful for camera-transition consistency |
| `temporal` | Time-gap gating between exit (cam A) and entry (cam B) | Prevents impossible matches (person can't teleport instantly) |
| `topology` | Which camera pairs are physically adjacent / plausible transition times | The single highest-value module for non-overlapping setups — without it ReID alone over-merges lookalikes |
| `quality` | Detection confidence / crop sharpness filtering | Drops low-quality crops before they pollute the gallery |
| `geometry` | Homography (only meaningful if cameras *do* overlap) | Keep as a module for completeness/future overlapping-camera use, but expect it disabled for a pure non-overlapping setup |
| `pose` | Skeleton keypoints, optional refinement cue | Lowest priority — add last, only if time permits |

Association score = weighted fusion of enabled modules, e.g.:

```
score = w_reid * reid_sim + w_color * color_sim + w_motion * motion_sim
score = -inf  if topology/temporal gating rules it out
```

## Stack

```
ultralytics          # YOLO detection
supervision          # ByteTrack + box/label annotation
torchreid            # OSNet ReID (pip install git+https://github.com/KaiyangZhou/deep-person-reid.git)
faiss-cpu            # vector similarity search
scipy                # hierarchical clustering (linkage + fcluster)
opencv-python         # video I/O
numpy
pyyaml
```

## Config file (single source of truth — no other code changes needed to toggle features)

```yaml
system:
  video_dir: "./data/videos"
  output_dir: "./outputs"
  display_resolution: [1280, 720]

cameras:
  auto_discover: true        # cam01.mp4 -> cam01, cam02.mp4 -> cam02, ...
  topology:                  # only used if features.topology.enabled
    adjacency:                # which cameras a person can plausibly walk between
      cam01: [cam02]
      cam02: [cam01, cam03]
      cam03: [cam02]
    max_transition_seconds: 30
    min_transition_seconds: 1

detector:
  model: "yolo11n.pt"
  device: "cuda"
  confidence: 0.35
  classes: [0]                # COCO person

tracker:
  type: "bytetrack"
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30

features:
  reid:
    enabled: true
    model: "osnet_x1_0"
    input_size: [256, 128]
    weight: 0.6
  color:
    enabled: true
    weight: 0.15
  motion:
    enabled: false
    weight: 0.1
  temporal:
    enabled: true
    weight: 0.15            # used as gating, not just scoring
  topology:
    enabled: true            # strongly recommended ON for non-overlapping cameras
  quality:
    enabled: true
    min_confidence: 0.5
    min_crop_pixels: [40, 90]
  geometry:
    enabled: false           # only relevant if cameras overlap
  pose:
    enabled: false

association:
  similarity_threshold: 0.75   # cosine sim -> distance = 1 - 0.75 = 0.25
  clustering_method: "average"  # scipy linkage method

output:
  video: "outputs/visualization/multi_camera.mp4"
  save_tracklets: true         # outputs/tracks/*.json, enables caching
```

## Caching

Cache detection + tracking output (`outputs/tracks/*.json`) so re-running
with a changed `features.*` or `association.*` config does **not** require
re-running YOLO/ByteTrack. This is a big token/time saver during iteration
— the agent should check for cached tracklets before recomputing.

## Pipeline steps

**1. Camera discovery** — auto-discover `.mp4` files in `video_dir`, derive
camera IDs from filenames unless `cameras.auto_discover: false`.

**2. Per-camera local tracking (independent, parallelizable later)**
For each camera video: run YOLO detection → ByteTrack → local `track_id`s.
Persist tracklets (frame ranges, boxes, crops or crop refs) to
`outputs/tracks/<camera_id>.json`.

**3. Feature extraction (only for enabled modules)**
For each local tracklet, extract and pool per-feature descriptors across
all its frames (mean-pool for ReID/color, entry/exit point + velocity for
motion). End state: one descriptor bundle per `(camera_id, local_track_id)`.

**4. Candidate generation**
Before scoring, apply hard gates: if `topology.enabled`, only compare
tracklet pairs from adjacent cameras within `[min_transition_seconds,
max_transition_seconds]`; if `temporal.enabled`, additionally reject
overlapping-time pairs when cameras are known non-overlapping.

**5. Association scoring**
Compute the weighted fusion score for every surviving candidate pair
(FAISS `IndexFlatIP` for the ReID cosine-similarity component is fine for
POC scale). Build the full similarity matrix, convert to distance
(`1 - similarity`), cluster with `scipy.cluster.hierarchy.linkage`
(`average`) + `fcluster` at `1 - similarity_threshold`.

**6. Global identity gallery**
Map `(camera_id, local_track_id) -> global_id` from the cluster assignment.

**7. Visualization**
Re-open all videos with `cv2.VideoCapture`, draw boxes/labels with
`sv.BoxAnnotator` / `sv.LabelAnnotator` using the mapped `global_id`
(same color per `global_id` across all cameras), stitch into a grid with
`np.hstack`/`np.vstack`, write with `cv2.VideoWriter`.

## Project structure

```
mtmc_tracker/
├── run.py
├── requirements.txt
├── configs/poc.yaml
├── scripts/download_poc_dataset.py
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── detection/detector.py
│   ├── tracking/single_camera_tracker.py
│   ├── tracklets/tracklet_manager.py
│   ├── features/
│   │   ├── base.py            # shared interface: extract(tracklet) -> descriptor
│   │   ├── manager.py         # loads only enabled modules
│   │   ├── reid.py
│   │   ├── color.py
│   │   ├── motion.py
│   │   ├── temporal.py
│   │   ├── topology.py
│   │   ├── quality.py
│   │   └── geometry.py
│   ├── association/
│   │   ├── candidate_generator.py
│   │   ├── scorer.py
│   │   └── gallery.py
│   └── visualization/renderer.py
└── outputs/
    ├── tracks/
    └── visualization/
```

## Dataset

Default to a small public non-overlapping-capable multi-camera pedestrian
set (e.g. a short PETS2009 subset or EPFL Terrace). Provide
`scripts/download_poc_dataset.py`, but don't hard-fail if the official
download requires manual registration — document the manual step and let
`data/videos/*.mp4` always work as a manual drop-in. The pipeline itself
must not depend on any dataset-specific annotation format.

## Build order for the coding agent (sequential, each step independently testable)

1. Config loader + `--dry-run` (prints resolved config)
2. Camera auto-discovery from `video_dir`
3. YOLO detection
4. ByteTrack local tracking
5. Tracklet persistence (`outputs/tracks/*.json`) + caching check
6. ReID module + tracklet-level pooling
7. **ReID-only** global association working end-to-end (must work before anything else is added)
8. Topology + temporal gating (highest priority add — this is what makes non-overlapping matching reliable)
9. Color feature
10. Quality filtering
11. Motion feature
12. Geometry (stub/disabled by default — only wire up if cameras overlap)
13. Pose (optional, last)
14. Visualization — per-camera annotated video
15. Combined grid video
16. End-to-end run via `python run.py --config configs/poc.yaml --videos ./data/videos`

## Success criterion

Given several non-overlapping camera videos, the system maintains a
consistent `global_id` for the same person as they walk from one camera's
view into another's, using ReID plus topology/temporal gating as the
primary signals, with color/motion/quality as tunable secondary signals —
all independently switchable via `configs/poc.yaml` with no code changes.
