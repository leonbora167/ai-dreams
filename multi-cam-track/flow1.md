# MTMC tracker: engineering flow and handoff

This document is a self-contained description of the repository as it exists
today. It is intended for a developer or AI agent taking over the project. It
explains the goal, the code paths, the saved data, the current behavior, known
limitations, and the recommended next changes. It does not require reading any
other Markdown file first.

## 1. What this project is trying to do

The project is a local Python proof of concept for multi-target,
multi-camera (MTMC) people tracking.

Its job is to answer two different questions:

1. **Local tracking:** within one camera, which detections belong to the same
   person over time?
2. **Global association:** across different cameras, which local tracks belong
   to the same real-world person?

The program assigns two types of identifier:

- `local_track_id`: an ID created independently by ByteTrack inside one camera.
  `cam01:12` and `cam02:12` have no implied relationship.
- `global_id`: the cross-camera ID shown as `G1`, `G2`, and so on. The intended
  result is that the same person receives the same global ID in all cameras.

The code has two execution modes:

- **Offline mode** is the default. It completes local tracking on each video,
  then extracts features and performs association.
- **Incremental mode** (`--incremental`) treats a folder of MP4 files like a
  set of live streams. It interleaves frames from every camera, maintains
  camera-local tracking state, makes provisional global assignments when a
  tracklet closes, then performs a final global reconciliation after all input
  videos end.

The current EPFL example uses synchronized, overlapping camera views. The
intended real deployment is non-overlapping cameras or RTSP streams, where
topology and travel-time constraints matter much more.

## 2. Main processing concepts

### Detection

YOLO detects only COCO class `0`, which is `person`. Every detection contains
a bounding box in `[x1, y1, x2, y2]` form and a confidence value.

### Local tracker and tracklet

`supervision.ByteTrack` maintains a separate state machine for every camera in
incremental mode. A local tracklet is the history of one tracker ID:

```text
camera_id
local_track_id
frame numbers
bounding boxes
detection confidences
start/end timestamps
video path
global_id (once association completes)
```

A local tracklet is much more useful than a single detection crop: it contains
multiple views of the person and a time interval.

### ReID embedding

Re-identification (ReID) converts a person crop into a floating-point feature
vector. Similar-looking crops should have nearby vectors. The code pools a
small reservoir sample of crops across a tracklet into one normalized vector.

The default EPFL configuration uses an OSNet-AIN x1.0 checkpoint trained on
MSMT17. The expected ReID vector is generally 512 dimensions. The stored vector
is suitable for a later vector database, but a vector database is not part of
this POC.

### Association

The active association score is a weighted cosine similarity:

```text
score = 0.9 * cosine(ReID) + 0.1 * cosine(color)
```

The `epfl_4p.yaml` configuration uses `0.70` as the current minimum score.
This value is a starting point, not a universally correct threshold.

For the final pass, eligible cross-camera edges are sorted from highest score
to lowest. The algorithm joins clusters only when doing so does not put two
overlapping tracklets from the **same camera** into one global identity. This
constraint prevents two people visible at the same time in one camera from
sharing a global ID.

For non-overlapping cameras, topology and temporal gates should be enabled.
They remove impossible camera pairs and impossible travel times before ReID
comparison. For the overlapping EPFL demo they are disabled.

## 3. End-to-end incremental flow

The main command is:

```bash
python run.py --config configs/epfl_4p.yaml --videos ./data/videos --incremental
```

The logical flow is:

```text
Folder of camera MP4 files
        |
        v
One next frame from cam01, cam02, ..., camNN (round-robin)
        |
        v
One shared YOLO detector
        |
        v
One persistent ByteTrack instance per camera
        |
        +--> per-frame observations written as JSONL
        |
        v
Active local tracklets; representative person crops retained in memory
        |
        v
Tracklet becomes inactive or stream ends
        |
        +--> pooled ReID/color/motion descriptor saved as NPZ
        +--> provisional global gallery assignment
        |
        v
All streams complete
        |
        v
Batch global reconciliation over all completed tracklets
        |
        +--> final global IDs written into JSON/JSONL/NPZ records
        |
        v
Stitched review MP4 rendered from saved tracklets and original videos
```

The round-robin arrangement is deliberately not "finish camera 1, then start
camera 2." It simulates synchronized stream progression while avoiding one
separate YOLO model copy per camera. It is currently one Python process and
does not use separate inference threads.

## 4. Repository file map

| File | Purpose | Important implementation notes |
|---|---|---|
| `run.py` | Command-line entry point. | Loads config, discovers videos, prints progress, and selects offline or `--incremental` mode. `--dry-run` only resolves config and videos; it does not load models. |
| `src/config.py` | Configuration defaults and YAML loader. | Deep-merges user YAML into `DEFAULTS`; discovers `*.mp4` files by filename stem. |
| `src/tracklets.py` | Tracklet data model and JSON persistence. | Defines `Tracklet`; JSON stores lists of frames, boxes, confidences, timestamps via FPS, and `global_id`. |
| `src/tracking.py` | Offline local tracking implementation. | Uses Ultralytics `YOLO.track()` over a full video and writes one per-camera cache JSON. This is not the preferred path for incremental MP4/RTSP-like operation. |
| `src/pipeline.py` | Offline orchestration. | Reuses cached local tracklets when available, then extracts features, associates them, and renders a video. |
| `src/streaming_pipeline.py` | Incremental orchestration. | Main current pipeline: shared YOLO detector, one explicit `supervision.ByteTrack` object per camera, active tracklets, feature persistence, provisional gallery, final batch reconciliation, and rendering. |
| `src/features.py` | Feature extraction. | Lazily loads only enabled modules. Supports ReID, HSV color histogram, simple motion, and confidence quality filtering. ReID crops are RGB/ImageNet normalized before inference. |
| `src/association.py` | Cross-camera matching logic. | Contains old offline `associate()` and active final `batch_associate()`. The latter uses descending-score, constrained union grouping. |
| `src/visualization.py` | Review-video renderer. | Reads original videos, finds stored boxes per frame, draws `G<global_id>`, and uses a near-square tile grid: 4 cameras become 2x2, 9 become 3x3. |
| `scripts/download_poc_dataset.py` | Optional EPFL source-video downloader. | Downloads four official AVI files and converts them to `cam01.mp4` through `cam04.mp4`. Operator-run only. |
| `scripts/download_reid_weights.py` | ReID checkpoint downloader. | Uses `gdown` to retrieve the configured official OSNet-AIN MSMT17 checkpoint into `models/`. Operator-run only. |
| `scripts/visualize_outputs.py` | Standalone renderer. | Reads saved `tracks/*.json` and global IDs, then produces the stitched MP4 without detector/ReID inference. Adds repository root to `sys.path` when run directly. |
| `scripts/diagnose_reid.py` | Read-only ReID analysis. | Reads saved NPZ vectors and prints cross-camera cosine-similarity percentiles and highest-scoring pairs. It does not run inference. |
| `configs/poc.yaml` | Generic non-overlapping-camera POC config. | Topology and temporal gates are enabled; use after editing camera adjacency and travel-time settings. |
| `configs/epfl_4p.yaml` | EPFL overlapping-camera demo config. | Uses OSNet-AIN checkpoint, disables topology/temporal gates, and writes to `outputs/epfl_4p_v2/`. |
| `requirements.txt` | Python dependency list. | Includes YOLO, Supervision, OpenCV, Torchreid dependencies, SciPy, and `gdown`. PyTorch itself is intentionally environment-managed. |
| `install.bat` | Historical install helper. | Despite its filename, its content is Bash-style. Prefer the existing Conda environment and install missing packages explicitly. |
| `yolo11n.pt` | YOLO detector weights. | Local model asset, used by default configuration. `yolo11s.pt` is a potential higher-accuracy alternative. |
| `models/` | Created by the ReID download script. | Holds ReID checkpoints. It may not exist until the operator runs the script. |
| `data/videos/` | Input directory. | Not committed by design. Files should be named `cam01.mp4`, `cam02.mp4`, and so on. |
| `outputs/` | Generated results. | Not source code. See the output schema below. |

## 5. Configuration reference

### System

```yaml
system:
  video_dir: ./data/videos
  output_dir: ./outputs/epfl_4p_v2
  display_resolution: [1280, 720]
```

`output_dir` separates experiment runs. Change it when comparing tracker or
association settings; do not overwrite results that need to be inspected.

### Cameras and topology

```yaml
cameras:
  auto_discover: true
  topology:
    adjacency: {cam01: [cam02]}
    min_transition_seconds: 1
    max_transition_seconds: 30
```

For non-overlapping cameras, `adjacency` should include only physically
possible transitions. For example, if a person can move from cam01 to cam02
but cannot reach cam04 directly, do not include cam04 under cam01. This is one
of the most valuable safeguards against lookalike false matches.

### Detector

```yaml
detector:
  model: yolo11n.pt
  device: cuda
  confidence: 0.35
  classes: [0]
```

`yolo11n.pt` is fast but small. `yolo11s.pt` or a larger detector can improve
small/occluded person detection at higher GPU cost.

### Tracker

```yaml
tracker:
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
```

In the incremental implementation, `track_high_thresh` and `track_buffer` are
passed to Supervision ByteTrack. `track_low_thresh` and `new_track_thresh` are
present for config compatibility but are not yet separately mapped to the
Supervision API. `track_buffer` is frames, not seconds; 30 equals about 1.2
seconds at 25 FPS.

### Features

```yaml
features:
  reid:
    enabled: true
    model: osnet_ain_x1_0
    weight_path: ./models/osnet_ain_x1_0_msmt17_256x128.pth
    input_size: [256, 128]
    weight: 0.9
  color: {enabled: true, weight: 0.1}
  motion: {enabled: false}
  temporal: {enabled: false}
  topology: {enabled: false}
  quality: {enabled: true, min_confidence: 0.5}
  geometry: {enabled: false}
  pose: {enabled: false}
```

The feature manager only imports/initializes the ReID model when ReID is
enabled. The `weight_path` must exist when configured; the code raises a clear
error otherwise.

`geometry` and `pose` are configuration placeholders, not implemented feature
extractors. Do not enable them expecting working geometry or pose association.

### Association

```yaml
association:
  similarity_threshold: 0.70
  clustering_method: average
```

`clustering_method` is retained from the original specification, but the active
incremental final pass does not currently call SciPy hierarchical clustering.
It uses constrained score-ordered grouping instead.

## 6. Generated output schema

For `output_dir: outputs/epfl_4p_v2`, the incremental run produces:

```text
outputs/epfl_4p_v2/
  observations/
    cam01.jsonl
    cam02.jsonl
    ...
  tracks/
    cam01.json
    cam02.json
    ...
  features/
    cam01_1.npz
    cam01_2.npz
    ...
  global_id_map.json
  visualization/
    multi_camera.mp4
```

### `observations/camXX.jsonl`

JSON Lines means every line is one standalone JSON object. It is intended for
streaming/log ingestion and is human-readable.

```json
{
  "camera_id": "cam01",
  "frame_id": 1250,
  "timestamp_sec": 50.0,
  "local_track_id": 12,
  "bbox_xyxy": [100.2, 80.1, 210.4, 350.8],
  "confidence": 0.91,
  "global_id": 4
}
```

During processing `global_id` starts as `null`; the file is backfilled with the
final ID after batch reconciliation completes. It is not a true real-time event
log with immutable final IDs yet.

### `tracks/camXX.json`

This is an array of finalized tracklets. A representative entry is:

```json
{
  "camera_id": "cam01",
  "local_id": 12,
  "frames": [1220, 1221, 1222],
  "boxes": [[100.2, 80.1, 210.4, 350.8]],
  "confidences": [0.91],
  "fps": 25.0,
  "video_path": "data/videos/cam01.mp4",
  "global_id": 4
}
```

The `frames`, `boxes`, and `confidences` arrays are aligned by index.

### `features/camXX_LOCALID.npz`

This is a compressed NumPy archive for one local tracklet. Keys normally are:

| Key | Type | Meaning |
|---|---|---|
| `reid` | float32 vector, normally 512-D | Pooled OSNet appearance descriptor. |
| `color` | float32 vector | Pooled HSV histogram; present when enabled. |
| `motion` | float32 vector | Start-to-end image trajectory; present when enabled. |
| `quality` | scalar float | Mean detector confidence. |
| `camera_id` | scalar string array | Source camera. |
| `local_track_id` | scalar integer array | ByteTrack local ID. |
| `global_id` | scalar integer array | Final reconciled identity ID. |

For a future vector database, store the normalized `reid` vector as the vector
field and the metadata as filterable fields. Keep `camera_id`, time range,
local ID, global ID, model name/version, and feature quality. Do not assume a
global ID is immutable until a production-grade reconciliation policy exists.

### `global_id_map.json`

Maps a compound local identity to its final global identity:

```json
{
  "cam01:12": 4,
  "cam03:5": 4
}
```

### `visualization/multi_camera.mp4`

This is a review artifact, not source-of-truth tracking data. It is regenerated
from the input videos and saved tracklets. Camera labels are drawn in each tile,
boxes are thin green outlines, and the global label is drawn as `G<number>`.

## 7. Current observed behavior and drawbacks

### The tracker has improved, but it does not yet produce the known four IDs

On the EPFL 4-person sequence, the latest output visibly keeps several people
consistent across cameras, but it still splits some real people into multiple
global IDs such as `G5`, `G7`, and so on. It should not be treated as an
accurate production tracker or as a validated benchmark result.

### Local track fragmentation remains

The explicit per-camera ByteTrack implementation is much better than the prior
per-frame `YOLO.track()` attempt, but it can still create multiple local
tracklets when people are occluded, blurred, partially out of frame, or missed
by the detector. ReID then has to reconnect fragments.

### ReID similarity alone is ambiguous

In this sequence, some different people produce extremely high cosine
similarities. Clothing is similar, views are low resolution, people are often
seen from the back, and the scene differs from the public ReID training domain.
ReID provides useful evidence but should not be the only decision signal.

### The current association is graph-based, not a full MTMC optimizer

The score-ordered union algorithm is practical but greedy. An early high-score
edge can affect later choices. It does not optimize a globally consistent
assignment over all cameras and time. It also has no explicit model of one
person's continuity across fragmented local tracks beyond the overlap guard.

### Geometry is not implemented

The EPFL sample has overlapping, calibrated camera views. A ground-plane
homography would provide a powerful check: detections projected from different
cameras should agree in world position at the same timestamp. The current
implementation ignores that information even though it is ideally suited to
this dataset.

### The MP4 incremental mode is a live simulation, not actual RTSP support

`cv2.VideoCapture` reads files and knows their length. A real RTSP service needs
connection/reconnection logic, queues, bounded latency, clock alignment,
dropped-frame handling, and an output policy for identities that may change
after delayed association.

### Global IDs are final only after the stream ends

The pipeline makes provisional assignments while running, but backfills the
saved observations after final reconciliation. A production monitoring system
needs versioned identity events or a stable online gallery policy instead of
rewriting historical data.

### Performance limitations

The current design runs one detector inference at a time in one Python process.
This avoids loading one detector per camera, but it does not exploit batch
inference. ReID features are extracted only at tracklet close; this is good for
cost but delays association. The renderer performs simple list lookups per
frame and is intended for small POC videos, not long recordings.

### No formal evaluation is implemented

There is no MOT, IDF1, HOTA, mAP, precision/recall, or MTMC metric computation.
Visual inspection is useful but insufficient for threshold/model decisions.

## 8. Recommended improvement roadmap

### Priority 1: Add geometry for overlapping cameras

For EPFL, download/use the provided calibration or homographies and implement:

1. Foot-point extraction from each person box.
2. Projection of camera foot points to a shared ground plane.
3. Same-time cross-camera gating based on projected distance.
4. A fused score combining geometry and ReID.
5. A one-to-one bipartite assignment at each synchronized time step.

This will likely improve this EPFL demo more than another ReID threshold change.
It is not applicable to truly non-overlapping cameras.

### Priority 2: Improve tracklet stitching within a camera

Before cross-camera association, reconnect non-overlapping local fragments when
all of the following agree:

- short time gap;
- plausible last-to-first image position and velocity;
- high appearance similarity;
- no conflicting simultaneous track in that camera.

This turns 40-70 local fragments per EPFL camera toward the expected 4 long
local trajectories and simplifies global association.

### Priority 3: Replace greedy grouping with constrained optimization

Build a graph whose nodes are tracklets and whose edges contain ReID, color,
time, topology, motion, and geometry evidence. Solve a constrained assignment
or min-cost-flow problem with rules such as:

- no simultaneous same-camera tracks in one identity;
- one candidate per camera/time slot;
- known topology/travel-time gates;
- penalties for abrupt identity switches;
- delayed confirmation for uncertain assignments.

For offline video, hierarchical clustering or correlation clustering with
cannot-link constraints is suitable. For online RTSP, maintain a gallery of
identity prototypes plus a limited revision window.

### Priority 4: Improve ReID quality and train for the domain

- Use only sharp, sufficiently large, non-occluded crops for gallery updates.
- Weight each crop by detection confidence, size, sharpness, and pose/view.
- Keep multiple prototypes per global identity, such as front, back, and side
  view, instead of one mean vector.
- Fine-tune a ReID model using samples from the target cameras when permitted.
- Record model version and embedding dimension with every vector-db record.

### Priority 5: Tune detector and ByteTrack using measurement

Try `yolo11s.pt` and compare missed detections and ID switches against
`yolo11n.pt`. Test `track_buffer` values around 60-90 for 25 FPS video. Tune
detection confidence using measured false-positive and missed-person rates,
not only visual preference. Add diagnostics that report local tracklet count,
mean duration, short-track rate, and global cluster count after every run.

### Priority 6: Add proper evaluation and regression fixtures

Use EPFL labels/calibration where appropriate and establish a small benchmark:

- detection precision/recall;
- single-camera IDF1/HOTA;
- cross-camera pair precision/recall;
- final global identity count versus known identity count;
- visual regression clips for occlusion and crossing events.

Keep expected metrics for a short fixed video segment so association changes can
be accepted or rejected with evidence.

### Priority 7: Productionize the streaming architecture

For actual RTSP sources, use one capture worker per camera and a bounded queue.
Run detector inference in a batcher process or GPU worker. Keep one ByteTrack
state machine per camera. Send finalized tracklets to an association service
that emits versioned identity updates. Persist immutable detection/tracklet
events and treat later global-ID corrections as new events rather than rewriting
the past.

## 9. Operator commands

All commands should be run from the repository root in `tracker-env`.

```bash
conda activate tracker-env
```

Download the demo videos:

```bash
python scripts/download_poc_dataset.py
```

Download the configured ReID checkpoint:

```bash
pip install gdown
python scripts/download_reid_weights.py
```

Check paths/config without model or video processing:

```bash
python run.py --config configs/epfl_4p.yaml --videos ./data/videos --dry-run
```

Run incremental MP4-as-stream processing:

```bash
python run.py --config configs/epfl_4p.yaml --videos ./data/videos --incremental
```

Render from saved tracking data without detector/ReID inference:

```bash
python scripts/visualize_outputs.py --config configs/epfl_4p.yaml --videos ./data/videos
```

Inspect stored ReID score distribution without detector/ReID inference:

```bash
python scripts/diagnose_reid.py --output-dir outputs/epfl_4p_v2
```

## 10. Important maintenance rules for future agents

1. Do not silently download models or datasets during a normal tracker run.
   Keep downloads as explicit scripts.
2. Preserve output directories from experiments. Use a new `output_dir` for a
   materially different model, threshold, or association algorithm.
3. Do not claim global IDs are ground truth without evaluation.
4. Keep detector/local tracking and cross-camera association as separate
   modules. A local tracker ID is not a global ID.
5. Ensure disabled feature modules do not load models or perform work.
6. For non-overlapping cameras, prioritize topology and temporal constraints.
   For overlapping calibrated cameras, prioritize geometry.
7. Treat `.npz` embeddings as model-versioned data. Changing ReID model or crop
   preprocessing requires regenerating embeddings and association outputs.
