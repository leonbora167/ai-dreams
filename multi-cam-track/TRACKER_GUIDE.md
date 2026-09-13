# How the tracker works

> **Looking for the ELI5 intuition guide or production architecture blueprint?**
> Check out [`INTUITION_GUIDE.md`](INTUITION_GUIDE.md) for a non-technical "explain like I'm 5" walkthrough and full engineer blueprint.

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

# How the Tracker Works (Operations & Architecture Guide)

> **Looking for the ELI5 intuition guide or production architecture blueprint?**  
> Check out [`INTUITION_GUIDE.md`](INTUITION_GUIDE.md) for a non-technical "explain like I'm 5" walkthrough and full engineer blueprint.

---

## 1. The Core Lifecycle

The multi-camera tracker operates in four distinct stages:

1. **Detection (Every Frame)**: YOLO11 identifies all human bounding boxes (`[x1, y1, x2, y2]`) on every video frame.
2. **Local Tracking (Within One Camera)**: ByteTrack associates frame-by-frame detections into stable continuous journeys, assigning a temporary `local_track_id` (e.g. `cam01: [L1]`).
3. **Appearance Signature Extraction (When Tracklet Ends)**: When a person exits or is occluded, representative crops are analyzed by OSNet to create a 512-dimensional ReID embedding (plus color and quality metrics).
4. **Global Identity Association (Across Cameras & Time)**: The descriptor is matched against the cumulative gallery to assign or propagate a permanent `global_id` (e.g. `G1`, `G2`).

---

## 2. The Three Execution Modes Compared

The codebase provides three distinct execution modes with fundamentally different scheduling, feature extraction timing, association mechanics, and ID progression:

### Side-by-Side Comparison Table

| Attribute | Mode 1: Offline Batch (`run.py`) | Mode 2: Incremental / Streaming (`run.py --incremental`) | Mode 3: Sequential Camera-by-Camera (`run_sequential.py`) |
| :--- | :--- | :--- | :--- |
| **CLI Command** | `python run.py --config <config>` | `python run.py --config <config> --incremental` | `python run_sequential.py --config <config>` |
| **Frame Ingestion** | Camera-by-camera (fully batches tracks to disk) | Interleaved frame-by-frame across all cameras concurrently (`c1_f1 → c2_f1 → c3_f1 → c1_f2 → ...`) | Camera-by-camera from start to finish (`cam01` completely, then `cam02`, etc.) |
| **Local Tracking** | ByteTrack per camera; tracklets cached to disk | Independent ByteTrack state per camera running in lockstep | ByteTrack instance fresh per camera video |
| **Feature Extraction** | Batch extracted from pooled crops and cached to `.npz` files | Extracted on-the-fly when a tracklet becomes inactive (timeout) | Extracted for finalized tracklets sorted by arrival order (`start_frame`) |
| **Association Timing** | **Post-hoc global clustering**: Agglomerative clustering / Hungarian matching across ALL tracklets from all cameras | **Online + Post-hoc**: Provisional matching as tracklets close + final global reconciliation pass (Union-Find) | **Online sequential**: Matches against cumulative gallery in arrival order as each camera finishes |
| **ID Assignment Flow** | Global IDs assigned by global clustering; Frame 0 across entire facility starts at `G1`, `G2`... | Provisional IDs minted live, reconciled after all streams finish | `cam01` starts at `G1 [L1]`, `G2 [L2]`... `cam02` keeps gallery, re-uses matched IDs (`G1`), mints next available (`G4`) |
| **Cross-Camera Memory** | Full global visibility across all past, present, and future camera videos | In-memory gallery of recently closed tracklets across all live streams | Cumulative gallery preserved and expanded across cameras (`cam01` → `cam02` → `cam03` → ...) |
| **Rendered Videos** | Single multi-camera grid video (`multi_camera.mp4`) | Single multi-camera grid video (`multi_camera.mp4`) | Individual videos per camera (`cam01_tracked.mp4`, etc.) + synchronized stitched grid (`stitched_review.mp4`) |
| **Best Used For** | Offline forensic review, benchmark evaluation, maximum global clustering accuracy | Real-time RTSP stream simulation, live edge-computing prototypes | Sequential multi-video inspection, camera-by-camera human review, continuous cross-video ReID testing |

---

### Detailed Flow Diagrams

#### Flow 1: Offline Batch Mode (`run.py`)
```text
[Camera 1 Video] ──> YOLO + ByteTrack ──> tracks/cam01.json + features/cam01_*.npz
[Camera 2 Video] ──> YOLO + ByteTrack ──> tracks/cam02.json + features/cam02_*.npz
[Camera 3 Video] ──> YOLO + ByteTrack ──> tracks/cam03.json + features/cam03_*.npz
                           │
                           ▼
          [Global Agglomerative Clustering]
   (Analyzes all tracklets across all cameras at once)
                           │
                           ▼
       Global IDs assigned: G1, G2, G3 across facility
                           │
                           ▼
          Final multi-camera review video rendered
```
* **Key characteristic**: The algorithm has "god-view" access to the future and past of all cameras simultaneously. It clusters all tracklet embeddings globally.

---

#### Flow 2: Incremental / Streaming Mode (`run.py --incremental`)
```text
Parallel Video Streams / Live RTSP
  │
  ├── Frame t:   Cam 1 ──> YOLO + ByteTrack (Cam 1)
  ├── Frame t:   Cam 2 ──> YOLO + ByteTrack (Cam 2)
  ├── Frame t:   Cam 3 ──> YOLO + ByteTrack (Cam 3)
  │
  └── Inactivity Timeout (30 frames without detection)
        │
        ├── Extract OSNet ReID embedding from crops
        ├── Match against active in-memory gallery
        └── Assign provisional Global ID (G1, G2...)
  │
  └── Streams Conclude
        │
        ▼
   [Final Global Reconciliation Pass (Union-Find)]
   (Resolves co-location conflicts and merges split tracks)
        │
        ▼
   Corrected IDs saved to disk + multi-camera video rendered
```
* **Key characteristic**: Simulates true real-time surveillance. Memory is kept in RAM during execution, with minimal latency per frame.

---

#### Flow 3: Sequential Camera-by-Camera Mode (`run_sequential.py`)
```text
[Video 1: cam01]
  ├── YOLO + ByteTrack run start to finish
  ├── Tracklets ordered by arrival time (start_frame)
  ├── First person gets G1 [L1], second gets G2 [L2], third gets G3 [L3]
  ├── Cumulative Gallery initialized in RAM with Cam 1 features
  └── Render cam01_tracked.mp4 immediately
          │
          ▼
[Video 2: cam02]  <── (Carries forward Cumulative Gallery from Cam 1!)
  ├── YOLO + ByteTrack run start to finish
  ├── Tracklets matched against Cumulative Gallery:
  │     ├── Person A matches Cam 1 G1  ──> Assigned G1
  │     ├── Person B matches Cam 1 G2  ──> Assigned G2
  │     └── Person D (brand-new)       ──> Assigned G4 (next available ID)
  ├── Cumulative Gallery updated with Cam 2 features
  └── Render cam02_tracked.mp4 immediately
          │
          ▼
[Subsequent Videos: cam03, cam04...]
  └── Continue accumulating gallery features and minting sequential IDs
          │
          ▼
[Stitching Stage]
  └── Stitches all individual videos into synchronized stitched_review.mp4
```
* **Key characteristic**: Pure sequential progression without any retroactive renumbering. What happens in Video 1 remains fixed as `G1, G2...`, and subsequent cameras continuously learn and accumulate from previous cameras.

---

### Concrete Scenario: How the Same People are Handled in Each Mode

Suppose we have two cameras (`cam01` and `cam02`):
- In `cam01`, **Alice** walks from $t=0$ to $10\text{s}$, and **Bob** walks from $t=5$ to $15\text{s}$.
- In `cam02`, **Alice** re-appears from $t=15$ to $25\text{s}$, and a new person **David** walks from $t=20$ to $30\text{s}$.

| Mode | What happens in Camera 1? | What happens in Camera 2? | Final IDs Assigned |
| :--- | :--- | :--- | :--- |
| **Offline Mode** | Tracks Alice and Bob. | Tracks Alice and David. | Global clustering sees all embeddings: Alice $\to$ **G1** in both cams, Bob $\to$ **G2** in cam01, David $\to$ **G3** in cam02. |
| **Incremental Mode** | Frames interleave. Alice is tracked live in cam01. At $t=10\text{s}$, Alice exits cam01 and her ReID embedding enters the active gallery. | At $t=15\text{s}$, Alice enters cam02. When she exits at $t=25\text{s}$, her embedding matches the gallery and links to **G1**. David gets **G3**. | Alice $\to$ **G1** (both cams), Bob $\to$ **G2**, David $\to$ **G3**. |
| **Sequential Mode** | Cam 1 runs completely first. Alice arrives at $t=0$ $\to$ **G1 [L1]**. Bob arrives at $t=5$ $\to$ **G2 [L2]**. Video renders with G1 and G2. | Cam 2 starts with Cam 1 gallery intact. Alice matches Cam 1 G1 $\to$ **G1 [L1]**. David is new $\to$ **G3 [L2]**. Video renders with G1 and G3. | Alice $\to$ **G1** (both cams), Bob $\to$ **G2**, David $\to$ **G3**. |

---


## 3. Tracker Memory Rules & Timeouts

The tracker enforces strict temporal and physical rules to maintain tracking consistency:

### 1. Local Track Timeout (`tracker.track_buffer: 30`)
- While tracking inside a camera, ByteTrack maintains a Kalman filter state for every active person.
- If a person is temporarily occluded or missed by the detector, ByteTrack holds their local track alive for up to **30 frames** (`track_buffer`).
- If 30 frames elapse without detection, the local tracklet is finalized, and its crops are passed to the ReID feature extractor.

### 2. Gallery Temporal Validity Window (`cameras.topology.max_transition_seconds: 60`)
- When matching a tracklet against the gallery, the system computes the time gap:
  $$\Delta t = \max(0.0,\, t_{\text{start}} - t_{\text{last\_seen}})$$
- If $\Delta t \le \text{max\_transition\_seconds}$ (e.g. 60 seconds), the person is eligible for re-identification.
- If a person has disappeared for longer than this threshold, the gallery treats them as expired, preventing false positive matches with new people arriving minutes later.

### 3. Intra-Camera Co-Location Constraint (Overlap Filter)
- A single person **cannot physically be in two different places at the exact same millisecond in the same camera**.
- If tracklet $A$ and tracklet $B$ co-occur at overlapping frame timestamps in the same camera, they can never receive the same global ID.

### 4. Intra-Camera Re-Identification
- If a person walks behind a large pillar or leaves the camera view for 5 seconds and returns, their new tracklet can match their earlier tracklet from the **same camera** via ReID cosine similarity, re-linking them to their original identity.

---

## 4. Evaluation & Benchmarking

The evaluation suite (`scripts/evaluate.py`) computes two categories of metrics:

### 1. Ground Truth Benchmark Metrics (When Annotations Exist)
Matches tracker predicted bounding boxes against ground truth annotations at $\text{IoU} \ge 0.5$ using Hungarian matching:
- **MOTA (Multi-Object Tracking Accuracy)**: Measures overall tracking fidelity combining False Positives (FP), False Negatives (FN), and Identity Switches (IDSW):
  $$\text{MOTA} = 1 - \frac{\text{FN} + \text{FP} + \text{IDSW}}{\text{Total Ground Truth}}$$
- **MOTP (Multi-Object Tracking Precision)**: Average bounding box overlap ($\text{IoU}$) on correctly matched detections.
- **IDF1 (Identification F1-Score)**: Ratio of correctly identified detections over the average number of ground truth and tracker detections. Measures trajectory-level identity preservation.
- **IDSW (Identity Switches)**: Number of times a tracked trajectory flips to an incorrect ID.

### 2. Multi-Camera Structural Metrics
Assesses physical consistency and cross-camera performance across the network:
- **Overlap Conflicts**: Must be **0**. Verifies that no identity appears in two places at the same time within any camera view.
- **Mean / Median Tracklet Length**: Measures continuous tracking stability per camera before occlusion.
- **Cross-Camera Coverage**: Counts how many identities are successfully verified across $\ge 2$, $\ge 3$, or all $N$ cameras.

### Consolidating Results into Summaries
```bash
# Offline benchmark summary:
python scripts/consolidate_benchmarks.py  # Outputs benchmark_evaluation_summary.csv

# Sequential benchmark summary:
python scripts/consolidate_sequential_benchmarks.py  # Outputs benchmark_sequential_evaluation_summary.csv
```

---

## 5. Saved Data Formats

Each run produces standard output artifacts in its designated `output_dir`:

1. `observations/{camera}.jsonl`: Line-by-line JSON record for every detection:
   ```json
   {"camera_id": "cam01", "frame_id": 60, "timestamp_sec": 2.4, "local_track_id": 1, "bbox_xyxy": [120.5, 80.0, 310.2, 540.8], "confidence": 0.892, "global_id": 1}
   ```
2. `tracks/{camera}.json`: Complete tracklet histories grouped per camera (frame IDs, bounding boxes, confidences, local IDs, and global IDs).
3. `features/{camera}_{local_id}.npz`: NumPy archive containing:
   - `reid`: 512-dimensional float32 L2-normalized feature vector.
   - `color`: 24-bin HSV color histogram.
   - `camera_id`, `local_track_id`, `global_id`: Metadata tags.
4. `global_id_map.json`: Quick-lookup mapping `"{camera}:{local_track_id}" -> global_id`.
5. `visualization/`:
   - `{camera}_tracked.mp4`: Individual annotated video for each camera.
   - `stitched_review.mp4`: Multi-camera synchronized grid video.

