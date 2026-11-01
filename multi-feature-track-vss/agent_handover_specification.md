# AGENT HANDOVER: Single-Camera Auditorium Tracker & Video Semantic Search (VSS) with Local VLM QA

## 1. Project Overview & Scope

Build a lightweight, modular computer vision and visual retrieval pipeline in Python for a **single video feed** (such as an auditorium CCTV, lecture hall, or hallway camera). Multi-camera tracking, spatial handoffs, and multi-feed ingestion are strictly out of scope for this proof-of-concept (POC).

The system executes the following sequence:
1. **Person Detection & Short-Term Tracking:** Detects persons on every frame and maintains tracklet identities in RAM using motion tracking.
2. **24-Frame Stability Gate & Best Crop Selection:** Tracks are kept in an in-memory tentative buffer. When a tracklet reaches **24 consecutive frames** of visibility, the pipeline computes a sharpness and quality score to select the single best full-body image crop.
3. **Local Multi-Modal Embedding (Run Once per Track):**
   * **SigLIP Multimodal Vector:** Extracted from the best crop using a local Hugging Face model (`google/siglip-base-patch16-224`) to enable natural language semantic search (*"person wearing white shirt"*).
   * **OSNet Body Re-ID Vector:** Extracted for fine-grained full-body appearance re-identification.
   * **ArcFace Face Vector:** Extracted opportunistically using InsightFace if a clear, frontal face is detected in the crop.
4. **Local Vector Database Storage:** Stores tracklet records, vectors, and temporal timestamps inside an embedded, serverless **LanceDB** database.
5. **Post-Execution VLM/LLM Q&A Agent:** A standalone script (`vlm_qa.py`) running a local, small vision-language model (`Qwen/Qwen2-VL-2B-Instruct` or `HuggingFaceTB/SmolVLM-Instruct`) to search the vector database and answer natural language questions about people and events in the video.
6. **Automated Sample Video Download:** Includes a script to automatically download sample auditorium/pedestrian footage for immediate testing.

---

## 2. Core Architecture Rules for Coding Agent

* **100% Local Inference:** No external third-party APIs (OpenAI, Gemini API, etc.). All models run locally via PyTorch, ONNX, and Hugging Face `transformers`.
* **Flat File Structure:** Do NOT create nested package structures, `src/` directories, or `__init__.py` boilerplate files. Keep all modules in the root folder.
* **Low-Code First:** Use established, stable libraries:
  * Detector: `ultralytics` (`yolov8n.pt`)
  * Tracker: `supervision` (`ByteTrack`)
  * Re-ID: `torchreid` (`osnet_x0_5`)
  * Multimodal: `transformers` (`google/siglip-base-patch16-224`)
  * Face Module: `insightface` (`buffalo_s` or `buffalo_l`)
  * Vector Store: `lancedb` (embedded file-backed, zero setup)
  * Local VLM: `transformers` (`Qwen/Qwen2-VL-2B-Instruct`)
* **Endpoint-Ready Separation:** Encapsulate every component in a clean Python class with clear input/output types so that local models can be replaced with remote microservice endpoints later without rewriting pipeline logic.
* **No Heavy Embedding Per Frame:** The tracker runs in RAM using motion association. High-dimensional models (SigLIP, OSNet, ArcFace) must **only** be invoked when a tracklet is confirmed at frame 24.

---

## 3. Repository File Structure

```text
auditorium_vss/
├── requirements.txt      # Pinned Python package dependencies
├── download_sample.py    # Fetches sample test video to data/raw_videos/sample.mp4
├── detector.py           # YOLO person detector returning bboxes and crops
├── tracker.py            # ByteTrack wrapper + 24-frame RAM buffer & crop selector
├── embedder.py           # Local OSNet & SigLIP feature extractors
├── face_module.py        # InsightFace detector & ArcFace vector extractor
├── vector_store.py       # Embedded LanceDB schema, insertion, and vector search
├── pipeline.py           # Master execution script (Detect -> Track -> Index)
├── search_engine.py      # Standalone CLI visual retrieval engine (Text, Body, Face)
├── vlm_qa.py             # Post-execution local VLM Question-Answering script
└── data/                 # Auto-generated directory for assets
    ├── raw_videos/       # Input video files
    ├── crops/            # Saved high-quality full-body crops and face crops
    └── lancedb/          # Local LanceDB table files
```

---

## 4. Module Specifications & Interfaces

### 4.1. `download_sample.py`
* **Task:** Downloads a royalty-free pedestrian or auditorium video (MP4) to `./data/raw_videos/sample.mp4` if not already present.
* **Requirements:** Use standard Python `urllib.request` with a fallback direct download link so tests can run immediately without manual user file placement.

### 4.2. `detector.py`
* **Class:** `PersonDetector`
* **Implementation Details:**
  * Loads `yolov8n.pt` using `ultralytics.YOLO`.
  * Filters detections strictly for `class_id == 0` (person) with confidence threshold $\ge 0.40$.
* **Contract:**
  ```python
  def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
      """
      Runs person detection on a single BGR video frame.
      Returns:
          bboxes: np.ndarray of shape (N, 5) with [x1, y1, x2, y2, confidence]
          crops: list of np.ndarray image crops corresponding to each detected person
      """
  ```

### 4.3. `tracker.py`
* **Class:** `AuditoriumTracker`
* **Implementation Details:**
  * Wraps `supervision.ByteTrack` for real-time frame-to-frame association in system memory (RAM).
  * Maintains an in-memory dictionary buffer: `self.active_buffers = {track_id: list_of_frame_entries}`.
  * **24-Frame Gating Logic:**
    1. For frames 1 through 23, save bounding box coordinates, frame number, timestamp, and crop to memory.
    2. At frame 24, calculate image sharpness using Laplacian variance:
       $$\text{Sharpness} = \text{Var}(\nabla^2 I_{\text{gray}})$$
       Score each crop: $\text{Quality} = \text{Sharpness} \times \text{Confidence} \times \sqrt{\text{Crop Area}}$.
    3. Select the highest-scoring crop as `best_crop`.
    4. Mark tracklet as `indexed = True` and emit the tracklet for database insertion.
    5. For frame 25 onward, continue tracking in RAM; log trajectory without re-extracting heavy embeddings.
* **Contract:**
  ```python
  def update(self, detections: np.ndarray, frame: np.ndarray, frame_id: int, timestamp: float) -> tuple[list[dict], list[dict]]:
      """
      Returns:
          active_tracks: list of dicts [{'track_id': int, 'bbox': [x1, y1, x2, y2]}]
          newly_confirmed_tracks: list of dicts [{'track_id': int, 'best_crop': np.ndarray, 'start_frame': int, 'start_sec': float, 'trajectory': list}]
      """
  ```

### 4.4. `embedder.py`
* **Class:** `LocalEmbedder`
* **Implementation Details:**
  * **OSNet:** Loads `osnet_x0_5` via `torchreid.models.build_model`. Outputs a 512-dimensional L2-normalized vector.
  * **SigLIP:** Loads `google/siglip-base-patch16-224` via Hugging Face `transformers.AutoModel` and `AutoProcessor`. Outputs a 768-dimensional L2-normalized vector for both text and image queries.
* **Contract:**
  ```python
  def extract_body_reid(self, crop: np.ndarray) -> list[float]:
      """Returns 512-d normalized OSNet vector."""
  def extract_multimodal_image(self, crop: np.ndarray) -> list[float]:
      """Returns 768-d normalized SigLIP image vector."""
  def extract_multimodal_text(self, text: str) -> list[float]:
      """Returns 768-d normalized SigLIP text vector."""
  ```

### 4.5. `face_module.py`
* **Class:** `FaceModule`
* **Implementation Details:**
  * Initializes InsightFace `FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider' or 'CUDAExecutionProvider'])`.
  * Analyzes the upper 40% of the body crop.
  * If a face is found with detection confidence $\ge 0.60$, extracts its 512-d ArcFace vector and crops the face.
  * If no face is detected, returns `(None, None)`.
* **Contract:**
  ```python
  def extract_face(self, crop: np.ndarray) -> tuple[list[float] | None, np.ndarray | None]:
      """Returns (face_vector_512d, face_crop_image) or (None, None)."""
  ```

### 4.6. `vector_store.py`
* **Class:** `AuditoriumVectorDB`
* **Implementation Details:**
  * Opens embedded LanceDB at `./data/lancedb/`.
  * Manages table `person_tracks` with pyarrow schema:
    * `track_id` (`str`)
    * `siglip_vector` (`vector[768]`)
    * `body_vector` (`vector[512]`)
    * `face_vector` (`vector[512]`, nullable)
    * `has_face` (`bool`)
    * `start_sec` (`float`)
    * `end_sec` (`float`)
    * `start_frame` (`int`)
    * `end_frame` (`int`)
    * `best_crop_path` (`str`)
    * `face_crop_path` (`str`)
    * `video_source` (`str`)
* **Contract:**
  ```python
  def insert_track(self, record: dict) -> None:
  def update_track_duration(self, track_id: str, end_sec: float, end_frame: int) -> None:
  def search_by_text_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
  def search_by_body_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
  def search_by_face_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
  def get_all_records(self) -> list[dict]:
  ```

### 4.7. `pipeline.py`
* **Main Orchestrator CLI:**
  ```bash
  python pipeline.py --video data/raw_videos/sample.mp4 --display
  ```
* **Loop Workflow:**
  1. Open video stream using `cv2.VideoCapture`.
  2. For each frame:
     - Run `detector.detect(frame)`.
     - Pass detections to `tracker.update(...)`.
     - If any track reaches frame 24:
       - Save `best_crop` to `data/crops/crop_{track_id}.jpg`.
       - Extract SigLIP image embedding.
       - Extract OSNet body embedding.
       - Run `face_module.extract_face()`. If face exists, save `face_{track_id}.jpg`.
       - Insert consolidated record into LanceDB.
     - Optionally draw active bounding boxes and display live video.
  3. When video ends, update final `end_sec` and `end_frame` for all remaining active tracks in LanceDB.

### 4.8. `search_engine.py`
* **Standalone Retrieval CLI:**
  ```bash
  # Search by text prompt
  python search_engine.py --text "person wearing dark blue jacket"

  # Search by reference person image
  python search_engine.py --image path/to/person.jpg

  # Search by reference face image
  python search_engine.py --face path/to/face.jpg
  ```
* Returns matching Track IDs, confidence scores, timestamps (`start_sec` – `end_sec`), and paths to stored crops.

### 4.9. `vlm_qa.py` (Post-Execution Local VLM Q&A)
* **Standalone Interactive CLI:**
  ```bash
  python vlm_qa.py --question "Did anyone wearing a red shirt enter the room, and when?"
  ```
* **Implementation Details:**
  * Uses local Hugging Face model: `Qwen/Qwen2-VL-2B-Instruct` or `HuggingFaceTB/SmolVLM-Instruct`.
  * **Hybrid RAG Flow:**
    1. Accepts user natural language question.
    2. Embeds question using SigLIP text encoder $\rightarrow$ Queries LanceDB for top 3–5 matching track records.
    3. Retrieves corresponding `best_crop_path` images and timeline metadata (`start_sec`, `end_sec`).
    4. Constructs multimodal prompt passing candidate images and timestamp context to the local VLM.
    5. Generates grounded answer citing specific Track IDs and timeline intervals.

---

## 5. Dependencies (`requirements.txt`)

```text
torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.1.0
supervision>=0.19.0
torchreid>=0.2.5
insightface>=0.7.3
onnxruntime>=1.17.0
transformers>=4.40.0
accelerate>=0.28.0
lancedb>=0.6.0
pyarrow>=15.0.0
opencv-python>=4.9.0
pillow>=10.2.0
numpy>=1.24.0
```

---

## 6. Execution Order for Coding Agent

1. **Setup:** Create `requirements.txt` and verify dependencies.
2. **Data Ingestion:** Implement `download_sample.py` to ensure local test video availability.
3. **Detection & Tracking Core:** Implement `detector.py` and `tracker.py` with 24-frame RAM buffering.
4. **Feature Extraction:** Implement `embedder.py` (OSNet + SigLIP) and `face_module.py` (ArcFace).
5. **Storage:** Implement `vector_store.py` with LanceDB table definition.
6. **Pipeline:** Combine modules in `pipeline.py` and run on the sample video.
7. **Retrieval CLI:** Implement `search_engine.py` for text, image, and face vector search.
8. **VLM Reasoning:** Implement `vlm_qa.py` integrating SigLIP vector filtering and local VLM image-question answering.