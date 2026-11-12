# Output Logs & UI Integration Guide (`output_logs.md`)

This document explains **where every output is printed**, **which module produces which data**, and **how a UI / backend team can pull structured data** instead of scraping terminal logs.

---

## Part 1: Plain-English Guide (For Product, Management & UI Teams)

Think of the system as **two separate stages**:

1. **Stage 1 — Watching & Saving (`pipeline.py`)**
   - Reads the video frame-by-frame.
   - Runs AI models to spot **people**, **objects/bags**, **faces**, **poses** (standing/sitting), and **scene summaries**.
   - Saves cropped images (`.jpg`) to `data/crops/`, `data/object_crops/`, and `data/scenes/`.
   - Saves all structured metadata (timestamps, IDs, owners, statuses, summaries) and search vectors into a local database folder called **LanceDB** (for example `data/lancedb/`).
2. **Stage 2 — Searching & Answering (`search_engine.py`, `vlm_qa.py`, `visualize_db.py`)**
   - **Nothing in Stage 2 re-runs the video detectors.** Everything comes from the saved **LanceDB** tables and image folders.
   - `search_engine.py` runs fast vector similarity search and prints formatted text blocks to your terminal.
   - `vlm_qa.py` looks up the best matching people/objects/events in LanceDB, sends those images + metadata to the Vision-Language Model (VLM), and prints the final English answer to your terminal.
   - `visualize_db.py` is already a working FastAPI web server that serves LanceDB data and images as **JSON + HTTP** for a browser UI.

### Quick Map: "What I Want to Show in the UI" -> "Where It Lives"

| What the UI Needs to Display | Plain-English Source | Technical Source |
| :--- | :--- | :--- |
| **List of detected people, body/face crops, timestamps** | Saved in DB after tracking + Re-ID | LanceDB `person_tracks` table (`vector_store.py`) |
| **Tracked bags/objects, owner person ID, & abandoned alerts** | Saved in DB by Object Tracker | LanceDB `object_tracks` table (`vector_store.py`) |
| **Evidence snapshot image when a bag was abandoned** | Saved as a `.jpg` scene image | `abandoned_frame_path` column -> `data/scenes/` |
| **Scene keyframe gallery & people counts over time** | Saved every N seconds during video run | LanceDB `scene_frames` table (`vector_store.py`) |
| **AI text summaries of what happened in each 8s window** | Saved by Phase 2 VLM summarizer | LanceDB `video_events` table (`vector_store.py`) |
| **Search results ("find black backpack", "person in red")** | Returned by `AuditoriumVectorDB.search_*()` | `vector_store.py` (printed by `search_engine.py`) |
| **Natural-language Q&A answer ("Who owns this bag?")** | Returned by `LocalVLMQA.answer_question()` | `vlm_qa.py` (printed at the bottom of `vlm_qa.py`) |
| **Ready-to-use JSON API + image server for a web UI** | Built-in FastAPI dashboard backend | `visualize_db.py` (`/api/dbs`, `/api/data`, `/media`) |

---

## Part 2: Technical Reference (Exact Print Sites, Schemas & APIs)

### 1. Where `search_engine.py` Prints Its Outputs

In [`search_engine.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py), CLI queries call `AuditoriumVectorDB` methods in [`vector_store.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vector_store.py) (which return `list[dict]`) and pass those lists to four formatting functions:

| CLI Flag | DB Method Called (`vector_store.py`) | Print Function (`search_engine.py`) |
| :--- | :--- | :--- |
| `--text`, `--image`, `--video` | `search_by_text_vector` / `search_by_body_vector` / `search_by_multimodal_vector` | [`print_results`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py#L19-L63) |
| `--face` | `search_by_face_vector(...)` | [`print_results`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py#L19-L63) |
| `--scenes --text` | `search_scene_frames(...)` | [`print_scene_results`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py#L66-L89) |
| `--object`, `--object-text` | `search_objects_by_vector(...)` | [`print_object_results`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py#L92-L121) |
| `--events` | `search_video_events(...)` | [`print_event_results`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/search_engine.py#L124-L144) |

**For UI integration:** Do not parse stdout from `search_engine.py`. Instead, call `vector_db.search_*()` directly—every result dictionary already contains all metadata plus `_distance` (cosine distance) and image file paths (`best_crop_path`, `face_crop_path`, `owner_crop_path`, `abandoned_frame_path`, `frame_path`, `keyframe_path`).

---

### 2. Where `vlm_qa.py` Prints Its Outputs

In [`vlm_qa.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vlm_qa.py), execution has three visible console stages inside `main()`:

1. **Route Classification** ([`vlm_qa.py#L661-L668`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vlm_qa.py#L661-L668)):
   - `QueryRouter.classify(...)` selects one of 5 routes: `track_id`, `counting`, `person_forensic`, `object_ownership`, or `video_summary`.
   - Printed as: `-> Selected Route: [ROUTE] (...)`.
2. **Retrieved Evidence Summary (`[2/3]`)** ([`vlm_qa.py#L750-L769`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vlm_qa.py#L750-L769)):
   - Prints counts and summaries of `candidates` (persons), `object_records` (objects), `video_events`, `scene_frames`, and `structured_stats`.
3. **Final VLM Answer (`[3/3]`)** ([`vlm_qa.py#L794-L809`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vlm_qa.py#L794-L809)):
   - `answer = vlm.answer_question(...)` returns a plain `str` generated by either `HuggingFaceLocalBackend` (local `InternVL3-2B-hf`, `SmolVLM2`, `Phi-4`, `Llama-3.2`) or `OpenAICompatibleRemoteBackend` (`--endpoint-url`).
   - Printed inside the `VLM ANSWER:` banner at lines `805–809`.

**For UI integration:** Call `QueryRouter.classify(...)`, run the DB retrieval block to collect `candidates`, `object_records`, `video_events`, `scene_frames`, `structured_stats`, and call `vlm.answer_question(...)`. Return `{ "route": route, "answer": answer, "evidence": { ... } }` as JSON so the UI can show both the LLM answer and clickable evidence crops/frames.

---

### 3. Live Pipeline Modules (What Each Module Outputs & Logs)

During ingestion ([`pipeline.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/pipeline.py)), each component produces in-memory outputs and console logs:

| Module | File | Method & Return Value | Console Logs Printed |
| :--- | :--- | :--- | :--- |
| **Person Detector** | [`detector.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/detector.py) | `PersonDetector.detect(frame)` -> `(bboxes: np.ndarray (N,5) [x1,y1,x2,y2,conf], crops: list[np.ndarray])` | Silent |
| **Person Tracker** | [`tracker.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/tracker.py) | `AuditoriumTracker.update(...)` -> `(active_tracks: [{'track_id', 'bbox'}], newly_confirmed: [{'track_id', 'best_crop', 'top_crops', 'start_sec', 'start_frame', ...}])` | Silent (`pipeline.py` logs `[Frame ...] Confirmed Local #...`) |
| **Pose Analyzer** | [`pose_module.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/pose_module.py) | `classify_crop(crop)` -> `'standing' \| 'sitting' \| 'unknown'`; `analyze_scene(frame)` -> `{'person_count', 'standing_count', 'sitting_count', 'unknown_count'}` | Silent |
| **Embedder & Face** | [`embedder.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/embedder.py), [`face_module.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/face_module.py) | `extract_body_reid` (512-d), `extract_multimodal_image/text` (768-d SigLIP), `extract_best_face_from_crops` -> `(face_vec, face_crop)` | Weight loading info |
| **Global Re-ID** | [`reid_manager.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/reid_manager.py) | `resolve_track_id(...)` -> `(global_id, is_reappearance, match_info)`; maintains `self.gallery[gid]` | `[Global Re-ID] Local Track #... registered / Matched ...` |
| **Object Detector** | [`object_detector.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/object_detector.py) | `detect(frame)` / `detect_tiled(frame)` / `detect_roi(frame, roi)` -> `(detections: (N,5), labels: list[str])` | `[ObjectDetector] Initialized 'yolov8s-worldv2.pt'...` |
| **Object Tracker** | [`object_tracker.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/object_tracker.py) | `update(...)` -> `(active_objs, new_objs, new_abandoned)`; `get_all_indexed_objects()` includes `status` (`attended` / `stationary` / `abandoned` / `unattended_at_end`) | `pipeline.py` prints `Indexed Object [OBJ:...]` and `[ALERT - ABANDONED OBJECT]` |
| **Video Summarizer** | [`video_summarizer.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/video_summarizer.py) | `maybe_summarize_window(...)` / `flush_final(...)` -> summary chunk dict with `chunk_id`, `summary_text`, `tile_notes_json`, `referenced_track_ids_json`, `referenced_object_ids_json`, `keyframe_path` | Tile & rollup progress when `--summarize` is enabled |

---

### 4. Persistent Storage: LanceDB Tables (`vector_store.py`)

All tables live inside the `--db-path` directory (default `data/lancedb/`). Schemas are defined in [`vector_store.py#L12-L79`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/vector_store.py#L12-L79):

1. **`person_tracks`**
   - `track_id`, `pose_label`, `has_face`, `start_sec`, `end_sec`, `start_frame`, `end_frame`, `best_crop_path`, `face_crop_path`, `video_source`, `intervals_json`, `appearance_crops_json`, `siglip_vector` (768-d), `body_vector` (512-d), `face_vector` (512-d, nullable).
2. **`object_tracks`**
   - `object_id`, `label`, `owner_track_id`, `owner_crop_path`, `status` (`attended`, `stationary`, `abandoned`, `unattended_at_end`), `abandoned_since_sec`, `abandoned_frame_path`, `start_sec`, `end_sec`, `start_frame`, `end_frame`, `best_crop_path`, `intervals_json`, `video_source`, `siglip_vector` (768-d).
3. **`scene_frames`**
   - `frame_id`, `timestamp_sec`, `frame_path`, `person_count`, `standing_count`, `sitting_count`, `unknown_count`, `video_source`, `siglip_vector` (768-d).
4. **`video_events`**
   - `chunk_id`, `start_sec`, `end_sec`, `trigger_reason`, `summary_text`, `tile_notes_json`, `referenced_track_ids_json`, `referenced_object_ids_json`, `keyframe_path`, `video_source`, `siglip_vector` (768-d).

---

### 5. Existing UI Backend & Recommended Integration Pattern

[`visualize_db.py`](file:///c:/Projects/ai-dreams/multi-feature-track-vss/visualize_db.py) already provides a FastAPI app (`create_app(default_db_path)`) with three endpoints you can reuse or extend:

- `GET /api/dbs` -> discovers all `data/lancedb*` directories.
- `GET /api/data?db_path=...` -> returns JSON with `stats`, `persons`, `objects`, `events`, and `scenes` (with vector arrays stripped and image paths converted to `/media?path=...` URLs).
- `GET /media?path=...` -> safely streams local `.jpg` crops and scene frames from inside the project directory.

**Recommended next step for full UI integration:** Add two JSON endpoints to `visualize_db.py` (or a dedicated FastAPI server):
- `POST /api/search` -> calls `LocalEmbedder` + `AuditoriumVectorDB.search_*()` and returns the raw `list[dict]` (converted with `_safe_media_url` for image paths).
- `POST /api/qa` -> calls `QueryRouter.classify()` + DB retrieval + `LocalVLMQA.answer_question()` and returns `{ "route", "answer", "persons", "objects", "events", "scenes" }`.
