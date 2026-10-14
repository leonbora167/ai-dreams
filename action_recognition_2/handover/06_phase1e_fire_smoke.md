# Phase 1E: fire and smoke detector (parallel branch)

Goal: a small object detector for fire and smoke, run every few frames, producing events that go straight to fusion and bypass tier 2. Independent from the other tier 1 phases; the user decides when to do it (after Phase 0 at the earliest).

Prerequisite: Phase 0 approved. Read MODELS.md section "Phase 1E".

Read DATASETS.md section "Phase 1E" before proposing datasets.

---

## Step 0: ask the user

1. Which fire-related events matter: flames, smoke, flares, pyrotechnics, burning objects? Flares and smoke bombs are common in stadiums. Decide if they count as alarms or not.
2. Do they have fire/smoke footage or images from stadiums? How many? (Public datasets rarely look like stadium CCTV.)
3. Is the intent to detect small early smoke or large visible fire? This affects resolution.
4. Dataset choices and licences. Propose candidates (e.g. D-Fire) but ask the user to confirm source, size and licence before downloading.

## Step 1: pretrained check first

Verify the chosen detector (YOLOv8n/s or YOLO11n via `ultralytics`) per MODELS.md. Run it with whatever fire/smoke weights the user supplies or a public checkpoint they approve. Do NOT start training in this step.

## Step 2: implement `src/fire.py`

- `load_fire_model(weights_path, device)`
- `detect_fire_smoke(model, frame, conf_thr) -> list of {"box":[x1,y1,x2,y2], "cls": "fire"|"smoke", "conf": float}`
- `fire_events_from_detections(detections_over_time, min_frames, min_conf) -> list of events` with `"source": "fire"`. Require persistence over several detections before an event, to cut flicker false alarms (stadium lights, sun glare, flags).
- Optional: `color_flicker_prefilter(frame_prev, frame) -> bool` classical check (reddish/orange saturated pixels with temporal change), only if the user wants it.

## Step 3: script `scripts/p1e_fire.py --video ...`

Runs every N-th analysis frame (N configurable, default 3). Measures ms/frame.

## Visual outputs (`outputs/phase1e/`)

1. `<video>_fire_overlay.mp4`: boxes with class and confidence.
2. `fire_samples_contact_sheet.png`: TP examples and misses on the user's fire samples.
3. `false_positive_hunt.png`: highest-confidence detections on NORMAL footage (stadium lights, flags, red clothing, sun glare). This sheet is the most important check.
4. `confidence_histogram.png`: confidence for detections on fire samples vs normal footage.

## What the user should check

- False positives on stadium-specific lights and colours.
- Misses on small/distant smoke.
- Whether a fine-tuning run is needed. If yes, propose a minimal plan (dataset, number of epochs, validation split by source) and wait for approval. Training itself is a separate approved step inside this phase.

## Acceptance checklist

- [ ] Model verified and running
- [ ] False positive hunt reviewed
- [ ] Decision: use as is / fine-tune (and fine-tuning done and re-checked if approved)
- [ ] `config.py`: `FIRE_EVERY_N_FRAMES`, `FIRE_CONF_THR`, `FIRE_MIN_FRAMES` set

STOP. Phase report, then wait for "approved phase 1E".
