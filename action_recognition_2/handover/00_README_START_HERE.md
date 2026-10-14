# Crowd anomaly detection pipeline: agent handover (START HERE)

Read this file first, every session. Then read ONLY the phase file for the phase currently in progress (see PROGRESS.md). Do not read ahead and do not build ahead.

---

## 1. What we are building

A tiered, edge-deployable video anomaly detection pipeline for large crowds in stadium CCTV footage.

Target events:
- Fighting / violence
- Crowd surge, stampede, sudden running, sudden halt, counter-flow
- Fire and smoke
- Out of scope for now: theft (not reliably detectable in large crowds)

Architecture (three tiers):

```
CCTV stream -> decode/subsample (5-10 FPS)
  |-- Fire/smoke detector (parallel branch, alerts go straight to fusion)
  |-- TIER 1 (every frame, cheap): motion -> density -> per-camera baseline -> trigger + region proposal
  |-- TIER 2 (only on flagged regions): ROI crop+buffer -> compact action model -> temporal smoothing
  |-- TIER 3: fusion + alert (+ optional server-side verifier, later)
```

Key design decisions already made (do not change without asking the user):
- NO per-person detection or tracking. We work with grid-level / region-level statistics.
- Tier 1 only says WHERE and WHEN something is unusual. Tier 2 says WHAT it is, from raw pixels of the flagged region.
- Tier 1 and Tier 2 are decoupled models. They communicate only through an "event" (box + time window). No feature sharing.
- The whole flagged region is sent to the action model as ONE crop, not per-person crops.
- Tier 1 is tuned for recall. Tier 2 and smoothing are tuned for precision.
- Target hardware: mid-range edge GPU (Orin AGX or small RTX-class box). Ask the user for exact specs in Phase 0.
- No VLMs on the edge. A server-side verifier is optional and comes last.

The user is a researcher who will read and edit the code themselves. Readability beats cleverness.

---

## 2. How you must work: the feedback loop

You build ONE phase at a time. Each phase follows this loop, no exceptions:

1. Read the phase file.
2. Confirm models and inputs with the user (see MODELS.md rules). Do not assume anything is available.
3. Write the minimum code for that phase.
4. Run it on the user's sample video(s) and produce the VISUAL OUTPUTS listed in the phase file into `outputs/<phase_name>/`.
5. Write a phase report (template in section 8) and update PROGRESS.md.
6. STOP. Wait for the user to type "approved <phase>" (or give feedback). Only then start the next phase.

If the user gives feedback, fix it within the same phase and re-run the visual checks. Do not move on.

Hard rules:
- Never skip a visual check. If you cannot run something (no GPU, no network, missing weights), say so plainly and ask the user. Do not fake outputs, invent metrics, or silently substitute a different model.
- Never report a number you did not measure. Latency, counts, accuracy, all must come from a run you actually did.
- If a model fails to load or behaves oddly, report it. Do not quietly swap it for another.
- Do not install heavy frameworks to "make things easier" (no Lightning, Hydra, MMDetection stacks, etc.) unless the user agrees. If a model's own repo needs one, ask first.
- Ask at most a few focused questions at a time. Put all questions at the end of your report.

---

## 3. Code style (important)

The user wants simple code they can read and modify.

- Use plain functions in plain modules. NO classes, unless a PyTorch model genuinely requires `nn.Module` (e.g. a tiny classifier head). If you need one, keep it under ~30 lines and comment it.
- Each function: one job, ideally under 40 lines, with a one-line docstring saying what goes in and what comes out (shapes included).
- Pass data as numpy arrays, lists, tuples and plain dicts. No custom data classes, no registries, no abstract base classes, no decorators, no dependency injection.
- Configuration lives in one file, `config.py`, as simple module-level constants. No YAML/Hydra.
- Prefer clear variable names over comments, but add short "why" comments where a choice is non-obvious (e.g. why we pad here).
- Scripts are plain files with `argparse` and a `main()` that calls functions in order, so the user can read the whole flow top to bottom.
- Add simple `assert` checks on shapes/dtypes at module boundaries. They catch most bugs and explain the contracts.
- No premature optimisation. Get it correct and visible first. Optimisation (ONNX/TensorRT, batching) is a later phase and only if asked.
- Python 3.10+, PyTorch, OpenCV, NumPy, Matplotlib. Add other libraries only when a phase needs them, and tell the user.

---

## 4. Repo layout

```
crowd_anomaly/
  config.py              # all constants (paths, fps, grid size, thresholds, model choices)
  MODELS.md              # model registry, verified by the user (see file)
  PROGRESS.md            # phase checklist + log, updated by you
  src/
    video_io.py          # read_frames, write_video, subsample
    viz.py               # overlays, heatmaps, grids/contact sheets, plots
    motion.py            # Phase 1A
    density.py           # Phase 1B
    baseline.py          # Phase 1C
    trigger.py           # Phase 1D
    fire.py              # Phase 1E
    roi.py               # Phase 2A
    action.py            # Phase 2B
    smoothing.py         # Phase 2C
    fusion.py            # Phase 3
  scripts/               # one runnable script per phase: p1a_motion.py, p1b_density.py ...
  data/
    videos/              # user-provided sample videos
    normal/              # normal footage per camera, used to fit baselines
    action_clips/        # prepared training clips (Phase 2B)
  weights/               # downloaded/finetuned weights
  outputs/               # visual outputs, one folder per phase
```

---

## 5. Data contracts between phases

These are the ONLY interfaces between modules. Keep them stable. If you must change one, tell the user and update this section.

- Frame: `np.ndarray`, shape (H, W, 3), dtype uint8, BGR (OpenCV convention). All frames in the pipeline are already subsampled and resized by `video_io`.
- Flow: `np.ndarray`, shape (H, W, 2), float32, pixels moved between two consecutive analysis frames, channels (dx, dy).
- Density map: `np.ndarray`, shape (H, W), float32, estimated people per pixel (sums to count). Also acceptable: per-cell counts (grid_h, grid_w) float32. State which one in the code comment.
- Grid stats: dict of `np.ndarray` each of shape (grid_h, grid_w) float32, keys such as `speed_mean`, `speed_p90`, `dir_coherence`, `divergence`, `density`.
- Anomaly score map: `np.ndarray`, shape (grid_h, grid_w), float32, 0 = normal, larger = more unusual.
- Event: plain dict
  `{"id": int, "start_frame": int, "end_frame": int, "box": [x1, y1, x2, y2], "score": float, "source": "motion" | "fire" | "action"}`
  Box is in analysis-frame pixel coordinates.
- Clip: `np.ndarray`, shape (T, S, S, 3), uint8, after ROI crop and resize (S = model input size).
- Action prediction: dict `{"probs": {class_name: float}, "top": class_name, "top_prob": float}`.

---

## 6. Phase index

Work in this order. Phase 1E (fire/smoke) is independent and may be done any time after Phase 0 if the user prefers.

| Phase | File | Summary |
|---|---|---|
| 0 | 01_phase0_setup.md | Environment, repo skeleton, video IO, viz helpers |
| 1A | 02_phase1a_motion.md | Motion estimation |
| 1B | 03_phase1b_density.md | Density / people localisation |
| 1C | 04_phase1c_baseline.md | Per-camera baseline and anomaly score |
| 1D | 05_phase1d_trigger_roi.md | Trigger and region proposal (events) |
| 1E | 06_phase1e_fire_smoke.md | Fire and smoke detector (parallel branch) |
| 2A | 07_phase2a_roi_crop.md | ROI crop, buffer, resize, tiling |
| 2B | 08_phase2b_action_model.md | Compact action model: data, finetune, eval |
| 2C+3 | 09_phase2c_3_smoothing_fusion.md | Temporal smoothing, fusion, alerts |
| 4 | 10_phase4_integration_eval.md | End-to-end run, evaluation, latency, report |

---

## 7. Where to find the model list and the dataset list

See `MODELS.md`. It lists candidate models per stage, which need training and which work directly. Every model there starts as UNVERIFIED. You must verify each one with the user before relying on it.

See `DATASETS.md`. It lists candidate datasets per stage, with access notes and pitfalls. Read it at the start of any step that involves data (Phase 1B, 1E, 2B) and follow its rules: ask before downloading, never bypass access requests, inspect samples before training, split by source. The user's own footage is the most important dataset.

---

## 8. Phase report template (use after every phase)

```
## Phase <id> report
**Built:** (files and functions, one line each)
**How to run:** (exact command)
**Visual outputs to inspect:** (paths in outputs/<phase>/ and what to look for in each)
**Measured:** (latency per frame, counts, etc. ONLY numbers you measured; say on what hardware)
**What worked / what looks wrong:** (be honest, including doubts)
**Decisions I made:** (anything the user should know or may want to change)
**Questions for you:** (numbered, short)
**Status:** AWAITING APPROVAL. I will not start the next phase until you say "approved <phase>".
```

---

## 9. Things you must never do

- Build several phases in one go.
- Treat a model as "available" because you remember it exists. Check with the user and actually load it.
- Download datasets or weights without telling the user the source, size and licence.
- Hide failures, e.g. wrap errors in try/except and continue silently.
- Add tracking (SORT/DeepSORT/ByteTrack) or per-person pipelines.
- Rewrite working code from an earlier approved phase unless the user asks.
