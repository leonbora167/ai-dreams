# Phase 4: end-to-end integration and evaluation

Goal: one script that runs the whole pipeline on a video or stream-like input, plus an honest evaluation: detection quality, false alarms per hour, delays, and latency per stage. No new models here. Optimisation (ONNX/TensorRT, batching, multi-camera) is NOT part of this phase unless the user asks.

Prerequisite: all earlier phases approved.

---

## Step 0: ask the user

1. Evaluation footage: which videos are fully held out (never used to fit baselines, tune thresholds or train)? Strongly recommend at least some from a different day or camera. Any ground truth timestamps for abnormal events?
2. Target cameras per edge box and the frame rate they will run at, for the latency budget.
3. Which numbers matter most for them: false alarms per camera-hour, detection delay, recall on specific classes.

## Step 1: `scripts/run_pipeline.py` (plain, readable)

Arguments: `--video`, `--camera`, `--out_dir`. The main function reads top to bottom:

1. read frames at ANALYSIS_FPS
2. motion + density -> grid stats -> score map -> events
3. (every N frames) fire detector
4. for each finished or ongoing event: crop clip, action model, smoothing
5. fusion -> alerts
6. save alerts, event logs, and a per-stage timing log

No hidden global state. Each stage is a function call with plain inputs and outputs per the README contracts.

## Step 2: evaluation script `scripts/eval_pipeline.py`

- Run on all evaluation videos.
- Match alerts to ground truth segments (overlap in time with a small tolerance; define the rule in a comment).
- Report per class: events detected, missed, false alarms. Report false alarms per hour on normal footage. Report detection delay distribution.
- Report how many events tier 1 fired vs how many were confirmed by tier 2 (this shows how much the gating saves).
- Be explicit about sample sizes; with few events, say that conclusions are weak.

## Step 3: latency profile

- Time each stage separately on the user's hardware: decode, flow, density, stats+score, trigger, crop, action model, fusion. Use `time.perf_counter`, exclude warm-up, report mean and p95.
- Compute the per-camera load: tier 1 cost per frame x fps, plus tier 2 cost x events/hour. State which stage is the bottleneck and the estimated max number of cameras per box, labelled as an estimate with its assumptions.

## Visual outputs (`outputs/phase4/`)

1. `pipeline_demo_<video>.mp4`: full overlay (events, action labels, alerts, fire boxes, small HUD with per-stage ms).
2. `eval_timeline_<video>.png`: ground truth segments vs events vs alerts.
3. `results_table.md`: counts, false alarms/hour, delays.
4. `latency_breakdown.png`: stacked bar per stage.
5. `failure_cases.png`: contact sheet of missed events and false alarms with short notes on probable cause.
6. `REPORT.md`: what works, what does not, and recommended next steps, honestly written.

## What the user should check

- Does the system fire at the right moments on footage it never saw?
- Are false alarms per hour acceptable for operators?
- Which stage fails most often: tier 1 miss, tier 2 misclassification, or smoothing delay?
- Is the latency budget OK for the target hardware?

## Candidate next steps (propose only, do not start)

- Fine-tune the action model on tier 1 harvested events from the user's cameras
- Time-of-day / crowd-size dependent baselines
- Comparison of a second compact action model
- ONNX/TensorRT export and INT8 quantisation, multi-camera batching
- Server-side verifier for flagged clips
- Camera calibration (ground-plane homography) for perspective-correct speed

## Acceptance checklist

- [ ] End-to-end run works on all evaluation videos
- [ ] Evaluation numbers reported with sample sizes
- [ ] Latency per stage measured
- [ ] REPORT.md reviewed by the user, next steps chosen by the user

STOP. Report. Wait for "approved phase 4".
