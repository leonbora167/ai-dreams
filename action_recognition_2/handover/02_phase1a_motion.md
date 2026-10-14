# Phase 1A: motion estimation

Goal: a function that takes two consecutive analysis frames and returns a dense flow field, with a simple way to compare a cheap method against a learned one. This feeds the per-camera baseline in Phase 1C.

Prerequisite: Phase 0 approved. Read MODELS.md section "Phase 1A".

---

## Step 0: model check with the user

Propose the order to try (default): 1) OpenCV DIS (RULE), 2) frame differencing as a sanity reference, 3) one learned flow (FastFlowNet or RAFT-small) only if DIS looks insufficient or the user wants the comparison.

Ask the user to confirm which to implement. For any learned model, follow the verification steps in MODELS.md before using it. Log results in MODELS.md.

## Step 1: implement `src/motion.py` (plain functions)

- `flow_dis(prev_frame, next_frame, preset="medium") -> flow (H, W, 2) float32`. Convert to gray internally.
- `flow_frame_diff(prev_frame, next_frame) -> magnitude-like (H, W) float32` for a crude reference (note it has no direction; keep it clearly separate from flow).
- If approved: `flow_learned(prev_frame, next_frame) -> flow (H, W, 2) float32`. Handle model-specific input size rules (e.g. dimensions divisible by 8) inside this function and return flow resized to the original frame size, with values scaled accordingly. Comment on the scaling.
- `flow_to_speed_dir(flow) -> (speed (H,W), angle (H,W))`.
- A short comment in the file on units: flow is in pixels per analysis-frame interval (i.e. per 1/ANALYSIS_FPS seconds).

## Step 2: script `scripts/p1a_motion.py`

Arguments: `--video`, `--method`, `--max_seconds`. Reads frames, computes flow for consecutive pairs, writes visual outputs, prints average ms per frame (measured with `time.perf_counter`, warm-up excluded, state hardware).

## Visual outputs (`outputs/phase1a/`)

For each video in the user's set (normal, running, fight if available):
1. `<video>_<method>_speed_overlay.mp4`: speed heatmap overlaid on the frame (fixed colour scale across the whole video so brightness means the same thing everywhere; state vmax).
2. `<video>_<method>_arrows.png`: contact sheet of 6 frames with flow arrows drawn on a coarse grid (e.g. every 24 px), arrow length scaled and capped.
3. `<video>_<method>_speed_timeline.png`: mean and 90th-percentile speed over time.
4. If more than one method: `<video>_compare.png` with side-by-side speed heatmaps of the same frame pairs, labelled with method and ms/frame.

## What the user should check

- Static background should be near zero speed. Camera shake or compression noise showing as motion is a red flag. Report it.
- Running/surge segments should clearly stand out in the timeline. If they do not, say so honestly.
- Arrow directions should match visible motion on a few spot checks.
- Latency per frame on the user's hardware, per method.
- Is ANALYSIS_FPS adequate? Very fast motion may break flow at low fps; very slow crowd motion may be too small to measure at high fps. Show an example if you see a problem.

## Questions to ask in the report

- Which method is the default for Phase 1C?
- Is the latency acceptable for N cameras (ask how many cameras per edge box)?

## Acceptance checklist

- [ ] Flow function(s) produce sensible output on all sample videos
- [ ] Visual outputs reviewed by user
- [ ] Latency numbers measured and reported
- [ ] One method chosen as default and recorded in `config.py` (`MOTION_METHOD = "..."`)
- [ ] MODELS.md verification entries filled in for any learned model used

STOP. Phase report, then wait for "approved phase 1A".
