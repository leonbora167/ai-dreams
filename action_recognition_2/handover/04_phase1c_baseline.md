# Phase 1C: per-camera baseline and anomaly score

Goal: learn what "normal" looks like for ONE camera, then score new frames by how much each grid cell deviates. This is the VelocityNet-style idea, but with per-camera statistics instead of global thresholds, and at grid level instead of per person.

Prerequisite: Phase 1B approved (needs motion + density).

Read DATASETS.md section "Phase 1C": baselines are fitted only on the user's own normal footage.

---

## Step 0: ask the user

1. Which normal footage should be used to fit the baseline for this camera? (Folder `data/normal/<camera_name>/`.) How many minutes? Is it representative (different times of day, crowd sizes, match vs non-match)?
2. Is there footage with KNOWN abnormal segments (running, surge, fight)? Ask for rough timestamps ("seconds 40-55 are running"). These are used ONLY to evaluate, never to fit the baseline.
3. Which abnormalities matter most for tier 1: sudden speed increase, sudden stop, direction reversal (counter-flow), density spike?

## Step 1: implement `src/baseline.py` (plain functions)

Features per grid cell, per frame (all (grid_h, grid_w) float32):
- `speed_mean`, `speed_p90` from flow
- `dir_coherence`: how aligned flow directions are in the cell (0 = chaotic, 1 = all same direction). Explain the formula in a comment.
- `divergence`: simple measure of spreading/converging flow (e.g. from the flow gradient averaged per cell)
- `density`: from Phase 1B

Functions:
- `compute_grid_stats(flow, density_grid, grid_h, grid_w) -> dict of arrays`
- `fit_baseline(list_of_stats_dicts) -> baseline dict`: per cell and per feature, store robust statistics (median and MAD, plus a few percentiles). Save with `np.savez` to `weights/baseline_<camera>.npz`.
- `score_frame(stats, baseline) -> score map (grid_h, grid_w)`: for each feature a robust z-score, then combine (e.g. max or weighted sum; start with max and say why). Keep the combination formula visible and easy to edit.
- `smooth_scores(score_maps_over_time, window)`: small moving average over time to remove single-frame noise (simple function).

Cells with little data in the baseline (e.g. empty seats, never any crowd) must not produce huge z-scores from tiny MADs. Add a floor on the scale and comment on it.

Start with the simplest approach (robust z-score per cell). Only try GMM/Isolation Forest/autoencoder if the user asks after reviewing results.

## Step 2: scripts

- `scripts/p1c_fit_baseline.py --normal_dir ... --camera name`: fits and saves the baseline, prints how many frames used.
- `scripts/p1c_score.py --video ... --camera name`: scores a video and writes visuals.

## Visual outputs (`outputs/phase1c/`)

1. `baseline_maps.png`: for each feature, a heatmap of the per-cell median and of the per-cell spread over the normal footage. These show what the model thinks normal is. User checks they make sense (e.g. busy aisles show more motion).
2. `<video>_score_overlay.mp4`: anomaly score heatmap over the frame, fixed colour scale.
3. `<video>_score_timeline.png`: max cell score and number of cells above threshold over time. Mark user-provided abnormal segments as shaded spans.
4. `<video>_top_frames.png`: contact sheet of the 12 highest-scoring frames with overlays.
5. `held_out_normal_timeline.png`: hold out part of the normal footage (e.g. last 20%) NOT used in fitting, and score it. This is the false alarm check.
6. `threshold_sweep.png`: for thresholds from low to high, plot (a) false alarm cells/frames per hour on held-out normal, (b) fraction of the user-marked abnormal frames detected. State clearly how many abnormal segments you had; small numbers mean weak evidence.

## What the user should check

- Do abnormal segments stand out above held-out normal footage? Be honest if they do not.
- Which feature drives the score in the top frames? (Print the dominant feature per top frame.)
- False alarms on held-out normal: where in the frame and why (e.g. a door opening, camera shake)?
- Does the baseline need to depend on time of day or crowd size? (Propose, do not implement without approval.)

## Questions to ask in the report

- Chosen starting threshold (recall-oriented) and the false alarms/hour it implies on held-out normal.
- Do they want separate baselines per crowd-density level?

## Acceptance checklist

- [ ] Baseline fitted and saved for the camera
- [ ] Baseline maps look plausible
- [ ] Score timelines reviewed, with honest statement of what is and is not detected
- [ ] Held-out false alarm rate measured and reported
- [ ] `config.py`: `SCORE_THRESHOLD`, `SCORE_SMOOTH_WINDOW` set

STOP. Phase report, then wait for "approved phase 1C".
