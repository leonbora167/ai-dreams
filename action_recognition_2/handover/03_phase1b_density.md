# Phase 1B: density / people localisation

Goal: estimate how many people are where, per frame, without tracking. The output is a density map or per-cell counts, used in Phase 1C together with motion.

Prerequisite: Phase 1A approved. Read MODELS.md section "Phase 1B".

Read DATASETS.md section "Phase 1B" before proposing any fine-tuning data.

---

## Step 0: decide the approach WITH the user, using evidence

The right choice depends on how big heads are in the user's footage. Before picking, do this quick measurement:

1. Show the user 6 frames from different parts of the stands at analysis resolution, and ask them to estimate typical head size in pixels (or measure a few yourself by drawing rulers on the image).
2. Rule of thumb to discuss (not a law): heads larger than roughly 10-15 px can work with a head detector; smaller than that, a density-map counting network or grid-only statistics is safer.

Propose one primary approach and at most one alternative:
- Head detector (YOLO nano/small): needs FINETUNE on head data. Pretrained person/head detectors may under-detect in dense crowds.
- Density-map model (P2PNet / DM-Count / CSRNet): pretrained weights exist but expect a domain gap. Plan to sanity-check against manual counts.
- Grid-flow only: skip density model. Use as fallback if nothing else works.

Ask the user to pick. Verify the chosen model per MODELS.md before use.

## Step 1: implement `src/density.py`

- `load_density_model(name, weights_path, device)`: returns the model (a plain object); keep it a function, not a class.
- `predict_density(model, frame) -> density map (H, W) float32` (or per-cell counts; document which). Handle tiling if the model needs a smaller input than the frame (tile with overlap, stitch, and comment on how overlaps are handled to avoid double counting).
- `density_to_grid(density, grid_h, grid_w) -> (grid_h, grid_w) float32`: sum within each cell.
- If using a detector: `detect_heads(model, frame) -> list of [x1,y1,x2,y2,conf]` and `boxes_to_grid_counts(boxes, grid_h, grid_w, frame_shape)`.

## Step 2: script `scripts/p1b_density.py`

Arguments: `--video`, `--n_frames`. Runs on evenly spaced frames, saves visuals, prints ms/frame (excluding warm-up).

## Visual outputs (`outputs/phase1b/`)

1. `density_overlay_contact_sheet.png`: 8 frames with density heatmap overlay and estimated total count in the label.
2. `grid_counts_overlay.png`: grid drawn on frames with per-cell counts written in the cells.
3. If detector: `head_boxes.png` with boxes drawn on 6 frames, including a crop zoom on the densest area.
4. `count_vs_manual.png`: for 10 frames the USER counts manually (or counts in a sub-region), plot estimated vs manual. Ask the user to supply manual counts; if they cannot, mark this check as pending and say what you could not validate.
5. `count_timeline.png`: estimated total count over time for each video.

## What the user should check

- Heatmaps should sit on crowds, not on seats, banners or shadows. Report false hotspots.
- Counts should roughly match manual counts. A consistent underestimate is acceptable for our purposes if relative changes are right; random errors are not. Say which you observe.
- Far-away dense areas: are they handled or ignored?
- Latency per frame, and whether you can run it every frame or every N-th frame (density changes slower than motion; discuss running it at lower rate).

## Questions to ask in the report

- Is fine-tuning needed? If yes, what labelled data exists? (Propose a minimal plan but do NOT start training in this phase unless the user approves.)
- Should density run every frame or every N frames?

## Acceptance checklist

- [ ] Approach chosen and verified (MODELS.md)
- [ ] Density/grid counts look sensible on sample videos
- [ ] Count check against manual counts done (or explicitly deferred with user's consent)
- [ ] Latency measured
- [ ] `config.py`: `DENSITY_METHOD`, `DENSITY_EVERY_N_FRAMES` set

STOP. Phase report, then wait for "approved phase 1B".
