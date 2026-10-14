# Phase 2A: ROI crop, buffer, resize, tiling

Goal: convert an event (box + time window) into one or more fixed-size clips that the action model can consume. No learned model in this phase, just careful geometry and visual verification. Mistakes here silently hurt accuracy later, so this phase is not optional.

Prerequisite: Phase 1D approved (events exist).

---

## Design rules (already decided)

- The model gets the WHOLE flagged region as one crop (not per-person crops).
- Expand the box with context margin (start at 25% on each side; tunable).
- Make the crop square by growing the shorter side, then resize to the model input size S. Never stretch.
- If the square crop goes outside the frame, shift it back inside the frame first. Only pad if the frame is smaller than the crop. Do not letterbox wide regions with big black bars.
- Use ONE crop box for ALL frames of the clip (union/smoothed event box over the window). A crop that moves per frame creates fake camera motion.
- Large regions: a crop much larger than S shrinks people heavily. Support optional tiling (overlapping square tiles, each classified separately later, combined by max or vote).

## Step 0: ask the user

1. Model input size S (depends on the model chosen in 2B; for now assume 224 and make it a parameter). Number of frames per clip T (start with 16 at ANALYSIS_FPS, tell the user what duration that is in seconds; they may want a different sampling stride).
2. Which source frames should the crop come from: the analysis-resolution frames (cheap) or the original-resolution frames (more detail for small regions)? Discuss tradeoff; default to analysis-resolution first, and show the user what small regions look like when upscaled.

## Step 1: implement `src/roi.py` (plain functions)

- `expand_box(box, margin, frame_shape) -> box`: add context margin.
- `square_box(box, frame_shape) -> box`: grow to a square, shift inside frame when possible.
- `crop_and_resize(frame, box, size) -> (S, S, 3) uint8`: crop with padding only if needed, then `cv2.resize` with `INTER_AREA` for downscale / `INTER_LINEAR` for upscale (comment why).
- `make_clip(frames, box, size, num_frames) -> (T, S, S, 3) uint8`: pick `num_frames` evenly from the event window; apply the same box to all.
- `tile_boxes(box, tile_size_px, overlap) -> list of boxes`: when the region is much larger than a tile, produce overlapping square tiles; cap the number of tiles (e.g. 4) and say what happens above the cap.
- `needs_tiling(box, size, ratio_limit) -> bool`
- Assert shapes and dtype at the end of `make_clip`.

## Step 2: script `scripts/p2a_crops.py --video ... --events outputs/phase1d/events.json`

Generates clips for every event, saved as small mp4s and PNG sheets.

## Visual outputs (`outputs/phase2a/`)

1. `<event>_crop_box.png`: full frame with three boxes drawn: raw event box, expanded box, final square crop box. Colour-coded, labelled.
2. `<event>_clip_strip.png`: T frames of the final clip in a horizontal strip.
3. `<event>_clip.mp4`: the clip at S x S.
4. `edge_cases.png`: crops for events at the frame border, tiny events, huge events (with tiles drawn). Create synthetic boxes if real ones do not cover these cases, and label them as synthetic.
5. `scale_report.png`: for each event, the effective size reduction (box side in pixels vs S). Events with heavy downscale or upscale are flagged.

## What the user should check

- Is the whole activity inside the crop with some context?
- Are people recognisable at the crop resolution? This predicts whether the action model has a chance.
- Is the crop stable across frames (no jitter)?
- For large regions, do tiles make sense, and which strategy looks best (single downscaled crop vs tiles)?

## Acceptance checklist

- [ ] Crop pipeline gives correct shapes and no stretching
- [ ] Edge cases handled
- [ ] Tiling decision made (on/off, max tiles)
- [ ] `config.py`: `CROP_MARGIN`, `CLIP_FRAMES`, `CLIP_SIZE`, `USE_TILING`, `MAX_TILES` set

STOP. Phase report, then wait for "approved phase 2A".
