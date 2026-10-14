# Phase 1D: trigger and region proposal (events)

Goal: turn the per-cell anomaly score maps into discrete EVENTS: a box and a time window. This is the output of tier 1 and the input of tier 2.

Prerequisite: Phase 1C approved.

---

## Step 0: ask the user

1. Minimum event duration before it counts (seconds)? Suggest starting at 1-2 s and explain it will be tuned.
2. Preferred behaviour for large events: one big box, or split into several? (Relevant to Phase 2A tiling; start with one box per connected region.)
3. How much of a delay is acceptable between the real start and the alert?

## Step 1: implement `src/trigger.py` (plain functions)

- `threshold_mask(score_map, threshold) -> bool (grid_h, grid_w)`
- `find_regions(mask, min_cells) -> list of boxes in GRID coordinates` using connected components (`cv2.connectedComponentsWithStats` or `scipy.ndimage.label`; say which). Drop regions smaller than `min_cells`.
- `grid_box_to_pixels(box, grid_h, grid_w, frame_shape) -> [x1,y1,x2,y2]`
- `update_events(open_events, regions_this_frame, frame_idx, ...) -> (open_events, finished_events)`: simple temporal logic. A region overlapping (IoU above a threshold) an open event extends it; a region with no match opens a new candidate; a candidate becomes an event only after `MIN_DURATION` frames (persistence); an open event ends after `MAX_GAP` frames without a match (hysteresis). Keep this as ONE readable function with comments; no state classes, just lists of dicts passed in and out.
- Events follow the contract in README section 5, with `"source": "motion"`.
- Smooth the event box over time (e.g. exponential average or union over the last few frames) and explain the choice.

## Step 2: script `scripts/p1d_events.py --video ... --camera ...`

Runs the Tier 1 chain end to end (flow, density, stats, score, trigger) on a video and writes `events.json` plus visuals.

## Visual outputs (`outputs/phase1d/`)

1. `<video>_events_overlay.mp4`: video with event boxes, event id, score and elapsed time drawn.
2. `<video>_events_timeline.png`: horizontal bars for each event over time, plus user-marked abnormal segments for comparison.
3. `<video>_event_keyframes.png`: for each event, 3 frames (start, middle, end) with the box.
4. `held_out_normal_events.png`: events fired on held-out normal footage, with count per hour printed.
5. `events.json`: list of events (contract format).

## What the user should check

- Do boxes cover the actual unusual activity, or only part of it, or too much background?
- Are events split/merged oddly (flicker, fragmentation)?
- Detection delay: time between the start of the real abnormal segment and the first event.
- Events/hour on normal footage. This is the load tier 2 must handle, so it matters for compute planning.

## Acceptance checklist

- [ ] Events produced with correct contract format
- [ ] Boxes and windows look right on the user-provided abnormal examples
- [ ] Events/hour on normal footage reported
- [ ] `config.py`: `MIN_EVENT_SECONDS`, `MAX_GAP_SECONDS`, `MIN_REGION_CELLS` set

STOP. Phase report, then wait for "approved phase 1D".
