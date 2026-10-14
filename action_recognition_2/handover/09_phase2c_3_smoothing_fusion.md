# Phase 2C and Phase 3: temporal smoothing, fusion, alerts

Goal: turn raw per-clip predictions into stable alerts, combine all signals (motion events, action predictions, fire detections), and output an alert with a short clip and metadata. All rule-based. Two sub-phases with separate approvals: 2C, then 3.

Prerequisite: Phase 2B and Phase 1E approved (1E may be skipped if the user says so).

---

## Phase 2C: temporal smoothing

### Step 0: ask the user
- How many seconds of agreement before alerting is acceptable? (Trade-off: fewer false alarms vs delay.)
- Per-class thresholds: do some classes (e.g. fire) need lower thresholds than others?

### Implement `src/smoothing.py` (plain functions)
- `ema(scores, alpha) -> list`: exponential moving average of a probability series.
- `k_of_n(flags, k, n) -> bool`: true if at least k of the last n windows are positive.
- `smooth_event_predictions(window_preds, ...) -> dict`: for one event, run the action model on overlapping windows along the event (stride tunable), smooth per-class probabilities, and return a final decision with the time at which the alert would have fired.
- Keep tier 2 running in a loop over events, not per frame.

### Visual outputs (`outputs/phase2c/`)
1. `<event>_prob_timeline_raw_vs_smoothed.png`: per-class probabilities over the event, raw and smoothed, with the alert time marked.
2. `alert_delay_table.md`: for each event, time from event start to alert, and whether the final class is right (for events with known labels).
3. `smoothing_sweep.png`: false alarms on normal-footage events and detection delay for a few (k, n, alpha) settings.

### Acceptance
- [ ] Smoothing reduces flicker visibly
- [ ] Alert delay acceptable
- [ ] `config.py`: `EMA_ALPHA`, `K_OF_N`, `CLASS_THRESHOLDS` set

STOP. Report. Wait for "approved phase 2C".

---

## Phase 3: fusion and alert output

### Step 0: ask the user
- Alert format: JSON file, HTTP POST, MQTT, just logs? (Start with JSON files + short mp4 clip on disk; real transports later.)
- Which combinations should raise an alert? Suggested starting rules, the user edits:
  - action alert (class != normal, probability and smoothing satisfied) -> alert
  - fire event -> alert immediately (bypasses tier 2)
  - motion-only event (tier 1 fires, tier 2 says normal) -> no alert, but log as "suppressed" so the user can review
  - motion-only event with extreme score and tier 2 uncertain -> optional "review" alert (ask whether wanted)

### Implement `src/fusion.py`
- `fuse(motion_events, action_results, fire_events) -> list of alerts`: plain dicts `{"id", "type", "start_time", "end_time", "box", "confidence", "sources", "clip_path"}`
- `save_alert(alert, frames, out_dir)`: write JSON and a short clip with the box drawn.
- Keep all rules in one function with comments, so changing a rule is a one-line edit.

### Visual outputs (`outputs/phase3/`)
1. `<video>_alerts_overlay.mp4`: video with alert banners and boxes.
2. `alerts_summary.png`: timeline with motion events, action results, fire events, final alerts and suppressed events on separate rows.
3. `alert_examples/`: JSON + clip for each alert.

### Acceptance
- [ ] Rules reviewed and match the user's intent
- [ ] Suppressed events are logged and reviewable
- [ ] Alert files look right
- [ ] `config.py`: fusion options set

STOP. Report. Wait for "approved phase 3".
