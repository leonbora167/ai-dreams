# Phase 0: setup, repo skeleton, video IO, visual helpers

Goal: a working environment and two tiny helper modules (`video_io.py`, `viz.py`) that every later phase will use to LOOK at results. No models yet.

Prerequisite: read 00_README_START_HERE.md.

---

## Step 0: ask the user (all in one message)

1. Hardware you will run on right now (GPU model and VRAM, CPU, RAM, OS) and the intended edge target (if known).
2. Python version and whether you may create a virtualenv / conda env.
3. Do you have internet access in this environment for downloading weights and packages? If not, who downloads and where do they place files?
4. Sample videos. Please place in `data/videos/`:
   - `normal_*.mp4`: ordinary crowd, calm (ideally several minutes, from the SAME camera you will deploy on)
   - `running_*.mp4`: crowd with sudden running / surge / stopping, if available
   - `fight_*.mp4`: fighting/violence footage, if available
   - `fire_*.mp4` or images: fire/smoke samples, if available
5. Camera facts: resolution, original fps, roughly how large a person's head is in pixels in the typical view (rough guess is fine), fixed or PTZ camera.

Wait for answers. If some videos are missing, continue with what exists and note what is missing.

## Step 1: environment

- Create the env and install only: `numpy`, `opencv-python` (check if `opencv-contrib-python` is needed later), `matplotlib`, `torch` + `torchvision` (match the user's CUDA), `scikit-learn`, `tqdm`.
- Do not install model-specific libraries yet.
- Write `scripts/p0_check_env.py` that prints versions, whether CUDA is available, GPU name, and runs a tiny torch matmul on GPU and CPU. Show the output to the user.

## Step 2: skeleton

Create the layout from section 4 of the README, empty modules with docstrings, `config.py`, copies of MODELS.md and PROGRESS.md.

`config.py` initial constants (user can edit):
```python
ANALYSIS_FPS = 5            # frames per second used by the pipeline
ANALYSIS_WIDTH = 960        # frames are resized to this width, aspect ratio kept
GRID_H, GRID_W = 12, 20     # grid cells for tier 1 statistics
VIDEO_DIR = "data/videos"
OUT_DIR = "outputs"
```

## Step 3: video_io.py (plain functions)

- `get_video_info(path) -> dict` with width, height, fps, frame_count.
- `read_frames(path, target_fps, width, max_frames=None, start_sec=0)`: generator yielding `(frame_index, timestamp_sec, frame)` where frames are subsampled to `target_fps` and resized to `width`. Document how subsampling is done (pick every Nth source frame; say what N is for the user's video).
- `write_video(frames, path, fps)`: write a list/iterable of BGR frames to mp4.

## Step 4: viz.py (plain functions)

- `save_image(img, path)`
- `contact_sheet(images, cols, path, labels=None)`: grid of images with optional text labels
- `overlay_heatmap(frame, heat, alpha=0.5, vmax=None)`: colourised heatmap over frame; `heat` any 2D array, resized to frame size
- `draw_boxes(frame, boxes, labels=None, color=(0,255,0))`
- `draw_grid(frame, grid_h, grid_w)`: thin grid lines (used later for tier 1)
- `plot_timeline(values_dict, path, title)`: line plot of one or more series vs. time, saved as png

## Visual outputs (save to `outputs/phase0/`)

1. `frames_contact_sheet.png`: 12 evenly spaced frames from each sample video, with timestamps.
2. `grid_overlay_example.png`: one frame with the GRID_H x GRID_W grid drawn.
3. `subsample_check.mp4`: first 10 seconds written back via `write_video` at ANALYSIS_FPS.
4. `heatmap_demo.png`: fake heatmap (e.g. a gaussian blob) overlaid on a frame to prove the overlay works.

## What the user should check

- Do frames look right (orientation, colours not swapped, aspect ratio kept)?
- Is the grid size sensible for people size / scene? (Tell the user how many pixels a cell covers at the analysis resolution.)
- Is the analysis resolution large enough that people/heads are still visible? If not, discuss before moving on, because everything downstream depends on this.
- Does the subsampled video still look smooth enough for the events of interest?

## Acceptance checklist (user approves)

- [ ] Environment works, GPU check output reviewed
- [ ] Sample videos load and subsample correctly
- [ ] Visual outputs look correct
- [ ] ANALYSIS_FPS, ANALYSIS_WIDTH, GRID_H/W agreed

STOP. Write the phase report and wait for "approved phase 0".
