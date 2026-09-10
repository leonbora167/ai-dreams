# MTMC tracker POC

For a plain-language and technical overview of the processing flow, see
[`TRACKER_GUIDE.md`](TRACKER_GUIDE.md).

This is a local, one-process, config-driven multi-camera pedestrian tracker.
The implementation follows `mtmc_tracker_spec.md`; the spec describes the
target behavior, while this README describes how to operate the repository.

## Run

Use the preconfigured environment and run a no-cost configuration check first:

```bash
conda activate tracker-env
python run.py --config configs/poc.yaml --videos ./data/videos --dry-run
```

Put camera videos in `data/videos` as `cam01.mp4`, `cam02.mp4`, etc. Tracklet
JSON is cached under `outputs/tracks`; deleting one cache file forces only that
camera's detection/tracking stage to be recomputed.

## Smallest ready-to-download smoke test

The recommended minimal dataset is the official EPFL 4-person laboratory
sequence: four synchronized cameras, four people, and about 2.5 minutes of
video. The views overlap, so use it to validate detection, local tracking,
ReID, and visualization—not non-overlapping transition gating. EPFL provides
the videos for research use and identifies the sequence as four cameras with
four people entering and walking indoors.

Download and convert it yourself with:

```bash
conda activate tracker-env
python scripts/download_poc_dataset.py
python run.py --config configs/epfl_4p.yaml --videos ./data/videos --dry-run
python run.py --config configs/epfl_4p.yaml --videos ./data/videos
```

The helper downloads the four official AVI files, converts them to
`cam01.mp4`–`cam04.mp4`, and removes the intermediate AVI files unless
`--keep-avi` is supplied.

## Incremental outputs

Run live-like processing with `--incremental`. It writes:

- `outputs/.../observations/cam01.jsonl`: one human-readable JSON record per detection, containing `camera_id`, `frame_id`, timestamp, bounding box, confidence, and local track ID.
- `outputs/.../tracks/cam01.json`: finalized tracklets with frame lists, bounding boxes, local IDs, and global IDs.
- `outputs/.../global_id_map.json`: `(camera, local_track_id)` to `global_id` mapping.
- `outputs/.../features/cam01_12.npz`: NumPy archive for one tracklet. It contains `reid` (typically a 512-D float vector), optional `color`/`motion` arrays, and metadata fields `camera_id`, `local_track_id`, and `global_id`.

To render later without rerunning inference:

```bash
python scripts/visualize_outputs.py --config configs/epfl_4p.yaml --videos ./data/videos
```

The actual inference/render command is intentionally left for the operator:

```bash
python run.py --config configs/poc.yaml --videos ./data/videos
```

For the recommended ReID checkpoint, install the small downloader dependency
and run this explicitly before using `configs/epfl_4p.yaml`:

```bash
conda activate tracker-env
pip install gdown
python scripts/download_reid_weights.py
```

Toggle feature modules only in `configs/poc.yaml`. ReID and video decoding are
loaded lazily; `--dry-run` does not load model weights.
