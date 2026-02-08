import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm
from torchvision.io import read_video

from crowd_action.features import CrowdDensityExtractor, RAFTFlowExtractor


def sample_clip(frames: torch.Tensor, frames_per_clip: int, image_size: int) -> torch.Tensor:
    if frames.shape[0] <= frames_per_clip:
        idx = np.linspace(0, max(0, frames.shape[0] - 1), frames_per_clip).astype(np.int64)
    else:
        idx = np.linspace(0, frames.shape[0] - 1, frames_per_clip).astype(np.int64)
    clip = frames[idx]
    transform = T.Compose(
        [
            T.ToImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
        ]
    )
    return torch.stack([transform(frame) for frame in clip], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    raft = RAFTFlowExtractor(device=device)
    crowd = CrowdDensityExtractor(device=device)

    df = pd.read_csv(args.manifest)
    rows = df.to_dict(orient="records")
    total = len(rows)
    processed = 0
    saved = 0
    skipped_existing = 0
    skipped_short = 0
    failed = 0

    for row in tqdm(rows):
        t0 = time.perf_counter()
        video_id = row["video_id"]
        target = out_dir / f"{video_id}.npz"
        processed += 1
        if target.exists():
            skipped_existing += 1
            continue

        try:
            frames, _, _ = read_video(row["video_path"], pts_unit="sec", output_format="TCHW")
        except Exception:
            failed += 1
            continue
        if frames.shape[0] < 2:
            skipped_short += 1
            continue
        clip = sample_clip(frames, args.frames_per_clip, args.image_size)
        flow = raft.compute(clip).numpy()
        density = crowd.compute(clip).numpy()
        np.savez_compressed(target, flow=flow, crowd=density)
        saved += 1
        payload = {
            "total": total,
            "processed": processed,
            "saved": saved,
            "skipped_existing": skipped_existing,
            "skipped_short": skipped_short,
            "failed": failed,
            "current_video_id": video_id,
            "elapsed_sec_last_video": round(time.perf_counter() - t0, 3),
        }
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"aux files written to {out_dir}")


if __name__ == "__main__":
    main()
