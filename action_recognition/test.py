import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

from crowd_action.config import load_config
from crowd_action.features import CrowdDensityExtractor, RAFTFlowExtractor
from crowd_action.models import MultiModalSwinClassifier


def resolve_class_names(config_path: str, checkpoint_path: str, explicit: str | None) -> list[str]:
    if explicit:
        names = [x.strip() for x in explicit.split(",") if x.strip()]
        if names:
            return names

    summary_path = Path(checkpoint_path).parent / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        names = summary.get("class_names", [])
        if names:
            return names

    cfg = load_config(config_path)
    if cfg.data.class_names:
        return cfg.data.class_names
    if Path(cfg.data.manifest_path).exists():
        df = pd.read_csv(cfg.data.manifest_path)
        inferred = sorted(df["label"].dropna().astype(str).unique().tolist())
        if inferred:
            return inferred
    raise ValueError("Could not resolve class names. Provide --class-names explicitly.")


def build_rgb_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        ]
    )


def build_aux_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
        ]
    )


def load_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    return frames, float(fps)


def window_starts(num_frames: int, frames_per_clip: int, stride: int) -> list[int]:
    if num_frames <= frames_per_clip:
        return [0]
    starts = list(range(0, num_frames - frames_per_clip + 1, max(1, stride)))
    last = num_frames - frames_per_clip
    if starts[-1] != last:
        starts.append(last)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_example.yaml")
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class names")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = resolve_class_names(args.config, args.checkpoint, args.class_names)
    num_classes = len(class_names)
    model_num_classes = cfg.model.num_classes or num_classes
    if model_num_classes != num_classes:
        raise ValueError(
            f"num_classes mismatch: config/model={model_num_classes}, inferred labels={num_classes}"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MultiModalSwinClassifier(
        num_classes=model_num_classes,
        dropout=cfg.model.dropout,
        flow_weight=cfg.model.flow_weight,
        crowd_weight=cfg.model.crowd_weight,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    rgb_transform = build_rgb_transform(cfg.data.image_size)
    aux_transform = build_aux_transform(cfg.data.image_size)
    raft = RAFTFlowExtractor(device=device)
    crowd = CrowdDensityExtractor(device=device)

    frames_bgr, fps = load_video_frames(args.video)
    num_frames = len(frames_bgr)
    starts = window_starts(num_frames, cfg.data.frames_per_clip, args.stride)

    probs_sum = torch.zeros((num_frames, num_classes), dtype=torch.float32)
    probs_count = torch.zeros((num_frames,), dtype=torch.float32)

    pbar = tqdm(starts, desc="inferring clips", dynamic_ncols=True)
    for start in pbar:
        end = min(start + cfg.data.frames_per_clip, num_frames)
        clip_bgr = frames_bgr[start:end]
        while len(clip_bgr) < cfg.data.frames_per_clip:
            clip_bgr.append(clip_bgr[-1])

        clip_rgb_u8 = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in clip_bgr]
        clip_tensor_u8 = torch.stack(
            [torch.from_numpy(frame).permute(2, 0, 1) for frame in clip_rgb_u8], dim=0
        )

        rgb_clip = torch.stack([rgb_transform(frame) for frame in clip_tensor_u8], dim=0)
        aux_clip = torch.stack([aux_transform(frame) for frame in clip_tensor_u8], dim=0)
        flow = raft.compute(aux_clip)
        crowd_map = crowd.compute(aux_clip)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(
                    rgb=rgb_clip.unsqueeze(0).to(device),
                    flow=flow.unsqueeze(0).to(device),
                    crowd=crowd_map.unsqueeze(0).to(device),
                )
                probs = torch.softmax(logits[0], dim=0).cpu()

        for idx in range(start, end):
            probs_sum[idx] += probs
            probs_count[idx] += 1.0

    default_probs = probs_sum.sum(dim=0)
    if float(default_probs.sum()) <= 0:
        default_probs = torch.ones((num_classes,), dtype=torch.float32) / num_classes
    else:
        default_probs = default_probs / default_probs.sum()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output writer: {output_path}")

    for i, frame in enumerate(tqdm(frames_bgr, desc="writing output", dynamic_ncols=True)):
        if probs_count[i] > 0:
            probs = probs_sum[i] / probs_count[i]
        else:
            probs = default_probs
        conf, pred_idx = torch.max(probs, dim=0)
        label = class_names[int(pred_idx)]
        text = f"{label} ({float(conf):.2f})"
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
