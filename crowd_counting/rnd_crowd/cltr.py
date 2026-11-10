#!/usr/bin/env python3
"""
CLTR: Crowd Localization Transformer.
Runs transformer-based crowd localization & counting inference on an image.
Usage:
    python cltr.py <image_path> [--threshold 0.35] [--weight weights/cltr_nwpu.pth] [--output overlay.jpg]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# Locate or download CLTR repository modules
def _ensure_cltr_repo():
    possible_paths = [
        os.environ.get("CLTR_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_clones", "cltr"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_clones", "cltr"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cltr"),
        os.path.expanduser("~/.cache/cltr")
    ]
    for p in possible_paths:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "Networks", "CDETR")):
            return os.path.abspath(p)

    # Auto-clone to ~/.cache/cltr if running standalone on another server
    cache_dir = os.path.expanduser("~/.cache/cltr")
    print(f"CLTR repository not found locally. Cloning to {cache_dir}...", file=sys.stderr)
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/dk-liang/CLTR.git", cache_dir])
    return cache_dir

cltr_root = _ensure_cltr_repo()
if cltr_root not in sys.path:
    sys.path.insert(0, cltr_root)

# Temporarily isolate sys.argv while importing config to prevent it from auto-parsing command line
_orig_argv = sys.argv
sys.argv = [_orig_argv[0]]
from Networks.CDETR import build_model
from config import return_args
sys.argv = _orig_argv


def main():
    parser = argparse.ArgumentParser(description="CLTR Crowd Counting & Localization")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Confidence score threshold for detections (default: 0.35)")
    parser.add_argument("--weight", default=None,
                        help="Path to pretrained CLTR checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Path to save visual overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    weight_path = args.weight
    if not weight_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_p = os.path.join(script_dir, "weights", "cltr_nwpu.pth")
        if os.path.exists(default_p):
            weight_path = default_p
        else:
            print(json.dumps({"error": f"Weight file not found at {default_p}. Specify with --weight."}))
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure CLTR args for NWPU video model checkpoint
    return_args.device = str(device)
    return_args.dataset_file = 'crowd_data'
    return_args.channel_point = 3
    return_args.num_queries = 700

    model, criterion, postprocessors = build_model(return_args)
    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    new_state = OrderedDict()
    for k, v in state.items():
        name = k.replace('module.', '').replace('bbox', 'point')
        new_state[name] = v

    model.load_state_dict(new_state, strict=True)
    model.to(device)
    model.eval()

    img_raw = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = img_raw.size

    # Round size to multiple of 256 for patch inference
    nw = max(256, (orig_w // 256) * 256)
    nh = max(256, (orig_h // 256) * 256)
    img_resized = img_raw.resize((nw, nh), Image.Resampling.LANCZOS)

    tensor_transform = T.ToTensor()
    img_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = img_transform(tensor_transform(img_resized))

    num_w = nw // 256
    num_h = nh // 256
    patches = img_tensor.view(3, num_h, 256, nw).view(3, num_h, 256, num_w, 256)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)
    patches = patches.to(device)

    with torch.no_grad():
        outputs = model(patches)

    out_logits = outputs['pred_logits']
    out_point = outputs['pred_points']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), return_args.num_queries, dim=1)

    topk_points = topk_indexes // out_logits.shape[2]
    out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
    out_point = out_point * 256  # scale within 256x256 patch

    # Map patch coordinates to global image coordinates
    all_points = []
    crop_size = 256
    scale_x = orig_w / float(nw)
    scale_y = orig_h / float(nh)

    for patch_idx in range(patches.shape[0]):
        row_idx = patch_idx // num_w
        col_idx = patch_idx % num_w
        base_x = col_idx * crop_size
        base_y = row_idx * crop_size

        vals = topk_values[patch_idx].detach().cpu().numpy()
        pts = out_point[patch_idx].detach().cpu().numpy()

        for j in range(len(vals)):
            if vals[j] >= args.threshold:
                px = (base_x + pts[j][0]) * scale_x
                py = (base_y + pts[j][1]) * scale_y
                all_points.append((px, py))

    count = len(all_points)

    # Render visual overlay
    orig_bgr = cv2.imread(args.image_path)
    dot_radius = max(2, int(min(orig_w, orig_h) / 250))
    for pt in all_points:
        px = int(pt[0])
        py = int(pt[1])
        cv2.circle(orig_bgr, (px, py), dot_radius, (255, 0, 255), -1)
        cv2.circle(orig_bgr, (px, py), dot_radius + 1, (255, 255, 255), 1)

    cv2.putText(orig_bgr, f"CLTR: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = args.output
    if not out_path:
        stem = Path(args.image_path).stem
        out_dir = Path(args.image_path).parent
        out_path = str(out_dir / f"{stem}_CLTR_overlay.jpg")

    cv2.imwrite(out_path, orig_bgr)

    res = {
        "method": "CLTR",
        "image": os.path.basename(args.image_path),
        "count": count,
        "threshold": args.threshold,
        "overlay": out_path
    }
    print(json.dumps(res))

if __name__ == "__main__":
    main()
