#!/usr/bin/env python3
"""
CCTrans: Simplifying and Improving Crowd Counting with Transformer.
Usage:
    python cctrans.py <image_path> [--threshold 0.0] [--weight weights/cctrans_ShanghaiA.pth] [--output overlay.jpg]
"""

import os
import sys
import json
import argparse
from pathlib import Path

def _find_repo():
    possible_paths = [
        os.environ.get("CCTRANS_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_clones", "cctrans"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_clones", "cctrans"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctrans")
    ]
    for p in possible_paths:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "Networks")):
            return os.path.abspath(p)
    return None

def main():
    parser = argparse.ArgumentParser(description="CCTrans Crowd Counting")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Density cutoff threshold (default: 0.0)")
    parser.add_argument("--weight", default=None,
                        help="Path to pretrained CCTrans checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Path to save visual overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    weight_path = args.weight
    if not weight_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_p = os.path.join(script_dir, "weights", "cctrans_ShanghaiA.pth")
        if os.path.exists(default_p):
            weight_path = default_p
        else:
            # Per project instructions: record unavailable weights gracefully
            res = {
                "method": "CCTrans",
                "image": os.path.basename(args.image_path),
                "status": "skipped",
                "reason": "Pretrained weights are hosted exclusively on Baidu-Disk (requires Chinese mobile registration, code se59). Specify local weights via --weight if available."
            }
            print(json.dumps(res))
            sys.exit(0)

    repo_dir = _find_repo()
    if not repo_dir:
        res = {
            "method": "CCTrans",
            "image": os.path.basename(args.image_path),
            "status": "skipped",
            "reason": "CCTrans repository not found. Set CCTRANS_PATH or clone https://github.com/wfs123456/CCTrans."
        }
        print(json.dumps(res))
        sys.exit(0)

    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from Networks import ALTGVT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALTGVT.alt_gvt_large()
    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    img_raw = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = img_raw.size
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img_raw).unsqueeze(0).to(device)

    with torch.no_grad():
        dmap = model(tensor_img).squeeze().cpu().numpy()

    if args.threshold > 0:
        dmap_filtered = np.where(dmap >= args.threshold, dmap, 0.0)
    else:
        dmap_filtered = dmap
    count = float(np.sum(dmap_filtered))

    orig_bgr = cv2.imread(args.image_path)
    h, w = orig_bgr.shape[:2]
    d_norm = cv2.normalize(dmap_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d_resized = cv2.resize(d_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(d_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)
    cv2.putText(overlay, f"CCTrans: {count:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = args.output or str(Path(args.image_path).parent / f"{Path(args.image_path).stem}_CCTrans_overlay.jpg")
    cv2.imwrite(out_path, overlay)

    res = {
        "method": "CCTrans",
        "image": os.path.basename(args.image_path),
        "count": round(count, 2),
        "threshold": args.threshold,
        "overlay": out_path
    }
    print(json.dumps(res))

if __name__ == "__main__":
    main()
