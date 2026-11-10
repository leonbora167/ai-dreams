#!/usr/bin/env python3
"""
LWCC: CSRNet and DM-Count Crowd Counting.
Standalone script for running inference on an image.
Usage:
    python lwcc.py <image_path> [--model CSRNet|DM-Count] [--threshold 0.0] [--output overlay.jpg]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Prevent filename shadowing installed package
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and sys.path[0] == _script_dir:
    sys.path.pop(0)

# Safe patch for LWCC ~/.lwcc directory on macOS and Linux
import lwcc.util.functions as lwcc_func

def _safe_weights_check(model_name, model_weights):
    weights_dir = os.path.join(str(Path.home()), ".lwcc", "weights")
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    file_name = f"{model_name}_{model_weights}.pth"
    url = lwcc_func.build_url(file_name)
    output = os.path.join(weights_dir, file_name)
    if not os.path.isfile(output):
        import gdown
        print(f"Downloading {file_name} to {output}...", file=sys.stderr)
        gdown.download(url, output, quiet=False)
    return output

lwcc_func.weights_check = _safe_weights_check

from lwcc import LWCC
import cv2
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="LWCC Crowd Counting (CSRNet / DM-Count)")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", choices=["CSRNet", "DM-Count"], default="CSRNet",
                        help="Model architecture (default: CSRNet)")
    parser.add_argument("--weights", default="SHA", help="Dataset weights (default: SHA)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Density cutoff threshold to suppress background false positives (default: 0.0)")
    parser.add_argument("--output", default=None, help="Path to save visual overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    raw_count, density = LWCC.get_count(
        args.image_path,
        model_name=args.model,
        model_weights=args.weights,
        return_density=True
    )

    if args.threshold > 0:
        filtered_density = np.where(density >= args.threshold, density, 0.0)
        final_count = float(np.sum(filtered_density))
    else:
        filtered_density = density
        final_count = float(raw_count)

    # Save visual overlay
    orig_bgr = cv2.imread(args.image_path)
    h, w = orig_bgr.shape[:2]
    d_norm = cv2.normalize(filtered_density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d_resized = cv2.resize(d_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(d_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

    # Draw count text on image
    cv2.putText(overlay, f"{args.model}: {final_count:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = args.output
    if not out_path:
        stem = Path(args.image_path).stem
        out_dir = Path(args.image_path).parent
        out_path = str(out_dir / f"{stem}_{args.model}_overlay.jpg")

    cv2.imwrite(out_path, overlay)

    res = {
        "method": f"LWCC_{args.model}",
        "image": os.path.basename(args.image_path),
        "count": round(final_count, 2),
        "raw_count": round(float(raw_count), 2),
        "threshold": args.threshold,
        "overlay": out_path
    }
    print(json.dumps(res))

if __name__ == "__main__":
    main()
