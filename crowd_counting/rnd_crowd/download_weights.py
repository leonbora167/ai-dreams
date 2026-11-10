#!/usr/bin/env python3
"""
Download Pretrained Weights for P2PNet and APGCC.
Places downloaded model checkpoints directly into a weights directory.

Usage:
    python download_weights.py [--output_dir ./weights] [--model all|p2pnet|apgcc] [--force]
"""

import os
import sys
import argparse
from pathlib import Path

# Weight sources
WEIGHTS = {
    "p2pnet": {
        "filename": "p2pnet_SHTechA.pth",
        "type": "direct",
        "url": "https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/weights/SHTechA.pth",
        "expected_size_mb": 82.4
    },
    "apgcc": {
        "filename": "apgcc_SHHA_best.pth",
        "type": "gdrive",
        "gdrive_id": "1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu",
        "expected_size_mb": 71.1
    }
}


def download_stream(url, destination):
    import requests
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    print(f"\r  Downloading: {mb:.1f}/{tot_mb:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Downloading: {mb:.1f} MB", end="", flush=True)
    print()


def download_gdrive(file_id, destination):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
    except ImportError:
        import subprocess
        print("  gdown not found. Installing gdown via pip...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Download P2PNet & APGCC Pretrained Weights")
    parser.add_argument("--output_dir", default=None,
                        help="Target weights directory (default: ./weights next to script)")
    parser.add_argument("--model", choices=["all", "p2pnet", "apgcc"], default="all",
                        help="Which model weight to download (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if weight file already exists")
    args = parser.parse_args()

    # Default output dir is ./weights relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(script_dir, "weights")
    os.makedirs(out_dir, exist_ok=True)

    models_to_download = ["p2pnet", "apgcc"] if args.model == "all" else [args.model]

    print(f"Target weights directory: {os.path.abspath(out_dir)}")

    for key in models_to_download:
        info = WEIGHTS[key]
        dest = os.path.join(out_dir, info["filename"])

        if os.path.exists(dest) and not args.force:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            if size_mb > 10:  # Valid checkpoint file
                print(f"[{key.upper()}] Weight already exists: {dest} ({size_mb:.1f} MB). Skipping (use --force to overwrite).")
                continue

        print(f"[{key.upper()}] Downloading {info['filename']} (~{info['expected_size_mb']} MB)...")
        if info["type"] == "direct":
            download_stream(info["url"], dest)
        elif info["type"] == "gdrive":
            download_gdrive(info["gdrive_id"], dest)

        if os.path.exists(dest):
            final_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[{key.upper()}] Successfully saved to: {dest} ({final_mb:.1f} MB)\n")
        else:
            print(f"[{key.upper()}] Download failed for {info['filename']}\n", file=sys.stderr)

    print("All requested weights are ready in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
