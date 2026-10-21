"""Standalone downloader for pipeline AI weights into the local 'weights/' directory.

Allows downloading models on a connected machine (local laptop, workstation, cloud VM)
and copying the resulting folders into 'weights/' on offline or firewall-restricted servers.

Supported Models:
- so400m    : google/siglip-so400m-patch14-384 (Default Tier 1, ~1.6 GB)
- smolvlm2  : HuggingFaceTB/SmolVLM2-2.2B-Instruct (Default Tier 2, ~4.5 GB)
- base      : google/siglip-base-patch16-224 (Edge Tier 1, ~350 MB)
- large     : google/siglip-large-patch16-384 (Large Tier 1, ~1.2 GB)
- clip      : openai/clip-vit-base-patch32 (~600 MB)
- all       : Downloads both default models (so400m + smolvlm2)
- Any custom Hugging Face repo ID (e.g. meta-llama/Llama-3.2-11B-Vision-Instruct)
"""

import os
import sys
import argparse
import shutil
from typing import List, Dict

# Model catalog mapping shorthands to official Hugging Face repository IDs and default local folder names
MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "so400m": {
        "repo_id": "google/siglip-so400m-patch14-384",
        "folder": "siglip-so400m-patch14-384",
        "description": "Google SigLIP SO400M (Default Tier 1 Classifier, 384x384)"
    },
    "smolvlm2": {
        "repo_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "folder": "SmolVLM2-2.2B-Instruct",
        "description": "SmolVLM2 2.2B Instruct (Default Tier 2 Multimodal VLM)"
    },
    "base": {
        "repo_id": "google/siglip-base-patch16-224",
        "folder": "siglip-base-patch16-224",
        "description": "Google SigLIP Base (Edge Tier 1 Classifier, 224x224)"
    },
    "large": {
        "repo_id": "google/siglip-large-patch16-384",
        "folder": "siglip-large-patch16-384",
        "description": "Google SigLIP Large (Large Tier 1 Classifier, 384x384)"
    },
    "clip": {
        "repo_id": "openai/clip-vit-base-patch32",
        "folder": "clip-vit-base-patch32",
        "description": "OpenAI ViT-B/32 CLIP (Legacy Appearance Extractor)"
    }
}


def get_dir_size_str(path: str) -> str:
    """Calculate human-readable total directory size."""
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp) and os.path.exists(fp):
                total_bytes += os.path.getsize(fp)
    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


def download_model(repo_id: str, target_dir: str, token: str = None) -> bool:
    """Download a Hugging Face repository snapshot directly into target_dir."""
    print(f"\n=======================================================")
    print(f"Downloading: {repo_id}")
    print(f"Destination: {target_dir}")
    print(f"=======================================================")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: 'huggingface_hub' is required. Please install via: pip install huggingface_hub")
        return False

    os.makedirs(target_dir, exist_ok=True)

    try:
        # snapshot_download downloads full repository files without symlinks for portability
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]  # Skip unnecessary non-PyTorch formats
        )

        size_str = get_dir_size_str(target_dir)
        print(f"\nSUCCESS: Downloaded '{repo_id}' ({size_str})")
        print(f"Saved at: {target_dir}")
        return True

    except Exception as e:
        print(f"\nFAILED downloading '{repo_id}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pipeline AI model weights to 'weights/' directory.")
    parser.add_argument("--models", "-m", nargs="+", default=["so400m", "smolvlm2"],
                        help="Models to download ('so400m', 'smolvlm2', 'base', 'large', 'clip', 'all', or any HF repo ID).")
    parser.add_argument("--weights_dir", "-w", default=None,
                        help="Target weights directory (defaults to '<project_root>/weights').")
    parser.add_argument("--token", "-t", default=None,
                        help="Optional Hugging Face authentication token for gated models.")
    args = parser.parse_args()

    # Determine default weights directory
    if args.weights_dir:
        weights_dir = os.path.abspath(args.weights_dir)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If in eval/scripts/, project root is ../..
        if "eval" in current_dir:
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        else:
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
        weights_dir = os.path.join(project_root, "weights")

    os.makedirs(weights_dir, exist_ok=True)
    print(f"Target weights directory: {weights_dir}")

    # Expand 'all' selection
    requested_models = args.models
    if "all" in requested_models:
        requested_models = ["so400m", "smolvlm2"]

    downloaded = 0
    failed = 0

    for item in requested_models:
        key = item.lower().strip()
        if key in MODEL_CATALOG:
            cat = MODEL_CATALOG[key]
            repo_id = cat["repo_id"]
            folder_name = cat["folder"]
        else:
            # Custom repo ID provided (e.g. "google/siglip-so400m-patch14-384" or "meta-llama/...")
            repo_id = item
            folder_name = item.split("/")[-1]

        target_path = os.path.join(weights_dir, folder_name)
        ok = download_model(repo_id=repo_id, target_dir=target_path, token=args.token)
        if ok:
            downloaded += 1
        else:
            failed += 1

    print("\n=======================================================")
    print(f"Summary: {downloaded} models downloaded, {failed} failed.")
    print(f"Weights folder location: {weights_dir}")
    print("=======================================================")
    print("\nHow offline loading works:")
    print("The pipeline automatically checks 'weights/' for these folders.")
    print("If you are transferring to a server without internet access:")
    print(f"Simply copy the folder(s) into your server's 'weights/' directory.")
    print("The pipeline will automatically detect and load them with local_files_only=True.\n")


if __name__ == "__main__":
    main()

