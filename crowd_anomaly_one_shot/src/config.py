"""Core configuration parameters for crowd and video anomaly detection.

Unified settings supporting the 7 surveillance categories, Stage 1 DIS optical flow,
and the non-Chinese open-source AI cascade (Google SigLIP + SmolVLM2).
"""

import os

# Target Class Taxonomy (7 classes)
CLASSES = [
    "Normal",
    "Violence",
    "Fighting",
    "Arson",
    "Car Accident",
    "Explosion",
    "Shooting"
]

# Pipeline Processing Constants
ANALYSIS_FPS = 5.0
ANALYSIS_WIDTH = 640
CLIP_SIZE = 224
CLIP_FRAMES = 16

# Stage 1 Baseline & Gate Thresholds
LOW_THRESHOLD = 3.5      # Z-score for candidate opening / event escalation
HIGH_THRESHOLD = 6.0     # Z-score for direct anomaly
EXTREME_Z_THR = 15.0     # Z-score for immediate high-priority alert

# Model Architectures (Strictly Non-Chinese Open Source)
SIGLIP_BACKBONES = {
    "base": "google/siglip-base-patch16-224",         # Fast-pass base tier 1 (86M params, 224px input)
    "so400m": "google/siglip-so400m-patch14-384",    # High-accuracy tier 1 (400M params, 384px input)
    "large": "google/siglip-large-patch16-384"       # Large tier 1 (300M params, 384px input)
}

# Configurable active backbone: "base" | "so400m" | "large"
TIER1_BACKBONE_NAME = "so400m"
TIER1_MODEL_ID = SIGLIP_BACKBONES.get(TIER1_BACKBONE_NAME, "google/siglip-so400m-patch14-384")

# Multimodal VLM Tier 2
TIER2_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"    # Multimodal VLM tier 2 (2.2B params, US/France)

# SmolVLM2 Multi-Frame Configuration:
# - 1: Single static snapshot (middle frame of peak commotion)
# - 4: 4-frame temporal sequence (uniform sampling across event crop)
# - 16: Full 16 frames arranged into a 4x4 temporal storyboard grid (preserves context window)
SMOLVLM2_NUM_FRAMES = 16

# Tier 1 -> Tier 2 Escalation
CASCADE_ESCALATE_THR = 0.65   # If Tier 1 confidence < 0.65 or top is ambiguous, escalate to Tier 2
ALERT_CONF_THR = 0.50         # Threshold to confirm active anomaly alert

# Grid configuration for spatial motion tracking
GRID_H = 12
GRID_W = 20

# Directory Layout & Weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")


def is_valid_model_dir(dir_path: str) -> bool:
    """Check if a directory contains necessary Hugging Face model/tokenizer files."""
    if not os.path.isdir(dir_path):
        return False
    indicators = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "model.safetensors.index.json"
    ]
    return any(os.path.exists(os.path.join(dir_path, f)) for f in indicators)


def resolve_model_path(model_id: str, weights_dir: str = None) -> str:
    """Resolve a Hugging Face model ID to a local directory in weights/ if available.

    Checks:
    1. Direct path if model_id is already a local directory with configs.
    2. weights/<basename> (e.g. weights/siglip-so400m-patch14-384, weights/SmolVLM2-2.2B-Instruct)
    3. weights/<alias> (e.g. weights/so400m, weights/smolvlm2, weights/base, weights/large)
    4. weights/<model_id> (e.g. weights/google/siglip-so400m-patch14-384)
    5. weights/<model_id with / replaced by -- or _>
    6. Hugging Face hub snapshot cache directories.

    Returns:
        The local path if valid files exist; otherwise, returns original model_id for auto-download.
    """
    if not model_id:
        return model_id
    if os.path.isabs(model_id) or model_id.startswith("."):
        if is_valid_model_dir(model_id):
            return os.path.abspath(model_id)

    w_dir = weights_dir or WEIGHTS_DIR

    aliases = {
        "google/siglip-so400m-patch14-384": ["so400m", "siglip-so400m", "siglip-so400m-patch14-384"],
        "google/siglip-base-patch16-224": ["base", "siglip-base", "siglip-base-patch16-224"],
        "google/siglip-large-patch16-384": ["large", "siglip-large", "siglip-large-patch16-384"],
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct": ["smolvlm2", "SmolVLM2", "SmolVLM2-2.2B-Instruct", "smolvlm2-2.2b"],
        "openai/clip-vit-base-patch32": ["clip", "clip-vit-base-patch32"]
    }

    subnames = [
        model_id.split("/")[-1],
        model_id,
        model_id.replace("/", "--"),
        model_id.replace("/", "_"),
    ]
    if model_id in aliases:
        subnames.extend(aliases[model_id])

    # Check direct candidates in weights/
    for sub in subnames:
        cand = os.path.join(w_dir, sub)
        if is_valid_model_dir(cand):
            return os.path.abspath(cand)
        snapshots = os.path.join(cand, "snapshots")
        if os.path.isdir(snapshots):
            for snap in sorted(os.listdir(snapshots), reverse=True):
                snap_cand = os.path.join(snapshots, snap)
                if is_valid_model_dir(snap_cand):
                    return os.path.abspath(snap_cand)

    # Check HF hub cache folders inside weights/
    clean_id = model_id.replace('/', '--')
    hub_caches = [
        os.path.join(w_dir, "hub", f"models--{clean_id}"),
        os.path.join(w_dir, ".cache", "huggingface", "hub", f"models--{clean_id}")
    ]
    for hc in hub_caches:
        snapshots = os.path.join(hc, "snapshots")
        if os.path.isdir(snapshots):
            for snap in sorted(os.listdir(snapshots), reverse=True):
                snap_cand = os.path.join(snapshots, snap)
                if is_valid_model_dir(snap_cand):
                    return os.path.abspath(snap_cand)

    return model_id


