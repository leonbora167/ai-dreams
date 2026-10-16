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

