"""Global configuration constants for crowd anomaly detection."""

ANALYSIS_FPS = 5            # frames per second used by the pipeline
ANALYSIS_WIDTH = 960        # frames are resized to this width, aspect ratio kept
GRID_H, GRID_W = 12, 20     # default grid for stage 1 statistics (more scales added in Phase 1C)
VIDEO_DIR = "data/videos"
OUT_DIR = "outputs"
MOTION_METHOD = "dis"        # default motion estimation method ("dis" | "fastflownet" | "raft")
DENSITY_METHOD = "csrnet"    # default crowd density method ("csrnet" | "none")
DENSITY_EVERY_N_FRAMES = 1   # evaluate density every N analysis frames (1 = every frame)
DENSITY_WEIGHTS = "weights/csrnet_shanghaitech.pth"
GRID_SCALES = [(12, 20), (24, 40)]   # multi-scale grids for stage 1 (coarse, fine)
SCORE_SMOOTH_WINDOW = 3              # temporal smoothing window for score maps (frames)
NORMAL_DIR = "data/normal"           # directory for per-camera normal training footage

# Phase 1D Gate & Events parameters
LOW_THRESHOLD = 3.5                  # Z-score threshold for opening candidate / "confirm" level
HIGH_THRESHOLD = 6.0                 # Z-score threshold for immediate "direct" level
MIN_EVENT_SECONDS = 1.5              # minimum sustained duration (seconds) before counting as event
MAX_GAP_SECONDS = 1.0                # maximum inactivity gap (seconds) before closing open event
MIN_REGION_CELLS = 2                 # minimum contiguous cells in connected component to filter 1-cell noise

# Phase 1E Second Gate & Audit parameters
CONFIRM_MIN_SECONDS = 2.0            # sustained duration required to confirm low-level events
USE_APPEARANCE_CHECK = True          # evaluate visual appearance embedding check on confirm events
APPEARANCE_MODEL = "mobilenet"       # appearance embedding model ("mobilenet" | "clip")
APPEARANCE_DIST_THRESHOLD = 0.25     # cosine distance threshold relative to normal embeddings
AUDIT_RATE_PER_HOUR = 8.0            # random audit clips sampled per camera-hour
USE_SCHEDULE_RELAXATION = False      # match schedule event relaxation (disabled by user)
AUDIT_DIR = "outputs/phase1e/audit_queue"  # audit review queue directory

# Phase 1F Fire and Smoke parameters
FIRE_WEIGHTS = "weights/yolov8n_fire_smoke.pt"  # YOLO fire/smoke weights
FIRE_EVERY_N_FRAMES = 3                         # run fire detector every N analysis frames
FIRE_CONF_THR = 0.25                            # minimum confidence threshold for detection
FIRE_MIN_FRAMES = 3                             # minimum persistent detections to form an event

# Phase 2A ROI Crop & Clip parameters
CROP_MARGIN = 0.25                              # context margin expansion on each side (25%)
CLIP_FRAMES = 16                                # temporal frames sampled per event clip
CLIP_SIZE = 224                                 # model input resolution (S x S)
USE_TILING = True                               # enable tiling for large regions
MAX_TILES = 4                                   # maximum overlapping tiles for large regions

# Phase 2B Labelled Set & Prompt parameters
CLASSES = ["none", "fight", "running_surge", "fire_smoke"]
PROMPT_VARIANT = "PROMPT_V2"                   # default prompt strategy ("PROMPT_V1" | "PROMPT_V2" | "PROMPT_V3")
LABEL_SHEET = "outputs/phase2b/labeling_sheet.csv"

# Phase 2C Zero-shot Classifier parameters
ZEROSHOT_BACKEND = "cascade"                    # Stage 2 backend ("cascade" | "vlm" | "clip")
ZEROSHOT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"    # Primary VLM engine
CLIP_MODEL = "openai/clip-vit-base-patch32"      # Fast-pass tier 1 engine
ZEROSHOT_CONF_THR = 0.50                        # Confidence threshold (calibrated for 0 FA)
CASCADE_ESCALATE_THR = 0.65                     # Escalation threshold from Tier 1 to Tier 2

# Phase 3-A Temporal Smoothing & Alert Trigger parameters
EMA_ALPHA = 0.35                                # exponential moving average smoothing factor
K_OF_N = (3, 4)                                 # default requirement (at least k positive of last n windows)
CLASS_KN = {                                    # asymmetric per-class persistence requirements
    "fight": (3, 4),                            # high sustained agreement for fights
    "running_surge": (2, 3),                    # 2 of 3 windows for running surges
    "fire_smoke": (1, 1)                        # immediate alert for fire and smoke safety
}
CLASS_THRESHOLDS = {                            # calibrated per-class alert confidence thresholds
    "none": 0.50,
    "fight": 0.55,
    "running_surge": 0.50,
    "fire_smoke": 0.35                          # fast-track fire/smoke for safety
}
WINDOW_STRIDE_FRAMES = 2                        # stride in analysis frames between sliding windows (0.4s at 5 FPS)

# Phase 3-B Multi-Signal Fusion & Output parameters
EXTREME_Z_SCORE_THR = 15.0                      # Z-score threshold for uncertain extreme review alerts
ALERT_DIR = "outputs/phase3b/alert_examples"    # directory storing confirmed active alerts
REVIEW_DIR = "outputs/phase3b/review_queue"     # directory storing suppressed events and audit clips
