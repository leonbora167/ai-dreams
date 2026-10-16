"""Prompt definitions and class descriptions for Stage 2 classifiers in Phase 2B.

Contains concrete visual descriptions, near-miss negative warnings,
lettered multiple-choice prompt templates for VLMs, and caption phrasings for CLIP.
"""

# Class taxonomy agreed for the stadium anomaly detection system
CLASSES = ["none", "fight", "running_surge", "fire_smoke"]

# Concrete visual definitions and near-miss guidance per class
CLASS_DEFINITIONS = {
    "none": (
        "Normal or benign activity. Includes pedestrians walking, standing, sitting, "
        "fans celebrating, cheering, jumping in excitement, hugging, or waving flags. "
        "Select this when no aggressive violence, panic running, or fire is present."
    ),
    "fight": (
        "Physical violence or aggressive brawl. People punching, hitting, shoving violently, "
        "kicking, grappling, or wrestling on the ground. "
        "DO NOT confuse with fans celebrating, hugging, or vigorous dancing."
    ),
    "running_surge": (
        "Crowd running rapidly, panic stampede, or sudden mass surge in one direction. "
        "People sprinting, fleeing, or pushing through bottlenecks in fear. "
        "DO NOT confuse with brisk walking or normal pedestrian movement."
    ),
    "fire_smoke": (
        "Visible open flames, burning objects, pyrotechnics, flares, or rising smoke plumes. "
        "DO NOT confuse with stadium floodlights, sun glare, steam, or white flags."
    )
}

# PROMPT_V1: Plain class list with standard definitions
PROMPT_V1 = """You are analyzing a CCTV surveillance video clip flagged by a motion detector.
Select the single option that best describes what is occurring in this video crop:

A: None (Normal crowd behavior, walking, standing, or benign fan celebration)
B: Fight (Physical altercation, punching, kicking, or aggressive wrestling)
C: Running Surge (Crowd running rapidly in panic, stampede, or mass rush)
D: Fire or Smoke (Flames, burning pyrotechnics, flares, or rising smoke plumes)

Respond with ONLY the single letter corresponding to the correct category: A, B, C, or D."""

# PROMPT_V2: Grouped guiding questions with explicit near-miss negatives (Default)
PROMPT_V2 = """You are an expert security surveillance operator reviewing a flagged stadium CCTV clip.
A region of interest has triggered a motion anomaly. Answer the following diagnostic questions:
1. Is anyone striking, punching, kicking, or violently grappling with someone else?
   (Note: Fans celebrating, jumping together, or hugging is NOT a fight.)
2. Is the crowd sprinting or running away rapidly in fear or panic?
   (Note: Normal walking or moving through aisles is NOT a running surge.)
3. Are there active flames, burning flares, or dense rising smoke plumes?
   (Note: Bright stadium floodlights or sun glare is NOT fire or smoke.)

Choose the single category that best matches:
A: None (Normal crowd behavior, peaceful gathering, or benign celebration)
B: Fight (Physical brawl, violence, hitting, or aggressive confrontation)
C: Running Surge (Sudden crowd running, panic evacuation, or stampede)
D: Fire or Smoke (Active fire, pyrotechnic flare, or developing smoke plume)

Answer with the single letter of your choice (A, B, C, or D):"""

# PROMPT_V3: Guiding questions augmented with Stage 1 physical motion telemetry
PROMPT_V3_TEMPLATE = """You are an expert security surveillance operator reviewing a flagged stadium CCTV clip.
Stage 1 Sensor Telemetry for this region:
- Velocity Anomaly Score: {z_score:.1f} (Z-score relative to normal baseline)
- Peak Motion Speed: {peak_speed:.1f} px/frame
- Motion Coherence: {coherence:.2f} (1.0 = unified flow, 0.0 = chaotic/turbulent)
- Crowd Density: {density_level}

Diagnostic Evaluation:
- If speed is high and motion is chaotic/turbulent with grappling -> Likely FIGHT.
- If speed is very high with unified directional flow away from a point -> Likely RUNNING SURGE.
- If visual appearance shows flames or rising haze -> Likely FIRE OR SMOKE.
- If motion is benign celebration or normal pedestrian traffic -> Select NONE.

Choose the single category that best matches:
A: None (Normal crowd behavior or benign celebration)
B: Fight (Physical brawl, violence, hitting, or aggressive confrontation)
C: Running Surge (Sudden crowd running, panic evacuation, or stampede)
D: Fire or Smoke (Active fire, pyrotechnic flare, or developing smoke plume)

Answer with the single letter of your choice (A, B, C, or D):"""

# Caption-style phrasings for CLIP-style zero-shot text embedding averaging
CLIP_CAPTIONS = {
    "none": [
        "a video of normal crowd walking peacefully",
        "people standing and talking in a stadium concourse",
        "fans celebrating, cheering, jumping, and hugging in excitement",
        "normal pedestrians walking along a sidewalk",
        "a peaceful crowd with no violence or emergency"
    ],
    "fight": [
        "a video of people violently fighting and punching",
        "two or more people in an aggressive physical altercation",
        "people hitting, kicking, and wrestling on the ground",
        "a violent brawl breaking out among individuals",
        "aggressive physical violence between people"
    ],
    "running_surge": [
        "a video of a crowd of people running in panic",
        "people sprinting and fleeing away in fear",
        "a sudden crowd surge and panic stampede",
        "people running rapidly across the area",
        "a crowd dispersing and running in emergency"
    ],
    "fire_smoke": [
        "a video of open flames and burning fire",
        "dense rising smoke plume from a fire",
        "a burning flare or pyrotechnic smoke grenade",
        "a fire with smoke burning outdoors",
        "flames and developing smoke hazard"
    ]
}


def get_prompt_text(variant: str = "PROMPT_V2", telemetry: dict = None) -> str:
    """Return formatted prompt string for the requested variant."""
    var = variant.upper()
    if var == "PROMPT_V1":
        return PROMPT_V1
    elif var == "PROMPT_V3":
        telem = telemetry or {"z_score": 5.0, "peak_speed": 12.0, "coherence": 0.4, "density_level": "medium"}
        return PROMPT_V3_TEMPLATE.format(**telem)
    return PROMPT_V2
