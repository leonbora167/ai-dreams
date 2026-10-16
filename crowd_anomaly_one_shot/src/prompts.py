"""Surveillance-tailored prompt definitions for the 7 target anomaly classes.

Unified captions for Google SigLIP (Tier 1) zero-shot vision-text matching
and diagnostic system/user prompts for SmolVLM2 (Tier 2).
"""

from typing import Dict, List

# Target Class Taxonomy
CLASSES = [
    "Normal",
    "Violence",
    "Fighting",
    "Arson",
    "Car Accident",
    "Explosion",
    "Shooting"
]

# SigLIP Zero-Shot Caption Bank (with hard negatives for high-accuracy discrimination)
SIGLIP_CAPTIONS: Dict[str, List[str]] = {
    "Normal": [
        "a surveillance video showing normal everyday pedestrians walking peacefully",
        "a security camera capturing normal crowd behavior with no violence or danger",
        "people walking, standing, sitting, or moving naturally in public",
        "a stationary CCTV view of a calm normal scene",
        "people running quickly, jogging, or rushing to catch a bus or train",
        "commuters walking briskly, carrying bags, backpacks, or pushing luggage carts",
        "people talking, gesturing with hands, laughing, and interacting naturally without fighting",
        "routine street traffic with cars, vans, and buses driving normally on the roadway",
        "indoor surveillance of routine hallway, store, or office activity with no emergency"
    ],
    "Violence": [
        "a surveillance video showing acts of physical violence, assault, and aggression",
        "a security camera capturing an angry mob, riot, or violent public disorder",
        "people committing aggressive violent acts, vandalism, or assaulting individuals",
        "a CCTV recording of violent civil unrest and physical altercation",
        "aggressive crowd disorder, rioters throwing objects, and public vandalism"
    ],
    "Fighting": [
        "a surveillance video showing people fighting, punching, and kicking each other",
        "a security camera capturing a physical brawl or fistfight between individuals",
        "two or more people engaged in a violent hand-to-hand fight and grappling",
        "a CCTV clip of a street fight with physical blows exchanged",
        "people wrestling aggressively on the ground, punching, and striking each other"
    ],
    "Arson": [
        "a surveillance video showing someone deliberately starting a fire or burning property",
        "a security camera capturing active arson, burning flames, and thick fire smoke",
        "a fire outbreak, burning building, flaming materials, or arsonist attack",
        "a CCTV recording of property on fire with visible smoke and flames",
        "thick smoke rising from fire, flames consuming objects, or structure ablaze"
    ],
    "Car Accident": [
        "a surveillance video showing a vehicle crash or traffic collision",
        "a security camera capturing a severe car accident involving motor vehicles",
        "automobiles colliding violently on the road or hitting pedestrians",
        "a CCTV recording of a traffic accident and vehicular wreckage",
        "cars crashing into each other, overturned vehicle, or road collision impact"
    ],
    "Explosion": [
        "a surveillance video showing a sudden violent explosion with blast shockwave",
        "a security camera capturing a detonating explosion and flying debris",
        "a bomb detonation, gas explosion, or sudden violent blast fireball",
        "a CCTV recording of a massive explosion destroying surroundings",
        "a fiery explosion with shockwave blast and smoke cloud erupting"
    ],
    "Shooting": [
        "a surveillance video showing a gunman shooting a firearm",
        "a security camera capturing an active shooter firing gunshots at people",
        "someone holding a gun, weapon discharge, and armed shooting incident",
        "a CCTV recording of a firearm shootout and people fleeing gunfire",
        "armed assailant pointing and discharging a gun at victims"
    ]
}

# SmolVLM2 Structured Diagnostic Prompt
SMOLVLM2_SYSTEM_INSTRUCTION = (
    "You are an expert CCTV surveillance security analyst inspecting video footage. "
    "Your duty is to accurately identify security anomalies. Analyze the visual scene carefully."
)

SMOLVLM2_USER_PROMPT = (
    "Analyze this surveillance camera frame. Classify the scene into exactly ONE category from this list:\n"
    "- Normal: peaceful everyday activity, walking, standing, sitting, vehicles moving normally, no threat.\n"
    "- Violence: mob riot, assault with weapons, vandalism, property destruction, civil unrest.\n"
    "- Fighting: physical fistfight, hand-to-hand combat, brawling, punching, kicking, wrestling.\n"
    "- Arson: intentional fire, burning objects, flames, smoke plumes.\n"
    "- Car Accident: vehicular collision, car crash, traffic accident, vehicle wreck.\n"
    "- Explosion: sudden explosion, bomb blast shockwave, fireball, flying debris.\n"
    "- Shooting: gunman holding a firearm, active shooter, weapon discharge, gunfire.\n\n"
    "Respond with ONLY the category name (e.g., 'Normal', 'Fighting', 'Arson', 'Car Accident', 'Explosion', 'Shooting', 'Violence'). Do not write any other words."
)

