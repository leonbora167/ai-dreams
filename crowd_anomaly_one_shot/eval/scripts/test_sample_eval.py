import os
import cv2
import torch
import time
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import eval.config_eval as config_eval
import eval.models_eval as models_eval
import eval.prompts_eval as prompts_eval

df = pd.read_csv("eval/data/manifest.csv")
sample_rows = []
for d in ["xd_violence", "ucf_crime"]:
    for c in ["Car Accident", "Explosion", "Fighting", "Shooting", "Violence", "Normal", "Arson"]:
        sub = df[(df["dataset"] == d) & (df["ground_truth"] == c)]
        if len(sub) > 0:
            sample_rows.append(sub.iloc[0])

print(f"Selected {len(sample_rows)} test clips across both datasets.")

print("Loading Tier 1 (SigLIP)...")
t1_bundle = models_eval.load_tier1("cuda")
print("Loading Tier 2 (SmolVLM2)...")
t2_bundle = models_eval.load_tier2("cuda")
print("Models loaded.\n")

for r in sample_rows:
    p = r["full_path"]
    gt = r["ground_truth"]
    dname = r["dataset"]
    vname = os.path.basename(p)
    cap = cv2.VideoCapture(p)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Read 30-50 frames from the middle or active section
    start_f = max(0, total_f // 3)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(50):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        continue

    t0 = time.time()
    t1_res = models_eval.classify_tier1(t1_bundle[0], t1_bundle[1], t1_bundle[2], frames, device="cuda")
    t2_res = models_eval.classify_tier2(t2_bundle[0], t2_bundle[1], frames, device="cuda")
    dt = time.time() - t0

    t1_top = t1_res["top"]
    t1_conf = t1_res["top_prob"]
    t2_top = t2_res["top"]
    raw = t2_res["raw_response"].replace("\n", " ")[:20]

    print(f"[{dname:<11}] GT: {gt:<13} | SigLIP: {t1_top:<12} ({t1_conf*100:4.1f}%) | SmolVLM2: {t2_top:<12} (raw: {raw:<20}) | {dt*1000:4.0f}ms")
