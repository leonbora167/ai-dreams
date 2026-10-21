"""Model loader and inference routines for SigLIP (Tier 1) and SmolVLM2 (Tier 2)."""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

# Ensure Hugging Face backend skips SSL checks on servers behind proxy firewalls
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
try:
    import urllib3
    urllib3.disable_warnings()
    import requests
    from huggingface_hub import configure_http_backend
    configure_http_backend(backend_factory=lambda: (lambda s: (setattr(s, "verify", False), s)[1])(requests.Session()))
except Exception:
    pass

from transformers import AutoProcessor, AutoModel, AutoModelForImageTextToText

import eval.config_eval as config_eval
import eval.prompts_eval as prompts_eval


def load_tier1(model_id: str = None, device: str = "cuda") -> Tuple[object, object, torch.Tensor]:
    """Load Google SigLIP model, processor, and pre-computed text embeddings.
    
    Checks local 'weights/' directory first. If present, loads entirely offline.
    If not present locally, falls back to automatic download from Hugging Face.
    """
    if model_id is None:
        model_id = config_eval.TIER1_MODEL_ID

    resolved_path = config_eval.resolve_model_path(model_id)
    is_local = os.path.isdir(resolved_path)

    if is_local:
        print(f"Loading Tier 1 (SigLIP) from local directory: {resolved_path}")
        try:
            processor = AutoProcessor.from_pretrained(resolved_path, local_files_only=True)
            model = AutoModel.from_pretrained(resolved_path, torch_dtype=torch.float16, local_files_only=True).to(device)
        except Exception as e:
            print(f"Warning: Failed loading local files ({e}). Falling back to Hugging Face Hub...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    else:
        print(f"Local weights not found for '{model_id}' in {config_eval.WEIGHTS_DIR}. Attempting auto-download from Hugging Face...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    model.eval()

    # Precompute mean normalized text embeddings for each of the 7 classes
    class_embs = []
    with torch.no_grad():
        for c in prompts_eval.CLASSES:
            caps = prompts_eval.SIGLIP_CAPTIONS[c]
            inputs = processor(text=caps, padding="max_length", return_tensors="pt").to(device)
            txt_feats = model.get_text_features(**inputs)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            mean_feat = txt_feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            class_embs.append(mean_feat)

    all_txt_embs = torch.cat(class_embs, dim=0)  # [7, D]
    return model, processor, all_txt_embs


def classify_tier1(model, processor, text_embs: torch.Tensor,
                   frames: List[np.ndarray], device: str = "cuda") -> Dict:
    """Classify video clip using Google SigLIP zero-shot vision-text matching."""
    t0 = time.perf_counter()
    # Sample up to 8 evenly spaced frames from the clip
    step = max(1, len(frames) // 8)
    sampled = [Image.fromarray(frames[i]) for i in range(0, len(frames), step)][:8]

    inputs = processor(images=sampled, return_tensors="pt").to(device)
    # Ensure inputs are in model dtype (half)
    if hasattr(inputs, "pixel_values") and inputs.pixel_values.dtype != torch.float16:
        inputs.pixel_values = inputs.pixel_values.to(torch.float16)

    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # Average image features across the clip
        clip_feat = img_feats.mean(dim=0, keepdim=True)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity to the 7 class text embeddings with standard temperature
        logits = torch.matmul(clip_feat, text_embs.t())[0] * 80.0  # optimal temperature scaling
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    top_idx = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "top": prompts_eval.CLASSES[top_idx],
        "top_prob": float(probs[top_idx]),
        "probs": {c: float(p) for c, p in zip(prompts_eval.CLASSES, probs)},
        "latency_ms": latency_ms,
        "model": getattr(model, "name_or_path", config_eval.TIER1_MODEL_ID)
    }


def load_tier2(model_id: str = None, device: str = "cuda") -> Tuple[object, object]:
    """Load HuggingFace SmolVLM2-2.2B-Instruct multimodal model.
    
    Checks local 'weights/' directory first. If present, loads entirely offline.
    If not present locally, falls back to automatic download from Hugging Face.
    """
    if model_id is None:
        model_id = config_eval.TIER2_MODEL_ID

    resolved_path = config_eval.resolve_model_path(model_id)
    is_local = os.path.isdir(resolved_path)

    if is_local:
        print(f"Loading Tier 2 (SmolVLM2) from local directory: {resolved_path}")
        try:
            processor = AutoProcessor.from_pretrained(resolved_path, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(
                resolved_path,
                dtype=torch.float16,
                _attn_implementation="sdpa",
                local_files_only=True
            ).to(device)
        except Exception as e:
            print(f"Warning: Failed loading local files ({e}). Falling back to Hugging Face Hub...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.float16,
                _attn_implementation="sdpa"
            ).to(device)
    else:
        print(f"Local weights not found for '{model_id}' in {config_eval.WEIGHTS_DIR}. Attempting auto-download from Hugging Face...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            _attn_implementation="sdpa"
        ).to(device)

    model.eval()
    return model, processor


def classify_tier2(model, processor, frames: List[np.ndarray], device: str = "cuda",
                   num_frames: int = config_eval.SMOLVLM2_NUM_FRAMES) -> Dict:
    """Classify video clip using SmolVLM2 generative multimodal comprehension.
    
    Supports:
    - num_frames = 1: Single static snapshot (center frame of commotion)
    - 2 <= num_frames <= 6: Multi-frame temporal sequence sampled uniformly across event crop
    - num_frames > 6 (e.g. 16): 4x4 temporal storyboard grid of all 16 frames (shows all 16 frames without context overflow)
    """
    t0 = time.perf_counter()
    import cv2

    if not frames:
        return {"top": "Normal", "raw_response": "No frames", "latency_ms": 0.0, "model": config_eval.TIER2_MODEL_ID}

    # Case 1: Single static snapshot
    if num_frames == 1 or len(frames) == 1:
        mid_idx = len(frames) // 2
        f = cv2.resize(frames[mid_idx], (224, 224))
        sampled = [Image.fromarray(f)]
        prompt_text = prompts_eval.SMOLVLM2_USER_PROMPT
    # Case 2: Multi-frame sequence (2 to 6 frames)
    elif 2 <= num_frames <= 6:
        indices = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)), dtype=int)
        sampled = [Image.fromarray(cv2.resize(frames[idx], (224, 224))) for idx in indices]
        prompt_text = f"Analyze this {len(sampled)}-frame surveillance video sequence.\n" + prompts_eval.SMOLVLM2_USER_PROMPT
    # Case 3: 16-frame temporal storyboard grid (or > 6 frames)
    else:
        # Arrange 16 frames into a 4x4 temporal grid
        target_n = 16
        if len(frames) >= target_n:
            indices = np.linspace(0, len(frames) - 1, target_n, dtype=int)
            sel_frames = [cv2.resize(frames[idx], (160, 120)) for idx in indices]
        else:
            sel_frames = [cv2.resize(frames[min(i, len(frames)-1)], (160, 120)) for i in range(target_n)]

        row1 = np.hstack(sel_frames[0:4])
        row2 = np.hstack(sel_frames[4:8])
        row3 = np.hstack(sel_frames[8:12])
        row4 = np.hstack(sel_frames[12:16])
        grid = np.vstack([row1, row2, row3, row4])
        sampled = [Image.fromarray(grid)]
        prompt_text = (
            "Analyze this 4x4 temporal storyboard containing 16 sequential video frames of the event.\n"
            + prompts_eval.SMOLVLM2_USER_PROMPT
        )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in sampled] + [{"type": "text", "text": prompt_text}]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=sampled, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Parse response into one of the 7 valid classes
    pred_class = "Normal"
    import re
    # 1. Exact class match (check anomaly classes first to avoid 'abnormal' matching 'normal')
    anomaly_classes = [c for c in prompts_eval.CLASSES if c != "Normal"]
    matched = False
    for c in anomaly_classes:
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, response_text, re.IGNORECASE):
            pred_class = c
            matched = True
            break

    # 2. Digit match if model outputted option number
    if not matched:
        digit_match = re.search(r"\b([1-7])\b", response_text)
        if digit_match:
            digit_map = {
                "1": "Normal",
                "2": "Violence",
                "3": "Fighting",
                "4": "Arson",
                "5": "Car Accident",
                "6": "Explosion",
                "7": "Shooting",
            }
            pred_class = digit_map.get(digit_match.group(1), "Normal")
            matched = True

    # 3. Fallback keywords
    if not matched:
        lower = response_text.lower()
        if any(w in lower for w in ["fire", "flame", "burn", "arson"]):
            pred_class = "Arson"
        elif any(w in lower for w in ["crash", "collision", "accident", "vehicular"]):
            pred_class = "Car Accident"
        elif any(w in lower for w in ["bomb", "blast", "explod"]):
            pred_class = "Explosion"
        elif any(w in lower for w in ["gun", "shoot", "pistol", "rifle", "bullet"]):
            pred_class = "Shooting"
        elif any(w in lower for w in ["fist", "punch", "brawl", "fight"]):
            pred_class = "Fighting"
        elif any(w in lower for w in ["riot", "assault", "vandal", "mob"]):
            pred_class = "Violence"
        elif any(w in lower for w in ["normal", "peaceful", "calm"]):
            pred_class = "Normal"

    return {
        "top": pred_class,
        "raw_response": response_text,
        "latency_ms": latency_ms,
        "model": config_eval.TIER2_MODEL_ID
    }


def classify_cascade(tier1_bundle: tuple, tier2_bundle: tuple,
                     frames: List[np.ndarray], device: str = "cuda",
                     escalate_thr: float = 0.65,
                     num_frames_vlm: int = config_eval.SMOLVLM2_NUM_FRAMES) -> Dict:
    """Two-tier cascade: Run SigLIP -> if ambiguous, escalate to SmolVLM2."""
    t1_model, t1_proc, t1_embs = tier1_bundle
    t1_res = classify_tier1(t1_model, t1_proc, t1_embs, frames, device=device)

    # Determine if Tier 2 escalation is required
    needs_tier2 = False
    if t1_res["top_prob"] < escalate_thr:
        needs_tier2 = True
    elif t1_res["top"] in ["Violence", "Fighting"] and t1_res["top_prob"] < 0.75:
        # Extra verification for physical altercation vs general violence
        needs_tier2 = True

    if not needs_tier2 or tier2_bundle is None:
        return {
            "top": t1_res["top"],
            "top_prob": t1_res["top_prob"],
            "tier1_res": t1_res,
            "tier2_res": None,
            "escalated": False,
            "total_latency_ms": t1_res["latency_ms"]
        }

    t2_model, t2_proc = tier2_bundle
    t2_res = classify_tier2(t2_model, t2_proc, frames, device=device, num_frames=num_frames_vlm)

    # Blend confidence: when Tier 2 confirms, use high confidence
    final_class = t2_res["top"]
    final_conf = max(t1_res["top_prob"], 0.85) if final_class != "Normal" else 0.90

    return {
        "top": final_class,
        "top_prob": final_conf,
        "tier1_res": t1_res,
        "tier2_res": t2_res,
        "escalated": True,
        "total_latency_ms": t1_res["latency_ms"] + t2_res["latency_ms"]
    }

