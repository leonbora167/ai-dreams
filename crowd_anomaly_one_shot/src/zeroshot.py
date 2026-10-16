"""Zero-shot classification module for Phase 2C.

Implements CLIP frame classifier and small video VLM interfaces, score normalization,
temporal aggregation (mean vs max), and label decision with confidence thresholding.
"""

import cv2
import numpy as np
import torch


def scores_to_prediction(class_scores: dict, classes: list, classifier_name: str) -> dict:
    """Normalize raw scores or logits via softmax into a calibrated probability distribution contract."""
    scores_arr = np.array([class_scores.get(c, 0.0) for c in classes], dtype=np.float64)
    # Numerical stability shift
    shifted = scores_arr - np.max(scores_arr)
    exp_scores = np.exp(shifted)
    probs_arr = exp_scores / max(np.sum(exp_scores), 1e-12)

    probs = {c: round(float(probs_arr[i]), 4) for i, c in enumerate(classes)}
    top_idx = int(np.argmax(probs_arr))
    top_cls = classes[top_idx]
    top_prob = round(float(probs_arr[top_idx]), 4)

    return {
        "probs": probs,
        "top": top_cls,
        "top_prob": top_prob,
        "classifier": classifier_name
    }


def choose_label(prediction: dict, thr: float = 0.5) -> str:
    """Return top alert label if above confidence threshold; otherwise return 'none'."""
    top_cls = prediction.get("top", "none")
    top_prob = prediction.get("top_prob", 0.0)
    if top_cls == "none" or top_prob < thr:
        return "none"
    return top_cls


def load_clip(name: str = "openai/clip-vit-base-patch32", device: str = None) -> tuple:
    """Load CLIP model and processor from Hugging Face onto target device."""
    from transformers import CLIPModel, CLIPProcessor
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(name).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def text_embeddings(model, class_phrases: dict, processor, device: str = None) -> dict:
    """Compute averaged L2-normalized text feature vectors for each class from a list of phrasings."""
    dev = next(model.parameters()).device
    emb_dict = {}
    for c, phrases in class_phrases.items():
        inputs = processor(text=phrases, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            avg_feat = text_feats.mean(dim=0)
            avg_feat = avg_feat / avg_feat.norm(dim=-1)
            emb_dict[c] = avg_feat.cpu().numpy()
    return emb_dict


def classify_clip_frames(model, clip: np.ndarray, text_emb: dict, processor,
                         agg: str = "mean", temperature: float = 100.0) -> dict:
    """Classify temporal clip (T, S, S, 3 uint8 BGR) using CLIP image encoder and temporal aggregation."""
    assert clip.ndim == 4 and clip.shape[3] == 3, f"Expected (T, S, S, 3), got {clip.shape}"
    dev = next(model.parameters()).device
    classes = list(text_emb.keys())

    # Preprocess frames to RGB list
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]
    inputs = processor(images=rgb_frames, return_tensors="pt").to(dev)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_np = img_feats.cpu().numpy()  # (T, D)

    # Compute cosine similarities for each class across all T frames
    raw_class_scores = {}
    for c in classes:
        w = text_emb[c]  # (D,)
        sims = np.dot(img_np, w)  # (T,)
        score = float(np.max(sims)) if agg.lower() == "max" else float(np.mean(sims))
        raw_class_scores[c] = score * (temperature / 10.0)  # scale for calibration

    return scores_to_prediction(raw_class_scores, classes, classifier_name=f"clip_{agg}")


def load_siglip(name: str = None, device: str = None) -> tuple:
    """Load Google SigLIP vision-text model and processor onto target device."""
    from transformers import AutoModel, AutoProcessor
    import src.config as config
    if name is None:
        name = config.TIER1_MODEL_ID
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def siglip_text_embeddings(model, class_phrases: dict, processor, device: str = None) -> dict:
    """Compute averaged L2-normalized text feature vectors for each class using SigLIP."""
    dev = next(model.parameters()).device
    emb_dict = {}
    for c, phrases in class_phrases.items():
        inputs = processor(text=phrases, return_tensors="pt", padding="max_length").to(dev)
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            avg_feat = text_feats.mean(dim=0)
            avg_feat = avg_feat / avg_feat.norm(dim=-1)
            emb_dict[c] = avg_feat.cpu().numpy()
    return emb_dict


def classify_siglip_frames(model, clip: np.ndarray, text_emb: dict, processor,
                           agg: str = "mean", temperature: float = 80.0) -> dict:
    """Classify temporal clip using SigLIP image encoder and cosine matching."""
    assert clip.ndim == 4 and clip.shape[3] == 3, f"Expected (T, S, S, 3), got {clip.shape}"
    dev = next(model.parameters()).device
    classes = list(text_emb.keys())

    from PIL import Image
    step = max(1, len(clip) // 8)
    pil_frames = [Image.fromarray(cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)) for i in range(0, len(clip), step)][:8]
    inputs = processor(images=pil_frames, return_tensors="pt").to(dev)
    if hasattr(inputs, "pixel_values") and inputs.pixel_values.dtype != torch.float16:
        inputs.pixel_values = inputs.pixel_values.to(torch.float16)

    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_np = img_feats.cpu().numpy()

    raw_class_scores = {}
    for c in classes:
        w = text_emb[c]
        sims = np.dot(img_np, w)
        score = float(np.max(sims)) if agg.lower() == "max" else float(np.mean(sims))
        raw_class_scores[c] = score * (temperature / 10.0)

    return scores_to_prediction(raw_class_scores, classes, classifier_name="siglip")


def load_vlm(name: str = None, device: str = None) -> tuple:
    """Load SmolVLM2 non-Chinese multimodal model and processor onto target device."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import src.config as config
    if name is None:
        name = config.TIER2_MODEL_ID
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        name,
        dtype=torch.float16,
        _attn_implementation="sdpa"
    ).to(dev).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def classify_vlm(model, processor, clip: np.ndarray, prompt: str, classes: list,
                 fps: float = 5.0, num_frames: int = 16) -> dict:
    """Classify video clip using SmolVLM2 generative multimodal comprehension.
    
    Supports:
    - num_frames = 1: Single static snapshot
    - 2 <= num_frames <= 6: Multi-frame temporal sequence
    - num_frames > 6 (e.g. 16): 4x4 temporal storyboard grid
    """
    from PIL import Image
    import re
    dev = next(model.parameters()).device

    if len(clip) == 0:
        return scores_to_prediction({c: 0.1 for c in classes}, classes, "smolvlm2")

    if num_frames == 1 or len(clip) == 1:
        mid_idx = len(clip) // 2
        f = cv2.cvtColor(cv2.resize(clip[mid_idx], (224, 224)), cv2.COLOR_BGR2RGB)
        sampled = [Image.fromarray(f)]
        prompt_text = prompt
    elif 2 <= num_frames <= 6:
        indices = np.linspace(0, len(clip) - 1, min(num_frames, len(clip)), dtype=int)
        sampled = [Image.fromarray(cv2.cvtColor(cv2.resize(clip[idx], (224, 224)), cv2.COLOR_BGR2RGB)) for idx in indices]
        prompt_text = f"Analyze this {len(sampled)}-frame surveillance video sequence.\n" + prompt
    else:
        # 16-frame 4x4 temporal storyboard grid
        target_n = 16
        if len(clip) >= target_n:
            indices = np.linspace(0, len(clip) - 1, target_n, dtype=int)
            sel_frames = [cv2.cvtColor(cv2.resize(clip[idx], (160, 120)), cv2.COLOR_BGR2RGB) for idx in indices]
        else:
            sel_frames = [cv2.cvtColor(cv2.resize(clip[min(i, len(clip)-1)], (160, 120)), cv2.COLOR_BGR2RGB) for i in range(target_n)]

        row1 = np.hstack(sel_frames[0:4])
        row2 = np.hstack(sel_frames[4:8])
        row3 = np.hstack(sel_frames[8:12])
        row4 = np.hstack(sel_frames[12:16])
        grid = np.vstack([row1, row2, row3, row4])
        sampled = [Image.fromarray(grid)]
        prompt_text = "Analyze this 4x4 temporal storyboard containing 16 sequential video frames of the event.\n" + prompt

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in sampled] + [{"type": "text", "text": prompt_text}]
        }
    ]
    formatted = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=formatted, images=sampled, return_tensors="pt").to(dev)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

    pred_class = "Normal"
    # Anomaly match
    anomaly_classes = [c for c in classes if c.lower() != "normal"]
    matched = False
    for c in anomaly_classes:
        if re.search(r"\b" + re.escape(c) + r"\b", response_text, re.IGNORECASE):
            pred_class = c
            matched = True
            break
    if not matched:
        if re.search(r"\bNormal\b", response_text, re.IGNORECASE):
            pred_class = "Normal"

    raw_scores = {c: (1.0 if c == pred_class else 0.05) for c in classes}
    res = scores_to_prediction(raw_scores, classes, classifier_name="smolvlm2")
    res["raw_response"] = response_text
    return res


def classify_cascade(tier1_bundle: tuple, vlm_bundle: tuple,
                     clip: np.ndarray, prompt: str, classes: list,
                     conf_thr: float = 0.65, num_frames_vlm: int = 16) -> dict:
    """Fast-pass SigLIP triage; escalate to SmolVLM2 if confidence is below threshold."""
    t1_model, t1_proc, t1_embs = tier1_bundle
    t1_pred = classify_siglip_frames(t1_model, clip, t1_embs, t1_proc, agg="mean")
    if t1_pred["top_prob"] >= conf_thr:
        t1_pred["classifier"] = "cascade_siglip"
        return t1_pred

    vlm_model, vlm_proc = vlm_bundle
    vlm_pred = classify_vlm(vlm_model, vlm_proc, clip, prompt, classes, num_frames=num_frames_vlm)
    vlm_pred["classifier"] = "cascade_smolvlm2"
    return vlm_pred
