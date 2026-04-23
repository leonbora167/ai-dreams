"""
VLM (vision+language) context-length checker.

Heuristics:
- Counts text tokens via the tokenizer.
- Attempts to infer image-token count per image from model.config.vision_config (patch_size, image_size).
- If it cannot infer, uses a default image token estimate (256) which you can override with --image-tokens.
- Final effective token usage = text_tokens + num_images * image_tokens_per_image

Usage:
python check_vlm_context.py --model "facebook/llama-vl-example" --prompt-file final_prompt.txt --num-images 1
"""
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def resolve_max_length(tokenizer, model=None) -> Optional[int]:
    try:
        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and tokenizer.model_max_length > 0:
            return int(tokenizer.model_max_length)
    except Exception:
        pass
    if model is not None and hasattr(model, "config"):
        cfg = model.config
        for attr in ("max_position_embeddings", "n_ctx", "context_length", "context_length_tokens", "max_length"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)
    return None


def infer_image_tokens_per_image(model) -> Optional[int]:
    # Try a couple of common patterns in HF VLM configs
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    # Try vision_config with patch_size & image_size
    vision_cfg = getattr(cfg, "vision_config", None) or getattr(cfg, "image_config", None)
    if vision_cfg is not None:
        patch = getattr(vision_cfg, "patch_size", None) or getattr(vision_cfg, "patches", None)
        image_size = getattr(vision_cfg, "image_size", None) or getattr(vision_cfg, "img_size", None)
        # some configs use tuple/list
        if isinstance(patch, (list, tuple)):
            patch = patch[0]
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        try:
            if patch and image_size:
                tokens = (image_size // patch) ** 2
                if tokens > 0:
                    return int(tokens)
        except Exception:
            pass

    # Try attribute directly on config: num_image_tokens / image_token_count
    for attr in ("num_image_tokens", "image_token_count", "num_patches"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)

    # No inference available
    return None


def count_text_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF multimodal model id/path")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Final rendered prompt text")
    group.add_argument("--prompt-file", help="Path to a file containing final prompt text")
    p.add_argument("--num-images", type=int, default=0, help="Number of images attached to the prompt")
    p.add_argument("--image-tokens", type=int, default=None, help="Override image tokens per image (integer)")
    p.add_argument("--load-model", action="store_true", help="Load model config (slower). If not set, only tokenizer is loaded.")
    args = p.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    model = None
    if args.load_model:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=False)

    max_len = resolve_max_length(tokenizer, model)
    text_tokens = count_text_tokens(prompt_text, tokenizer)

    # Determine image tokens per image
    if args.image_tokens is not None:
        img_tokens_per = args.image_tokens
        inferred = False
    else:
        inferred_val = infer_image_tokens_per_image(model) if model is not None else None
        if inferred_val is not None:
            img_tokens_per = inferred_val
            inferred = True
        else:
            img_tokens_per = 256  # conservative default heuristic
            inferred = False

    total_image_tokens = args.num_images * img_tokens_per
    effective_total = text_tokens + total_image_tokens

    out = {
        "model": args.model,
        "text_token_count": text_tokens,
        "num_images": args.num_images,
        "image_tokens_per_image": img_tokens_per,
        "image_tokens_included": total_image_tokens,
        "effective_total_tokens": effective_total,
        "max_length_detected": max_len,
        "ok": (max_len is None) or (effective_total <= max_len),
        "notes": []
    }

    if not inferred and args.image_tokens is None:
        out["notes"].append("Image tokens per image was NOT inferred; using default heuristic (256). Use --image-tokens to override if you know the actual value.")
    if max_len is None:
        out["notes"].append("Could not detect a max context length from tokenizer/model config.")
    else:
        if effective_total > max_len:
            out["notes"].append("Combined prompt (text + image tokens) exceeds detected max context length.")
        else:
            out["notes"].append("Combined prompt fits within detected max context length.")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()