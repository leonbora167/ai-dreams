"""
Simple LLM context-length checker.

Usage examples (do NOT run here):
python check_llm_context.py --model "meta-llama/Llama-2-7b-chat-hf" --prompt-file final_prompt.txt
python check_llm_context.py --model "decapoda-research/llama-7b-hf" --prompt "Hello world..."
"""
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def resolve_max_length(tokenizer, model=None) -> Optional[int]:
    # try tokenizer first
    try:
        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and tokenizer.model_max_length > 0:
            return int(tokenizer.model_max_length)
    except Exception:
        pass

    # try common config fields on model.config
    if model is not None and hasattr(model, "config"):
        cfg = model.config
        for attr in ("max_position_embeddings", "n_ctx", "context_length", "max_length"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)
    return None


def count_tokens(text: str, tokenizer) -> int:
    # use add_special_tokens=False because final prompt is user-rendered
    toks = tokenizer.encode(text, add_special_tokens=False)
    return len(toks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path (AutoTokenizer/AutoModel)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Final rendered prompt text")
    group.add_argument("--prompt-file", help="Path to a file containing final prompt text")
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
    token_count = count_tokens(prompt_text, tokenizer)

    out = {
        "model": args.model,
        "token_count": token_count,
        "max_length_detected": max_len,
        "ok": (max_len is None) or (token_count <= max_len),
        "notes": []
    }

    if max_len is None:
        out["notes"].append("Could not detect a max context length from tokenizer/model config. Tokenizer.model_max_length may be 'inf' or unknown.")
    else:
        if token_count > max_len:
            out["notes"].append("Prompt exceeds detected max context length.")
        else:
            out["notes"].append("Prompt fits within detected max context length.")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()