import argparse
import base64
import json
import re
import sys
import types
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from embedder import LocalEmbedder
from vector_store import AuditoriumVectorDB


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def extract_mentioned_track_ids(question: str) -> list[str]:
    """Extracts explicit IDs like 'Track 2', 'Track #4', or 'Person 1'."""
    matches = re.findall(r"(?:track|person|id)\s*#?\s*(\d+)", question, flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


class QueryRouter:
    """
    Classifies a natural-language query into one of three retrieval routes:
      - 'track_id': Explicit track/person ID lookup & timeline inspection
      - 'counting': Scene-wide headcounts, standing/sitting counts, crowd statistics
      - 'person_forensic': Individual appearance, clothing, props (drink, bag, mic)
    Uses structural ID extraction + zero-shot SigLIP semantic anchor similarity
    (no brittle keyword lists).
    """

    ROUTE_ANCHORS = {
        "counting": [
            "how many people are in the scene standing or sitting",
            "count the total number of people in the room or crowd",
            "statistics and headcount of people across the auditorium"
        ],
        "person_forensic": [
            "is anyone wearing a specific shirt holding a drink handbag or microphone",
            "find a person carrying an object or wearing specific clothes",
            "identify individual appearance clothing posture or props held by someone"
        ]
    }

    def __init__(self, embedder: LocalEmbedder):
        self.anchor_vectors: dict[str, np.ndarray] = {}
        for route_name, prompts in self.ROUTE_ANCHORS.items():
            vecs = [np.array(embedder.extract_multimodal_text(p), dtype=np.float32) for p in prompts]
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self.anchor_vectors[route_name] = mean_vec

    def classify(
        self,
        question: str,
        query_vec: list[float] | np.ndarray,
        forced_route: str = "auto"
    ) -> tuple[str, list[str], dict[str, float]]:
        """
        Returns (selected_route, mentioned_ids, similarity_scores).
        """
        mentioned_ids = extract_mentioned_track_ids(question)
        qv = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(qv)
        if q_norm > 0:
            qv = qv / q_norm

        scores = {
            name: float(np.dot(qv, anchor_vec))
            for name, anchor_vec in self.anchor_vectors.items()
        }

        if forced_route != "auto":
            return forced_route, mentioned_ids, scores

        if mentioned_ids:
            return "track_id", mentioned_ids, scores

        selected = "counting" if scores["counting"] >= scores["person_forensic"] else "person_forensic"
        return selected, mentioned_ids, scores


def _build_quantization_config(quantize: str):
    if not quantize or quantize == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Quantization requires a newer `transformers` version.") from exc

    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            f"--quantize {quantize} requested, but `bitsandbytes` is not installed. "
            "Install it via `pip install bitsandbytes`, or serve the model quantized via Ollama/vLLM and pass --endpoint-url."
        )

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantize mode: {quantize}")


# ==============================================================================
# Base VLM Provider Interface
# ==============================================================================

class BaseVLMBackend(ABC):
    """
    Abstract interface for Vision-Language Model execution.
    Decouples retrieval logic from model loading and API protocols.
    """
    @abstractmethod
    def generate_answer(
        self,
        question: str,
        prompt_text: str,
        visual_items: list[dict],
        images: list[Image.Image]
    ) -> str:
        pass


# ==============================================================================
# 1. Local Hugging Face Backend (SmolVLM2, Llama 3.2 Vision, Phi-4, Generic HF)
# ==============================================================================

class HuggingFaceLocalBackend(BaseVLMBackend):
    """
    Executes open-source multimodal models from Hugging Face.
    Handles input conventions, Phi-4 AutoConfig attention overrides (sdpa/eager),
    and optional 4-bit/8-bit BitsAndBytes quantization for large models (Llama-3.2-11B).
    """
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        attn_impl: str = "sdpa",
        quantize: str = "none"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        # Normalize common typo 'sdp' -> 'sdpa'
        norm_attn = "sdpa" if attn_impl.lower() in ("sdp", "sdpa") else attn_impl.lower()
        print(f"Loading local VLM ({model_id}) on {self.device} [attn={norm_attn}, quantize={quantize}]...")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        quant_cfg = _build_quantization_config(quantize)
        model_lower = model_id.lower()

        common_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
        }
        if quant_cfg is not None:
            common_kwargs["quantization_config"] = quant_cfg
        else:
            common_kwargs["dtype"] = dtype

        # Family 1: SmolVLM / SmolVLM2
        if "smolvlm" in model_lower:
            self.model_family = "smolvlm"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs)
            self.processor = AutoProcessor.from_pretrained(model_id)

            if "smolvlm2" in model_lower:
                def _fixed_smolvlm2_inputs_merger(m_self, input_ids, inputs_embeds, image_hidden_states):
                    if input_ids is None:
                        image_mask = inputs_embeds == m_self.get_input_embeddings()(
                            torch.full((), m_self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                        )
                        image_mask = image_mask[..., 0]
                    else:
                        image_mask = input_ids == m_self.config.image_token_id
                    image_embeds = torch.zeros_like(inputs_embeds)
                    image_embeds[image_mask] = image_hidden_states.reshape(-1, image_hidden_states.shape[-1]).to(inputs_embeds.dtype)
                    return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

                if hasattr(self.model, "model") and hasattr(self.model.model, "inputs_merger"):
                    self.model.model.inputs_merger = types.MethodType(_fixed_smolvlm2_inputs_merger, self.model.model)

        # Family 2: Meta Llama 3.2 Vision (11B)
        elif "llama" in model_lower:
            self.model_family = "llama"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs)
            self.processor = AutoProcessor.from_pretrained(model_id)

        # Family 3: Microsoft Phi-4 Multimodal
        elif "phi-4" in model_lower or "phi4" in model_lower:
            self.model_family = "phi4"
            from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
            # Explicitly patch baked-in flash_attention_2 flags inside AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            cfg._attn_implementation = norm_attn
            cfg._attn_implementation_internal = norm_attn

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=cfg,
                attn_implementation=norm_attn,
                trust_remote_code=True,
                **common_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Family 4: Generic Hugging Face Multimodal Fallback
        else:
            self.model_family = "generic_hf"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                **common_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def generate_answer(
        self,
        question: str,
        prompt_text: str,
        visual_items: list[dict],
        images: list[Image.Image]
    ) -> str:
        # Pattern A: Chat-Template Interleaved Multi-Image (SmolVLM2 & Llama 3.2)
        if self.model_family in ("smolvlm", "llama"):
            content = []
            for item in visual_items:
                content.append({"type": "image"})
                content.append({"type": "text", "text": f"[{item['label']}]\n"})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=350)
                decoded = self.processor.decode(out[0], skip_special_tokens=True)
                if self.model_family == "llama" and "assistant\n" in decoded:
                    return decoded.split("assistant\n")[-1].strip()
                elif "Assistant:" in decoded:
                    return decoded.split("Assistant:")[-1].strip()
                return decoded.strip()

        # Pattern B: Numbered Special Token Markup (Phi-4 Multimodal)
        elif self.model_family == "phi4":
            image_tags = "".join([f"<|image_{i}|> [{item['label']}]\n" for i, item in enumerate(visual_items, 1)])
            formatted_prompt = f"<|user|>\n{image_tags}{prompt_text}\n<|end|>\n<|assistant|>\n"
            inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=350)
                decoded = self.processor.decode(out[0], skip_special_tokens=True)
                return decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded.strip()

        # Pattern C: Generic Fallback
        else:
            try:
                content = []
                for item in visual_items:
                    content.append({"type": "image"})
                    content.append({"type": "text", "text": f"[{item['label']}]\n"})
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
            except Exception:
                inputs = self.processor(text=prompt_text, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=350)
                return self.processor.decode(out[0], skip_special_tokens=True).strip()


# ==============================================================================
# 2. Remote OpenAI-Compatible Endpoint Backend (Nemotron, vLLM, NIM, Ollama)
# ==============================================================================

class OpenAICompatibleRemoteBackend(BaseVLMBackend):
    def __init__(self, endpoint_url: str, model_name: str = "nvidia/nemotron-4-340b", api_key: str | None = None):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.api_key = api_key
        print(f"Configured remote VLM endpoint: {endpoint_url} (Model: {model_name})")

    def _pil_to_base64_data_uri(self, img: Image.Image) -> str:
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def generate_answer(
        self,
        question: str,
        prompt_text: str,
        visual_items: list[dict],
        images: list[Image.Image]
    ) -> str:
        content_items = []
        for item, img in zip(visual_items, images):
            data_uri = self._pil_to_base64_data_uri(img)
            content_items.append({"type": "text", "text": f"[{item['label']}]:"})
            content_items.append({"type": "image_url", "image_url": {"url": data_uri}})

        content_items.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_items}],
            "max_tokens": 512,
            "temperature": 0.2
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Remote API Request Error: {e}"


# ==============================================================================
# RAG Orchestrator
# ==============================================================================

class LocalVLMQA:
    def __init__(self, backend: BaseVLMBackend):
        self.backend = backend

    def answer_question(
        self,
        question: str,
        candidate_tracks: list[dict],
        scene_frames: list[dict] | None = None,
        structured_stats: dict | None = None
    ) -> str:
        images: list[Image.Image] = []
        visual_items: list[dict] = []
        context_lines: list[str] = []

        # Include full scene keyframes first (critical for scene-wide counting / posture / held props)
        if scene_frames:
            for i, sf in enumerate(scene_frames, 1):
                fpath = Path(sf.get("frame_path", ""))
                if fpath.exists():
                    try:
                        img = Image.open(str(fpath)).convert("RGB")
                        ts = sf.get("timestamp_sec", 0.0)
                        p_cnt = sf.get("person_count", 0)
                        st_cnt = sf.get("standing_count", 0)
                        si_cnt = sf.get("sitting_count", 0)
                        label = f"Scene Keyframe #{i} @ {format_timestamp(ts)} ({ts:.1f}s) | Detected={p_cnt}, Standing={st_cnt}, Sitting={si_cnt}"
                        images.append(img)
                        visual_items.append({"type": "scene", "label": label})
                        context_lines.append(f"- {label}")
                    except Exception as e:
                        print(f"Warning: Could not open scene frame {fpath}: {e}", file=sys.stderr)

        # Include individual candidate person crops
        for i, rec in enumerate(candidate_tracks, 1):
            track_id = rec.get("track_id", "Unknown")
            pose_label = rec.get("pose_label", "unknown")
            start_sec = rec.get("start_sec", 0.0)
            end_sec = rec.get("end_sec", 0.0)
            start_f = rec.get("start_frame", 0)
            end_f = rec.get("end_frame", 0)

            intervals_desc = ""
            intervals_json = rec.get("intervals_json", "")
            if intervals_json:
                try:
                    inv_list = json.loads(intervals_json)
                    if isinstance(inv_list, list) and len(inv_list) > 1:
                        intervals_desc = " (Intervals: " + ", ".join(
                            [f"{iv['start_sec']:.1f}s-{iv['end_sec']:.1f}s" for iv in inv_list]
                        ) + ")"
                except Exception:
                    pass

            time_str = f"{start_sec:.1f}s to {end_sec:.1f}s ({format_timestamp(start_sec)} - {format_timestamp(end_sec)}){intervals_desc}"
            crop_path = Path(rec.get("best_crop_path", ""))
            if crop_path.exists():
                try:
                    img = Image.open(str(crop_path)).convert("RGB")
                    label = f"Candidate #{i} / Global Track ID #{track_id} | Pose={pose_label} | Visible: {time_str}"
                    images.append(img)
                    visual_items.append({"type": "track", "label": label})
                    context_lines.append(f"- {label}, Frames: {start_f}-{end_f}")
                except Exception as e:
                    print(f"Warning: Could not open {crop_path}: {e}", file=sys.stderr)

        stats_block = ""
        if structured_stats:
            stats_block = (
                "Database Pose & Aggregation Summary (automated keypoint counts):\n"
                f"- Total Unique Global Tracks: {structured_stats.get('total_unique_tracks', 0)}\n"
                f"- Track-level Pose Counts -> Standing: {structured_stats.get('standing_tracks', 0)}, "
                f"Sitting: {structured_stats.get('sitting_tracks', 0)}, Unknown: {structured_stats.get('unknown_pose_tracks', 0)}\n"
                f"- Peak Simultaneous People in a Single Scene Frame: {structured_stats.get('peak_simultaneous_people', 0)} "
                f"(Max Standing in frame: {structured_stats.get('peak_simultaneous_standing', 0)}, "
                f"Max Sitting in frame: {structured_stats.get('peak_simultaneous_sitting', 0)})\n\n"
            )

        timeline_context = "\n".join(context_lines)
        prompt_text = (
            f"You are an AI video surveillance analyst assistant.\n"
            f"{stats_block}"
            f"Retrieved Visual Context (full scene keyframes and/or top person tracklets):\n"
            f"{timeline_context}\n\n"
            f"User Question: {question}\n\n"
            f"Instructions:\n"
            f"1. For counting or posture questions, combine the structured Database Pose Summary and full Scene Keyframes.\n"
            f"2. For person/object/time questions, inspect the visual appearance (clothing, props like microphones, posture) of each image.\n"
            f"3. Provide a concise, factual answer citing exact Global Track IDs, timestamps, or counts.\n"
            f"4. If the evidence does not match what is asked, explicitly state that."
        )

        return self.backend.generate_answer(question, prompt_text, visual_items, images)


# ==============================================================================
# Main CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auditorium Post-Execution VLM Q&A CLI")
    parser.add_argument("--question", type=str, required=True, help="Natural language question to ask about the video")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate tracklets to retrieve")
    parser.add_argument("--route", type=str, default="auto", choices=["auto", "track_id", "counting", "person_forensic"],
                        help="Query routing mode (default: auto semantic classification)")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384", help="SigLIP model ID")

    # Model Selection & Runtime Options
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                        help="HF model ID: SmolVLM2-2.2B, Llama-3.2-11B-Vision-Instruct, or Phi-4-multimodal-instruct")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "sdp", "eager"],
                        help="Attention implementation override for models like Phi-4 ('sdpa' or 'eager')")
    parser.add_argument("--quantize", type=str, default="none", choices=["none", "4bit", "8bit"],
                        help="Optional BitsAndBytes quantization for large local models like Llama-3.2-11B")
    parser.add_argument("--endpoint-url", type=str, default=None,
                        help="Optional remote OpenAI-compatible server URL (Nemotron, vLLM, NIM, Ollama)")
    parser.add_argument("--model-name", type=str, default="nvidia/nemotron-4-340b",
                        help="Model name identifier for remote endpoint")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Bearer API key for remote endpoint")
    args = parser.parse_args()

    print(f"\n[1/3] Classifying query & searching vector database for: \"{args.question}\"...")
    embedder = LocalEmbedder(siglip_id=args.siglip_model)
    text_vec = embedder.extract_multimodal_text(args.question)
    vector_db = AuditoriumVectorDB(siglip_dim=embedder.siglip_dim)

    router = QueryRouter(embedder)
    route, mentioned_ids, route_scores = router.classify(args.question, text_vec, forced_route=args.route)
    print(
        f"  -> Selected Route: [{route.upper()}] "
        f"(counting={route_scores['counting']:.3f}, person_forensic={route_scores['person_forensic']:.3f})"
    )

    candidates = []
    scene_frames = []
    structured_stats = None

    if route == "track_id":
        all_recs = {str(r["track_id"]): r for r in vector_db.get_all_records()}
        for mid in mentioned_ids:
            if mid in all_recs:
                candidates.append(all_recs[mid])
        # Supplement if no explicit ID was found in DB
        if not candidates:
            candidates = vector_db.search_by_text_vector(text_vec, top_k=args.top_k)

    elif route == "counting":
        structured_stats = vector_db.get_structured_stats()
        scene_frames = vector_db.search_scene_frames(text_vec, top_k=3)
        candidates = vector_db.search_by_text_vector(text_vec, top_k=max(args.top_k, 5))

    else:  # person_forensic
        candidates = vector_db.search_by_text_vector(text_vec, top_k=max(args.top_k, 5))

    if not candidates and not scene_frames:
        print("No tracklets or scene keyframes found in LanceDB. Please run pipeline.py first.")
        sys.exit(0)

    print(f"[2/3] Retrieved {len(candidates)} candidate track(s) and {len(scene_frames)} full scene keyframe(s):")
    for c in candidates:
        print(
            f"  - Track #{c.get('track_id')} | Pose: {c.get('pose_label', 'unknown')} "
            f"| Time: {c.get('start_sec', 0.0):.1f}s - {c.get('end_sec', 0.0):.1f}s"
        )
    if structured_stats:
        print(
            f"  - DB Pose Stats -> Standing Tracks: {structured_stats['standing_tracks']}, "
            f"Sitting Tracks: {structured_stats['sitting_tracks']}, "
            f"Peak People in Frame: {structured_stats['peak_simultaneous_people']}"
        )

    if args.endpoint_url:
        print(f"\n[3/3] Sending request to remote server ({args.endpoint_url})...\n")
        backend = OpenAICompatibleRemoteBackend(
            endpoint_url=args.endpoint_url,
            model_name=args.model_name,
            api_key=args.api_key
        )
    else:
        print(f"\n[3/3] Prompting local VLM ({args.model_id})...\n")
        backend = HuggingFaceLocalBackend(
            model_id=args.model_id,
            attn_impl=args.attn_impl,
            quantize=args.quantize
        )

    vlm = LocalVLMQA(backend=backend)
    answer = vlm.answer_question(
        args.question,
        candidate_tracks=candidates,
        scene_frames=scene_frames,
        structured_stats=structured_stats
    )

    print("=" * 80)
    print("VLM ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
