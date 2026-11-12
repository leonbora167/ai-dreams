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
    Classifies a natural-language query into one of five retrieval routes:
      - 'track_id': Explicit track/person ID lookup & timeline inspection
      - 'counting': Scene-wide headcounts, standing/sitting counts, crowd statistics
      - 'person_forensic': Individual appearance, clothing, props (drink, shirt, mic)
      - 'object_ownership': Abandoned objects, bag/item ownership, carrier lookup
      - 'video_summary': Whole-video events, activities, speakers, dancing, summaries
    Uses structural ID extraction + zero-shot SigLIP semantic anchor similarity.
    """

    ROUTE_ANCHORS = {
        "counting": [
            "how many people are in the scene standing or sitting",
            "count the total number of people in the room or crowd",
            "statistics and headcount of people across the auditorium",
        ],
        "person_forensic": [
            "is anyone wearing a specific blue or red shirt or glasses",
            "find a person wearing specific clothes posture or appearance",
            "identify individual person holding a microphone or drink",
        ],
        "object_ownership": [
            "who does this bag backpack suitcase or water bottle belong to",
            "was any bag luggage or item abandoned left unattended behind",
            "who was carrying leaving or owns the bag bottle or object",
            "find when this object or bag was visible and who brought it",
        ],
        "video_summary": [
            "what happened overall in the video summarize the main events",
            "who was the speaker on the stage or who was dancing performing",
            "describe activities interactions and timeline of events in the room",
            "give a summary of what people did across the video",
        ],
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
        forced_route: str = "auto",
        has_reference_image: bool = False,
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

        # Reference image passed with ownership/object question prioritizes object_ownership
        if has_reference_image and scores["object_ownership"] >= scores["person_forensic"] - 0.03:
            return "object_ownership", mentioned_ids, scores

        selected = max(scores.items(), key=lambda kv: kv[1])[0]
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
        images: list[Image.Image],
        max_new_tokens: int = 350,
    ) -> str:
        pass

    def generate_text_only(self, prompt_text: str, max_new_tokens: int = 350) -> str:
        return self.generate_answer(
            question="",
            prompt_text=prompt_text,
            visual_items=[],
            images=[],
            max_new_tokens=max_new_tokens,
        )


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
            "device_map": {"": 0} if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
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

        # Family 2: OpenGVLab InternVL3 / InternVL2.5 HF native
        elif "internvl" in model_lower:
            self.model_family = "internvl"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                attn_implementation=norm_attn,
                **common_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        # Family 3: Meta Llama 3.2 Vision (11B)
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

    def _decode_generated(self, out_ids: torch.Tensor, inputs: dict) -> str:
        input_ids = inputs.get("input_ids")
        if input_ids is not None and out_ids.shape[0] > input_ids.shape[1]:
            gen_ids = out_ids[input_ids.shape[1]:]
            return self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        decoded = self.processor.decode(out_ids, skip_special_tokens=True)
        for marker in ("<|assistant|>", "assistant\n", "Assistant:"):
            if marker in decoded:
                return decoded.split(marker)[-1].strip()
        return decoded.strip()

    def _move_inputs_to_device(self, inputs: dict) -> dict:
        moved = {}
        target_dtype = getattr(self.model, "dtype", torch.float16 if self.device == "cuda" else torch.float32)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v) and target_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    moved[k] = v.to(device=self.device, dtype=target_dtype)
                else:
                    moved[k] = v.to(device=self.device)
            else:
                moved[k] = v
        return moved

    def generate_answer(
        self,
        question: str,
        prompt_text: str,
        visual_items: list[dict],
        images: list[Image.Image],
        max_new_tokens: int = 350,
    ) -> str:
        # Pattern A: Chat-Template Interleaved Multi-Image (SmolVLM2, Llama 3.2, InternVL3-hf)
        if self.model_family in ("smolvlm", "llama", "internvl"):
            content = []
            for item in visual_items:
                content.append({"type": "image"})
                content.append({"type": "text", "text": f"[{item['label']}]\n"})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            if images:
                proc_kwargs = {"text": prompt, "images": images, "return_tensors": "pt", "padding": True}
                if self.model_family == "internvl":
                    proc_kwargs["images_kwargs"] = (
                        {"max_patches": 3} if len(images) <= 2 else {"crop_to_patches": False}
                    )
                inputs = self.processor(**proc_kwargs)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
            inputs = self._move_inputs_to_device(inputs)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                return self._decode_generated(out[0], inputs)

        # Pattern B: Numbered Special Token Markup (Phi-4 Multimodal)
        elif self.model_family == "phi4":
            image_tags = "".join([f"<|image_{i}|> [{item['label']}]\n" for i, item in enumerate(visual_items, 1)])
            formatted_prompt = f"<|user|>\n{image_tags}{prompt_text}\n<|end|>\n<|assistant|>\n"
            if images:
                inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=formatted_prompt, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                return self._decode_generated(out[0], inputs)

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
                if images:
                    inputs = self.processor(text=prompt, images=images, return_tensors="pt")
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt")
            except Exception:
                if images:
                    inputs = self.processor(text=prompt_text, images=images, return_tensors="pt")
                else:
                    inputs = self.processor(text=prompt_text, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                return self._decode_generated(out[0], inputs)


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
        images: list[Image.Image],
        max_new_tokens: int = 512,
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
            "max_tokens": max_new_tokens,
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
        structured_stats: dict | None = None,
        object_records: list[dict] | None = None,
        video_events: list[dict] | None = None,
        reference_image_path: str | None = None,
    ) -> str:
        images: list[Image.Image] = []
        visual_items: list[dict] = []
        context_lines: list[str] = []

        # Include query reference image first if provided (e.g., query bag crop)
        if reference_image_path:
            ref_path = Path(reference_image_path)
            if ref_path.exists():
                try:
                    img = Image.open(str(ref_path)).convert("RGB")
                    label = "User Query Reference Image"
                    images.append(img)
                    visual_items.append({"type": "query_image", "label": label})
                    context_lines.append(f"- {label}: {ref_path.name}")
                except Exception as e:
                    print(f"Warning: Could not open reference image {ref_path}: {e}", file=sys.stderr)

        # Include full scene keyframes (critical for scene-wide counting / posture / props)
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

        # Include matched open-vocab tracked objects + owner & abandonment metadata
        if object_records:
            for i, obj in enumerate(object_records, 1):
                obj_id = obj.get("object_id", "?")
                lbl = obj.get("label", "object")
                owner_id = obj.get("owner_track_id", "Unknown")
                status = obj.get("status", "attended")
                s_sec = float(obj.get("start_sec", 0.0))
                e_sec = float(obj.get("end_sec", 0.0))
                ab_sec = float(obj.get("abandoned_since_sec", -1.0))
                ab_note = (
                    f", AbandonedSince={format_timestamp(ab_sec)} ({ab_sec:.1f}s)"
                    if status == "abandoned" and ab_sec >= 0
                    else ""
                )
                label = (
                    f"Tracked Object #{i} [OBJ:{obj_id}] ({lbl}) | Owner=Person ID #{owner_id} "
                    f"| Status={status}{ab_note} | Visible: {format_timestamp(s_sec)} - {format_timestamp(e_sec)} ({s_sec:.1f}s-{e_sec:.1f}s)"
                )
                crop_path = Path(obj.get("best_crop_path", ""))
                if crop_path.exists():
                    try:
                        img = Image.open(str(crop_path)).convert("RGB")
                        images.append(img)
                        visual_items.append({"type": "object", "label": label})
                    except Exception:
                        pass
                ab_frame_path = Path(obj.get("abandoned_frame_path", ""))
                if status == "abandoned" and ab_frame_path.exists() and len(images) < 6:
                    try:
                        ab_img = Image.open(str(ab_frame_path)).convert("RGB")
                        ab_label = f"Abandonment Evidence Frame for [OBJ:{obj_id}] ({lbl}) @ {ab_sec:.1f}s"
                        images.append(ab_img)
                        visual_items.append({"type": "abandoned_scene", "label": ab_label})
                    except Exception:
                        pass
                context_lines.append(f"- {label}")

        # Include spatiotemporal video event summaries (Phase 2)
        events_block = ""
        if video_events:
            ev_lines = []
            seen_keyframes = set()
            for ev in video_events:
                s_sec = float(ev.get("start_sec", 0.0))
                e_sec = float(ev.get("end_sec", 0.0))
                reason = ev.get("trigger_reason", "periodic")
                summary = ev.get("summary_text", "").strip()
                ref_tracks = ev.get("referenced_track_ids_json", "[]")
                ev_lines.append(
                    f"- [{format_timestamp(s_sec)} - {format_timestamp(e_sec)} ({s_sec:.1f}s-{e_sec:.1f}s), trigger={reason}, tracks={ref_tracks}]: {summary}"
                )
                kpath = str(ev.get("keyframe_path", ""))
                if kpath and kpath not in seen_keyframes and Path(kpath).exists() and len(images) < 6:
                    seen_keyframes.add(kpath)
                    try:
                        k_img = Image.open(kpath).convert("RGB")
                        k_label = f"Event Keyframe ({format_timestamp(s_sec)} - {format_timestamp(e_sec)})"
                        images.append(k_img)
                        visual_items.append({"type": "event_keyframe", "label": k_label})
                    except Exception:
                        pass
            events_block = "Chronological / Relevant Video Event Summaries (Spatial-Windowed Timeline):\n" + "\n".join(ev_lines) + "\n\n"

        # Include individual candidate person crops (cap total images to keep 2B VLMs crisp)
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
            label = f"Candidate #{i} / Global Track ID #{track_id} | Pose={pose_label} | Visible: {time_str}"
            crop_path = Path(rec.get("best_crop_path", ""))
            if crop_path.exists() and len(images) < 8:
                try:
                    img = Image.open(str(crop_path)).convert("RGB")
                    images.append(img)
                    visual_items.append({"type": "track", "label": label})
                except Exception as e:
                    print(f"Warning: Could not open {crop_path}: {e}", file=sys.stderr)
            context_lines.append(f"- {label}, Frames: {start_f}-{end_f}")

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
            f"{events_block}"
            f"Retrieved Visual & Entity Context:\n"
            f"{timeline_context}\n\n"
            f"User Question: {question}\n\n"
            f"Instructions:\n"
            f"1. For counting or posture questions, combine the structured Database Pose Summary and full Scene Keyframes.\n"
            f"2. For object/bag ownership or abandonment questions, cite the Tracked Object ID, owner Global Person ID (#), status, and exact timestamps.\n"
            f"3. For whole-video activity or speaker questions, synthesize the Video Event Summaries and linked Global Person IDs.\n"
            f"4. For person forensic questions, inspect visual appearance (clothing, props, posture) of each crop.\n"
            f"5. Provide a concise, factual answer citing exact Global Track IDs, timestamps, or counts. If evidence does not match, say so."
        )

        return self.backend.generate_answer(question, prompt_text, visual_items, images)


# ==============================================================================
# Main CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auditorium Post-Execution VLM Q&A CLI")
    parser.add_argument("--question", type=str, required=True, help="Natural language question to ask about the video")
    parser.add_argument("--image", type=str, default=None, help="Optional reference crop image (e.g., bag crop for ownership lookup)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate items to retrieve")
    parser.add_argument("--db-path", type=str, default="data/lancedb", help="LanceDB directory path")
    parser.add_argument(
        "--route",
        type=str,
        default="auto",
        choices=["auto", "track_id", "counting", "person_forensic", "object_ownership", "video_summary"],
        help="Query routing mode (default: auto semantic classification)"
    )
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-384", help="SigLIP model ID")

    # Model Selection & Runtime Options
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                        help="HF model ID: SmolVLM2-2.2B, InternVL3-2B-hf, Llama-3.2-11B-Vision, or Phi-4-multimodal")
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

    image_vec = None
    if args.image:
        import cv2
        ref_bgr = cv2.imread(args.image)
        if ref_bgr is None:
            print(f"Error: Could not read --image '{args.image}'", file=sys.stderr)
            sys.exit(1)
        image_vec = embedder.extract_multimodal_image(ref_bgr)

    vector_db = AuditoriumVectorDB(db_path=args.db_path, siglip_dim=embedder.siglip_dim)
    all_tracks_map = {str(r["track_id"]): r for r in vector_db.get_all_records()}

    router = QueryRouter(embedder)
    route, mentioned_ids, route_scores = router.classify(
        args.question,
        text_vec,
        forced_route=args.route,
        has_reference_image=bool(args.image),
    )
    score_str = ", ".join(f"{k}={v:.3f}" for k, v in route_scores.items())
    print(f"  -> Selected Route: [{route.upper()}] ({score_str})")

    candidates = []
    scene_frames = []
    object_records = []
    video_events = []
    structured_stats = None

    if route == "track_id":
        for mid in mentioned_ids:
            if mid in all_tracks_map:
                candidates.append(all_tracks_map[mid])
        if not candidates:
            candidates = vector_db.search_by_text_vector(text_vec, top_k=args.top_k)

    elif route == "counting":
        structured_stats = vector_db.get_structured_stats()
        scene_frames = vector_db.search_scene_frames(text_vec, top_k=3)
        candidates = vector_db.search_by_text_vector(text_vec, top_k=max(args.top_k, 5))

    elif route == "object_ownership":
        search_vec = image_vec if image_vec is not None else text_vec
        object_records = vector_db.search_objects_by_vector(search_vec, top_k=max(args.top_k, 5))
        # Always include explicitly abandoned or unattended-at-end objects for unattended/abandoned queries
        abandoned = [
            o for o in vector_db.get_all_objects()
            if o.get("status") in ("abandoned", "unattended_at_end")
        ]
        seen_obj_ids = {str(o.get("object_id")) for o in object_records}
        for ab in abandoned:
            if str(ab.get("object_id")) not in seen_obj_ids:
                object_records.insert(0, ab)
                seen_obj_ids.add(str(ab.get("object_id")))
        object_records = object_records[:max(args.top_k, 5)]

        # Join each object's bound owner_track_id -> person_tracks record
        seen_owners = set()
        for obj in object_records:
            oid = str(obj.get("owner_track_id", ""))
            if oid and oid in all_tracks_map and oid not in seen_owners:
                candidates.append(all_tracks_map[oid])
                seen_owners.add(oid)
        if not candidates:
            candidates = vector_db.search_by_text_vector(text_vec, top_k=args.top_k)

    elif route == "video_summary":
        all_events = vector_db.get_all_video_events()
        if all_events:
            sem_events = vector_db.search_video_events(text_vec, top_k=max(args.top_k, 5))
            # Merge top semantic hits + chronological coverage
            by_cid = {e["chunk_id"]: e for e in sem_events}
            for e in all_events:
                if len(by_cid) >= 8:
                    break
                by_cid.setdefault(e["chunk_id"], e)
            video_events = sorted(by_cid.values(), key=lambda r: float(r.get("start_sec", 0.0)))

            # Attach crops for referenced Global Track IDs across events
            seen_tids = set()
            for ev in video_events:
                try:
                    tids = json.loads(ev.get("referenced_track_ids_json", "[]"))
                except Exception:
                    tids = []
                for tid in tids:
                    tid = str(tid)
                    if tid in all_tracks_map and tid not in seen_tids:
                        candidates.append(all_tracks_map[tid])
                        seen_tids.add(tid)
        if not scene_frames:
            scene_frames = vector_db.search_scene_frames(text_vec, top_k=2)
        if not candidates:
            candidates = vector_db.search_by_text_vector(text_vec, top_k=max(args.top_k, 4))

    else:  # person_forensic
        search_vec = image_vec if image_vec is not None else text_vec
        candidates = vector_db.search_by_text_vector(search_vec, top_k=max(args.top_k, 5))

    if not candidates and not scene_frames and not object_records and not video_events:
        print("No matching records found in LanceDB. Please run pipeline.py first.")
        sys.exit(0)

    print(
        f"[2/3] Retrieved {len(candidates)} person(s), {len(object_records)} object(s), "
        f"{len(video_events)} event chunk(s), {len(scene_frames)} scene keyframe(s):"
    )
    for o in object_records:
        print(
            f"  - Object #{o.get('object_id')} ({o.get('label')}) | Status: {o.get('status')} "
            f"| Owner: Person #{o.get('owner_track_id')} | Time: {o.get('start_sec', 0.0):.1f}s - {o.get('end_sec', 0.0):.1f}s"
        )
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
        # Release retrieval models (SigLIP / OSNet) before mapping large VLM weights
        import gc
        del router
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        structured_stats=structured_stats,
        object_records=object_records,
        video_events=video_events,
        reference_image_path=args.image,
    )

    print("=" * 80)
    print("VLM ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
