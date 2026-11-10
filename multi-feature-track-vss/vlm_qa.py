import argparse
import base64
import json
import sys
import types
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import torch

from embedder import LocalEmbedder
from vector_store import AuditoriumVectorDB

def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"

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
        candidate_records: list[dict],
        images: list[Image.Image]
    ) -> str:
        pass

# ==============================================================================
# 1. Local Hugging Face Backend (SmolVLM2, Llama 3.2 Vision, Phi-4, Generic HF)
# ==============================================================================

class HuggingFaceLocalBackend(BaseVLMBackend):
    """
    Executes open-source multimodal models from Hugging Face.
    Handles the different input preparation conventions across model families:
      - SmolVLM2 (SigLIP + SmolLM2)
      - Llama 3.2 Vision (Mllama)
      - Phi-4 Multimodal (Phi4MM)
      - Generic AutoModelForImageTextToText
    """
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        print(f"Loading local VLM ({model_id}) on {self.device}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_lower = model_id.lower()

        # Family 1: SmolVLM / SmolVLM2
        if "smolvlm" in model_lower:
            self.model_family = "smolvlm"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Upstream transformers patch for SmolVLM2 dynamic image token dimensions
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
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        # Family 3: Microsoft Phi-4 Multimodal
        elif "phi-4" in model_lower or "phi4" in model_lower:
            self.model_family = "phi4"
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Family 4: Generic Hugging Face Multimodal Fallback
        else:
            self.model_family = "generic_hf"
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def generate_answer(
        self,
        question: str,
        prompt_text: str,
        candidate_records: list[dict],
        images: list[Image.Image]
    ) -> str:
        # ------------------------------------------------------------------
        # Pattern A: Chat-Template Interleaved Multi-Image (SmolVLM2 & Llama 3.2)
        # Structure: One single prompt containing interleaved `{"type": "image"}`
        # tags and metadata text, passed alongside `images: list[Image.Image]`.
        # ------------------------------------------------------------------
        if self.model_family in ("smolvlm", "llama"):
            content = []
            for i, rec in enumerate(candidate_records, 1):
                track_id = rec.get("track_id", "Unknown")
                start_sec = rec.get("start_sec", 0.0)
                end_sec = rec.get("end_sec", 0.0)
                content.append({"type": "image"})
                content.append({
                    "type": "text",
                    "text": f"[Candidate #{i} / Track ID #{track_id} | Visible: {start_sec:.1f}s - {end_sec:.1f}s]\n"
                })
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=300)
                decoded = self.processor.decode(out[0], skip_special_tokens=True)
                if self.model_family == "llama" and "assistant\n" in decoded:
                    return decoded.split("assistant\n")[-1].strip()
                elif "Assistant:" in decoded:
                    return decoded.split("Assistant:")[-1].strip()
                return decoded.strip()

        # ------------------------------------------------------------------
        # Pattern B: Numbered Special Token Markup (Phi-4 Multimodal)
        # Structure: Explicit indexed tokens `<|image_1|> ... <|image_n|>` in text,
        # with matching list of images passed to processor.
        # ------------------------------------------------------------------
        elif self.model_family == "phi4":
            image_tags = "".join([f"<|image_{i}|>\n" for i in range(1, len(images) + 1)])
            formatted_prompt = f"<|user|>\n{image_tags}{prompt_text}\n<|end|>\n<|assistant|>\n"
            inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=300)
                decoded = self.processor.decode(out[0], skip_special_tokens=True)
                return decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded.strip()

        # ------------------------------------------------------------------
        # Pattern C: Generic Fallback (Standard HF Multimodal)
        # Attempts apply_chat_template if supported, else direct text + image list.
        # ------------------------------------------------------------------
        else:
            try:
                content = []
                for i, rec in enumerate(candidate_records, 1):
                    track_id = rec.get("track_id", "Unknown")
                    start_sec = rec.get("start_sec", 0.0)
                    end_sec = rec.get("end_sec", 0.0)
                    content.append({"type": "image"})
                    content.append({
                        "type": "text",
                        "text": f"[Candidate #{i} / Track ID #{track_id} | Visible: {start_sec:.1f}s - {end_sec:.1f}s]\n"
                    })
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
            except Exception:
                # Direct format fallback if chat_template does not support dict content
                inputs = self.processor(text=prompt_text, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=300)
                return self.processor.decode(out[0], skip_special_tokens=True).strip()

# ==============================================================================
# 2. Remote OpenAI-Compatible Endpoint Backend (Nemotron, vLLM, NIM, Ollama)
# ==============================================================================

class OpenAICompatibleRemoteBackend(BaseVLMBackend):
    """
    Integrates with self-hosted large models (NVIDIA Nemotron, Llama-70B, etc.)
    served via vLLM, NVIDIA NIM, TGI, Ollama, SGLang, or custom microservices.
    Standard: OpenAI Chat Completions Vision API format.
    """
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
        candidate_records: list[dict],
        images: list[Image.Image]
    ) -> str:
        content_items = []
        
        # Add visual crops with metadata as base64 data URIs
        for i, (rec, img) in enumerate(zip(candidate_records, images), 1):
            track_id = rec.get("track_id", "Unknown")
            start_sec = rec.get("start_sec", 0.0)
            end_sec = rec.get("end_sec", 0.0)
            data_uri = self._pil_to_base64_data_uri(img)
            
            content_items.append({
                "type": "text",
                "text": f"[Candidate #{i} / Track ID #{track_id} | Visible: {start_sec:.1f}s - {end_sec:.1f}s]:"
            })
            content_items.append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })

        # Add the instructions and user question
        content_items.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_items
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AuditoriumVSS/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.endpoint_url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                choices = resp_json.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "").strip()
                return str(resp_json)
        except Exception as e:
            return f"Error communicating with remote model endpoint ({self.endpoint_url}): {e}"

# ==============================================================================
# Orchestrator Class
# ==============================================================================

class LocalVLMQA:
    def __init__(self, backend: BaseVLMBackend):
        self.backend = backend

    def answer_question(self, question: str, candidate_records: list[dict]) -> str:
        if not candidate_records:
            return "No matching person tracks were found in the video database."

        context_lines = []
        images = []
        valid_records = []

        for i, rec in enumerate(candidate_records, 1):
            track_id = rec.get("track_id", "Unknown")
            start_sec = rec.get("start_sec", 0.0)
            end_sec = rec.get("end_sec", 0.0)
            start_f = rec.get("start_frame", 0)
            end_f = rec.get("end_frame", 0)
            time_str = f"{start_sec:.1f}s to {end_sec:.1f}s ({format_timestamp(start_sec)} - {format_timestamp(end_sec)})"
            context_lines.append(f"- Candidate #{i} -> Track ID: #{track_id}, Timeline: {time_str}, Frames: {start_f}-{end_f}")

            crop_path = Path(rec.get("best_crop_path", ""))
            if crop_path.exists():
                try:
                    img = Image.open(str(crop_path)).convert("RGB")
                    images.append(img)
                    valid_records.append(rec)
                except Exception as e:
                    print(f"Warning: Could not open {crop_path}: {e}", file=sys.stderr)

        timeline_context = "\n".join(context_lines)
        prompt_text = (
            f"You are an AI video surveillance analyst assistant.\n"
            f"Below are the top visual candidate tracklets retrieved from the auditorium video database with their timestamps:\n\n"
            f"{timeline_context}\n\n"
            f"User Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Examine the visual appearance (clothing color, posture, appearance) of each candidate image.\n"
            f"2. Determine which candidate(s) answer the question.\n"
            f"3. Provide a concise, factual answer citing the exact Track ID and timestamp interval.\n"
            f"4. If none match what is asked, explicitly state that."
        )

        return self.backend.generate_answer(question, prompt_text, valid_records, images)

# ==============================================================================
# Main CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auditorium Post-Execution VLM Q&A CLI")
    parser.add_argument("--question", type=str, required=True, help="Natural language question to ask about people in the video")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate tracklets to feed into the VLM")
    
    # Model Selection: Local HuggingFace vs Remote Endpoint
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", 
                        help="Hugging Face local model ID: 'HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct', or 'microsoft/Phi-4-multimodal-instruct'")
    parser.add_argument("--endpoint-url", type=str, default=None,
                        help="Optional remote OpenAI-compatible server URL (e.g. 'http://your-server:8000/v1/chat/completions') for Nemotron, vLLM, NIM, Ollama")
    parser.add_argument("--model-name", type=str, default="nvidia/nemotron-4-340b",
                        help="Model name identifier for the remote endpoint")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Bearer API key for the remote endpoint (if required)")
    args = parser.parse_args()

    print(f"\n[1/3] Searching vector database for candidates matching: \"{args.question}\"...")
    embedder = LocalEmbedder()
    text_vec = embedder.extract_multimodal_text(args.question)

    vector_db = AuditoriumVectorDB()
    candidates = vector_db.search_by_text_vector(text_vec, top_k=args.top_k)

    if not candidates:
        print("No tracklets found in LanceDB. Please run pipeline.py first.")
        sys.exit(0)

    print(f"[2/3] Retrieved {len(candidates)} candidate tracklet(s):")
    for c in candidates:
        print(f"  - Track #{c.get('track_id')} (Distance: {c.get('_distance', 0.0):.4f}, Time: {c.get('start_sec', 0.0):.1f}s - {c.get('end_sec', 0.0):.1f}s)")

    # Decide Backend: Remote Endpoint vs Local HuggingFace
    if args.endpoint_url:
        print(f"\n[3/3] Sending request to remote server ({args.endpoint_url})...\n")
        backend = OpenAICompatibleRemoteBackend(
            endpoint_url=args.endpoint_url,
            model_name=args.model_name,
            api_key=args.api_key
        )
    else:
        print(f"\n[3/3] Prompting local VLM ({args.model_id})...\n")
        backend = HuggingFaceLocalBackend(model_id=args.model_id)

    vlm = LocalVLMQA(backend=backend)
    answer = vlm.answer_question(args.question, candidates)

    print("=" * 80)
    print("VLM ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
