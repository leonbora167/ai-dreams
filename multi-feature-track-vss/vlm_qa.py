import argparse
import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from embedder import LocalEmbedder
from vector_store import AuditoriumVectorDB

def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"

class LocalVLMQA:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        print(f"Loading VLM ({model_id}) on {self.device}...")
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if "smolvlm" in model_id.lower():
            self.model_family = "smolvlm"
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            self.model_family = "qwen"
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

    def answer_question(self, question: str, candidate_records: list[dict]) -> str:
        """
        RAG VLM answering using retrieved tracklet crops and metadata.
        """
        if not candidate_records:
            return "No matching person tracks were found in the video database."

        context_lines = []
        for i, rec in enumerate(candidate_records, 1):
            track_id = rec.get("track_id", "Unknown")
            start_sec = rec.get("start_sec", 0.0)
            end_sec = rec.get("end_sec", 0.0)
            start_f = rec.get("start_frame", 0)
            end_f = rec.get("end_frame", 0)
            time_str = f"{start_sec:.1f}s to {end_sec:.1f}s ({format_timestamp(start_sec)} - {format_timestamp(end_sec)})"
            context_lines.append(f"- Candidate #{i} -> Track ID: #{track_id}, Timestamp: {time_str}, Frames: {start_f}-{end_f}")

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

        if self.model_family == "smolvlm":
            content = []
            images = []
            for i, rec in enumerate(candidate_records, 1):
                crop_path = Path(rec.get("best_crop_path", ""))
                track_id = rec.get("track_id", "Unknown")
                start_sec = rec.get("start_sec", 0.0)
                end_sec = rec.get("end_sec", 0.0)
                time_str = f"{start_sec:.1f}s to {end_sec:.1f}s"
                if crop_path.exists():
                    try:
                        img = Image.open(str(crop_path)).convert("RGB")
                        images.append(img)
                        content.append({"type": "image"})
                        content.append({
                            "type": "text",
                            "text": f"[Candidate #{i} / Track ID #{track_id} | Visible: {time_str}]\n"
                        })
                    except Exception as e:
                        print(f"Warning: Could not open {crop_path}: {e}", file=sys.stderr)

            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                decoded = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                if "Assistant:" in decoded:
                    response = decoded.split("Assistant:")[-1].strip()
                else:
                    response = decoded.strip()
            return response

        else:
            from qwen_vl_utils import process_vision_info
            content = []
            for i, rec in enumerate(candidate_records, 1):
                crop_path = Path(rec.get("best_crop_path", ""))
                track_id = rec.get("track_id", "Unknown")
                start_sec = rec.get("start_sec", 0.0)
                end_sec = rec.get("end_sec", 0.0)
                time_str = f"{start_sec:.1f}s to {end_sec:.1f}s"
                if crop_path.exists():
                    try:
                        img = Image.open(str(crop_path)).convert("RGB")
                        content.append({"type": "image", "image": img})
                        content.append({
                            "type": "text",
                            "text": f"[Candidate #{i} / Track ID #{track_id} | Visible: {time_str}]\n"
                        })
                    except Exception as e:
                        print(f"Warning: Could not open {crop_path}: {e}", file=sys.stderr)

            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Auditorium Post-Execution VLM Q&A CLI")
    parser.add_argument("--question", type=str, required=True, help="Natural language question to ask about people in the video")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate tracklets to feed into the VLM")
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-Instruct", 
                        help="VLM model ID (e.g. 'HuggingFaceTB/SmolVLM-Instruct' or 'Qwen/Qwen2-VL-2B-Instruct')")
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

    print(f"\n[3/3] Prompting local VLM ({args.model_id})...\n")
    vlm = LocalVLMQA(model_id=args.model_id)
    answer = vlm.answer_question(args.question, candidates)

    print("=" * 80)
    print("VLM ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
