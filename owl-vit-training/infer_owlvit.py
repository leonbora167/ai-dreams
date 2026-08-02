from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned OWL-ViT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint produced by training.")
    parser.add_argument("--image", required=True, help="Path to an image.")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Candidate labels, for example: --labels person bicycle car",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Confidence threshold.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]

    processor_dir = Path(args.checkpoint).parent / "processor"
    if processor_dir.exists():
        processor = OwlViTProcessor.from_pretrained(processor_dir)
    else:
        processor = OwlViTProcessor.from_pretrained(model_name)

    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    labels = args.labels or checkpoint.get("class_names")
    if not labels:
        raise ValueError("No labels supplied and checkpoint does not contain class_names.")

    prompt_template = checkpoint.get("prompt_template", "a photo of a {}")
    text_labels = [[prompt_template.format(label) for label in labels]]

    inputs = processor(text=text_labels, images=image, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=args.threshold,
        text_labels=text_labels,
    )[0]

    if len(results["boxes"]) == 0:
        print("No detections above threshold.")
        return

    for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
        coords = [round(float(value), 2) for value in box.tolist()]
        print(f"{label}: score={float(score):.4f}, box={coords}")


if __name__ == "__main__":
    main()
