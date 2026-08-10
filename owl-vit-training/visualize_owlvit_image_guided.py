from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import OwlViTForObjectDetection, OwlViTProcessor


def load_model_and_processor(checkpoint_path: str | None, model_name: str):
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_name = checkpoint["model_name"]

    processor_dir = Path(checkpoint_path).parent / "processor" if checkpoint_path else None
    if processor_dir and processor_dir.exists():
        processor = OwlViTProcessor.from_pretrained(processor_dir)
    else:
        processor = OwlViTProcessor.from_pretrained(model_name)

    model = OwlViTForObjectDetection.from_pretrained(model_name)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, processor


def load_query_image(
    query_image_path: str | None,
    query_source_image_path: str | None,
    query_box: list[int] | None,
    target_image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    if query_image_path:
        return Image.open(query_image_path).convert("RGB"), None

    if query_box:
        source_image = target_image
        if query_source_image_path:
            source_image = Image.open(query_source_image_path).convert("RGB")
        x1, y1, x2, y2 = query_box
        return source_image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

    raise ValueError("Provide either --query-image or --query-box.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize OWL-ViT image-guided detections.")
    parser.add_argument("--image", required=True, help="Target image path.")
    parser.add_argument("--output-image", required=True, help="Path to save rendered visualization.")
    parser.add_argument("--checkpoint", help="Optional fine-tuned checkpoint path.")
    parser.add_argument("--model-name", default="google/owlvit-base-patch32", help="HF model checkpoint to use.")
    parser.add_argument("--query-image", help="Optional exemplar crop image path.")
    parser.add_argument(
        "--query-source-image",
        help="Optional source image to crop from when using --query-box. Defaults to --image.",
    )
    parser.add_argument(
        "--query-box",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop coordinates for the exemplar patch.",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="NMS threshold.")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.checkpoint, args.model_name)

    target_image = Image.open(args.image).convert("RGB")
    query_image, query_box = load_query_image(args.query_image, args.query_source_image, args.query_box, target_image)

    inputs = processor(images=target_image, query_images=query_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)

    target_sizes = torch.tensor([(target_image.height, target_image.width)])
    results = processor.post_process_image_guided_detection(
        outputs=outputs,
        threshold=args.threshold,
        nms_threshold=args.nms_threshold,
        target_sizes=target_sizes,
    )[0]

    rendered = target_image.copy()
    draw = ImageDraw.Draw(rendered)

    if query_box:
        draw.rectangle(query_box, outline="cyan", width=4)
        draw.text((query_box[0] + 4, max(query_box[1] - 18, 0)), "QUERY", fill="cyan")

    for score, box in zip(results["scores"], results["boxes"]):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, max(y1 - 18, 0)), f"{float(score):.2f}", fill="red")

    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
