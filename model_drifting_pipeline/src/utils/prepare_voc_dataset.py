import os
import json
from typing import Optional
import torchvision
from PIL import Image
from src.utils.perturbations import ImagePerturbation

VOC_CLASSES = ["person", "car", "bicycle", "bus", "motorcycle", "dog", "cat"]

def generate_voc_detection_datasets(base_dir: str = "data", samples: Optional[int] = None):
    """
    Downloads VOC 2007 test set and prepares:
    1. data/golden/voc_golden (clean reference detection dataset with annotations)
    2. data/new/voc_camera_degraded (blur + noise on VOC images)
    3. data/new/voc_unlabelled (unlabelled new client detection dataset)
    If samples is None or 0, prepares all available samples in the dataset.
    """
    print("Preparing PASCAL VOC Detection datasets...")
    voc_data = torchvision.datasets.VOCDetection(
        root="/tmp/voc",
        year="2007",
        image_set="test",
        download=True
    )

    golden_dir = os.path.join(base_dir, "golden", "voc_golden")
    os.makedirs(os.path.join(golden_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(golden_dir, "annotations"), exist_ok=True)

    degraded_dir = os.path.join(base_dir, "new", "voc_camera_degraded")
    os.makedirs(os.path.join(degraded_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(degraded_dir, "annotations"), exist_ok=True)

    unlabelled_dir = os.path.join(base_dir, "new", "voc_unlabelled")
    os.makedirs(os.path.join(unlabelled_dir, "images"), exist_ok=True)

    collected = 0
    total_available = len(voc_data)
    target_count = samples if (samples and samples > 0) else total_available
    print(f"Target preparation count: {target_count} images (out of {total_available} available)...")

    for idx, (img, ann) in enumerate(voc_data):
        objs = ann["annotation"].get("object", [])
        if isinstance(objs, dict):
            objs = [objs]
        
        valid_boxes = []
        valid_labels = []
        for obj in objs:
            cls_name = obj["name"]
            if cls_name in VOC_CLASSES:
                bnd = obj["bndbox"]
                valid_boxes.append([float(bnd["xmin"]), float(bnd["ymin"]), float(bnd["xmax"]), float(bnd["ymax"])])
                valid_labels.append(cls_name)

        if not valid_boxes:
            continue

        fname = f"voc_{collected:04d}.jpg"
        # Golden
        img.save(os.path.join(golden_dir, "images", fname))
        with open(os.path.join(golden_dir, "annotations", f"voc_{collected:04d}.json"), 'w') as f:
            json.dump({"boxes": valid_boxes, "labels": valid_labels}, f, indent=2)

        # Degraded
        deg_img = ImagePerturbation.apply_camera_degradation(img)
        deg_img.save(os.path.join(degraded_dir, "images", fname))
        with open(os.path.join(degraded_dir, "annotations", f"voc_{collected:04d}.json"), 'w') as f:
            json.dump({"boxes": valid_boxes, "labels": valid_labels}, f, indent=2)

        # Unlabelled
        deg_img.save(os.path.join(unlabelled_dir, "images", fname))

        collected += 1
        if collected >= target_count:
            break

    print(f"PASCAL VOC detection datasets successfully prepared! Total images: {collected}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare VOC detection reference datasets")
    parser.add_argument("--samples", type=int, default=None, help="Number of VOC samples to prepare (leave empty or use --all for full dataset)")
    parser.add_argument("--all", action="store_true", help="Download and prepare the complete dataset")
    args = parser.parse_args()

    num_samples = None if args.all else args.samples
    generate_voc_detection_datasets(samples=num_samples)

