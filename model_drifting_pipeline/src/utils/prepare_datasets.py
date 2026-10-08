import os
import json
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from src.utils.perturbations import ImagePerturbation

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def generate_cifar_datasets(base_dir: str = "data", samples_per_class: int = 15):
    """
    Downloads CIFAR-10 test set and creates:
    1. data/golden/cifar10_golden (clean reference dataset)
    2. data/new/cifar10_identical (identical copy -> No drift)
    3. data/new/cifar10_camera_degraded (blurred + sensor noise + poor exposure -> Quality & Input Drift)
    4. data/new/cifar10_distribution_shifted (70% animals, 30% vehicles vs balanced 50/50 -> Distribution Drift)
    5. data/new/cifar10_unlabelled (raw images without labels -> Unlabelled case)
    """
    print("Preparing CIFAR-10 data subsets...")
    dataset = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True)
    
    # Class-wise collection
    class_images = {c: [] for c in range(10)}
    for img, lbl in dataset:
        if len(class_images[lbl]) < samples_per_class * 3:
            class_images[lbl].append(img)
        if all(len(v) >= samples_per_class * 3 for v in class_images.values()):
            break

    # 1. Golden Reference Dataset (Balanced: 15 samples per class = 150 images)
    golden_dir = os.path.join(base_dir, "golden", "cifar10_golden")
    os.makedirs(os.path.join(golden_dir, "images"), exist_ok=True)
    golden_labels = {}

    idx = 0
    golden_imgs = []
    for c in range(10):
        cls_name = CIFAR_CLASSES[c]
        for img in class_images[c][:samples_per_class]:
            fname = f"img_{idx:04d}_{cls_name}.png"
            img.save(os.path.join(golden_dir, "images", fname))
            golden_labels[fname] = cls_name
            golden_imgs.append((img, fname, cls_name))
            idx += 1

    with open(os.path.join(golden_dir, "labels.json"), 'w') as f:
        json.dump(golden_labels, f, indent=2)
    print(f"Golden dataset created: {len(golden_labels)} samples at {golden_dir}")

    # 2. Identical Copy
    identical_dir = os.path.join(base_dir, "new", "cifar10_identical")
    os.makedirs(os.path.join(identical_dir, "images"), exist_ok=True)
    ident_labels = {}
    for img, fname, cls_name in golden_imgs:
        img.save(os.path.join(identical_dir, "images", fname))
        ident_labels[fname] = cls_name
    with open(os.path.join(identical_dir, "labels.json"), 'w') as f:
        json.dump(ident_labels, f, indent=2)

    # 3. Camera Degraded (Blur + Noise + Brightness drop)
    degraded_dir = os.path.join(base_dir, "new", "cifar10_camera_degraded")
    os.makedirs(os.path.join(degraded_dir, "images"), exist_ok=True)
    deg_labels = {}
    for img, fname, cls_name in golden_imgs:
        degraded = ImagePerturbation.apply_camera_degradation(img)
        degraded.save(os.path.join(degraded_dir, "images", fname))
        deg_labels[fname] = cls_name
    with open(os.path.join(degraded_dir, "labels.json"), 'w') as f:
        json.dump(deg_labels, f, indent=2)

    # 4. Distribution Shifted (Heavily weighted towards 'dog' and 'cat', few vehicles)
    shifted_dir = os.path.join(base_dir, "new", "cifar10_distribution_shifted")
    os.makedirs(os.path.join(shifted_dir, "images"), exist_ok=True)
    shift_labels = {}
    s_idx = 0
    # Add 40 dogs, 40 cats, 5 cars, 5 airplanes
    spec = [(5, 40), (3, 40), (1, 5), (0, 5)] # (class_id, count)
    for cid, count in spec:
        cls_name = CIFAR_CLASSES[cid]
        for img in class_images[cid][:count]:
            fname = f"shift_{s_idx:04d}_{cls_name}.png"
            img.save(os.path.join(shifted_dir, "images", fname))
            shift_labels[fname] = cls_name
            s_idx += 1
    with open(os.path.join(shifted_dir, "labels.json"), 'w') as f:
        json.dump(shift_labels, f, indent=2)

    # 5. Unlabelled New Client Dataset (Camera Degraded, without labels.json)
    unlabelled_dir = os.path.join(base_dir, "new", "cifar10_unlabelled")
    os.makedirs(os.path.join(unlabelled_dir, "images"), exist_ok=True)
    for img, fname, _ in golden_imgs:
        degraded = ImagePerturbation.apply_gaussian_blur(img, radius=2.0)
        degraded.save(os.path.join(unlabelled_dir, "images", fname))

    print("All CIFAR test datasets successfully prepared!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 reference and drift datasets")
    parser.add_argument("--samples_per_class", type=int, default=15, help="Number of samples per class for golden baseline (default: 15, use 100+ for large runs)")
    parser.add_argument("--all", action="store_true", help="Download and prepare full dataset subset (100 samples/class = 1000 images)")
    args = parser.parse_args()

    count = 100 if args.all else args.samples_per_class
    generate_cifar_datasets(samples_per_class=count)
