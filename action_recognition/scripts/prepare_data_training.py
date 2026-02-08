import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_videos(source_root: Path) -> dict[str, list[Path]]:
    videos_by_class: dict[str, list[Path]] = defaultdict(list)
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = path.relative_to(source_root)
        class_name = rel.parts[0]
        videos_by_class[class_name].append(path)
    return dict(videos_by_class)


def split_items(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def copy_split(
    class_name: str,
    split_name: str,
    paths: list[Path],
    source_root: Path,
    out_root: Path,
) -> int:
    split_dir = out_root / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in paths:
        rel = src.relative_to(source_root)
        flat_name = "__".join(rel.parts[1:])
        dst = split_dir / flat_name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/raw/airtlab_dataset/violence-detection-dataset",
    )
    parser.add_argument("--output-root", type=str, default="data_training")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    videos_by_class = discover_videos(source_root)
    if not videos_by_class:
        raise ValueError(f"No videos found under {source_root}")

    total_train = 0
    total_val = 0
    for class_name in sorted(videos_by_class):
        items = sorted(videos_by_class[class_name])
        train_items, val_items = split_items(items, args.val_ratio, args.seed)
        total_train += copy_split(class_name, "train", train_items, source_root, out_root)
        total_val += copy_split(class_name, "val", val_items, source_root, out_root)
        print(
            f"class={class_name} total={len(items)} "
            f"train={len(train_items)} val={len(val_items)}"
        )

    print(
        f"prepared dataset at {out_root} (new copies: train={total_train}, val={total_val})"
    )


if __name__ == "__main__":
    main()
