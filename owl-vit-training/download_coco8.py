from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.owlvit_dataset import COCO80_NAMES


COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def write_manifest(dataset_root: Path) -> None:
    manifest = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "classes": COCO80_NAMES,
    }
    for split in ("train", "val"):
        images = sorted((dataset_root / "images" / split).glob("*"))
        labels = sorted((dataset_root / "labels" / split).glob("*.txt"))
        manifest["splits"][split] = {
            "num_images": len(images),
            "num_labels": len(labels),
            "images": [image.name for image in images],
        }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the tiny COCO8 dataset for OWL-ViT experiments.")
    parser.add_argument("--data-dir", default="data", help="Directory where coco8 should be placed.")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite the existing zip file.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    archive_path = data_dir / "downloads" / "coco8.zip"
    dataset_root = data_dir / "coco8"

    if archive_path.exists() and not args.force:
        print(f"Using existing archive: {archive_path}")
    else:
        download_file(COCO8_URL, archive_path)

    if dataset_root.exists():
        print(f"Dataset already extracted at: {dataset_root}")
    else:
        print(f"Extracting {archive_path} -> {data_dir}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)

    write_manifest(dataset_root)
    (dataset_root / "dataset.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in COCO80_NAMES.items()],
            ]
        )
        + "\n"
    )
    print(f"Dataset ready at: {dataset_root}")
    print(f"Manifest written to: {dataset_root / 'manifest.json'}")
    print(f"Dataset config written to: {dataset_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
