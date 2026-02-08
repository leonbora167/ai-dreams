import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


DEFAULT_DATASET_ZIP_URL = (
    "https://github.com/airtlab/"
    "A-Dataset-for-Automatic-Violence-Detection-in-Videos/archive/refs/heads/master.zip"
)


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response:  # nosec B310
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024
        with dest.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc="downloading dataset",
            dynamic_ncols=True,
        ) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def find_dataset_root(extract_root: Path) -> Path:
    candidates = list(extract_root.rglob("violence-detection-dataset"))
    if not candidates:
        raise ValueError("Could not find 'violence-detection-dataset' in downloaded archive.")
    return candidates[0]


def run_command(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=DEFAULT_DATASET_ZIP_URL)
    parser.add_argument("--raw-root", type=str, default="data/raw/airtlab_dataset")
    parser.add_argument("--data-training-root", type=str, default="data_training")
    parser.add_argument("--manifest-path", type=str, default="data_training/manifest.csv")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_root = (project_root / args.raw_root).resolve()
    data_training_root = (project_root / args.data_training_root).resolve()
    manifest_path = (project_root / args.manifest_path).resolve()

    raw_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="airtlab_dl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "airtlab.zip"

        print(f"downloading from: {args.url}")
        download_file(args.url, zip_path)

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print("extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        dataset_root = find_dataset_root(extract_dir)
        target_dataset_root = raw_root / "violence-detection-dataset"
        target_dataset_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"copying dataset to: {target_dataset_root}")
        shutil.copytree(dataset_root, target_dataset_root, dirs_exist_ok=True)

        extracted_readmes = list(extract_dir.rglob("README.md"))
        if extracted_readmes:
            shutil.copy2(extracted_readmes[0], raw_root / "readme.md")

    print("download + extraction complete")

    if args.download_only:
        print("download-only mode enabled. Skipping data_training preparation.")
        return

    run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_data_training.py"),
            "--source-root",
            str(target_dataset_root),
            "--output-root",
            str(data_training_root),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
    )

    run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "build_manifest.py"),
            "--data-root",
            str(data_training_root),
            "--output",
            str(manifest_path),
        ]
    )

    print("dataset is ready for training")
    print(f"raw dataset: {target_dataset_root}")
    print(f"training data: {data_training_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
