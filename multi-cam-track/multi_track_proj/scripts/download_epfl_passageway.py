"""Download and convert the EPFL Passageway 1 multi-camera sequence.

Downloads 4 synchronized camera views from official EPFL CVLab:
  - cam01: passageway1-c0.avi
  - cam02: passageway1-c1.avi
  - cam03: passageway1-c2.avi
  - cam04: passageway1-c3.avi
Converts them to MP4 via OpenCV VideoWriter, and downloads official
ground truth and calibration files.
"""
import argparse
from pathlib import Path
import cv2
import requests
from tqdm import tqdm

URLS = {
    'cam01': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c0.avi',
    'cam02': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c1.avi',
    'cam03': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c2.avi',
    'cam04': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/passageway1-c3.avi',
}

GT_URL = 'https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_passageway1.txt'
CALIB_URL = 'https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-passageway.txt'


def download_file(url: str, dest_path: Path):
    """Download a file over HTTP with a progress bar."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0'}
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(dest_path, 'wb') as out, tqdm(total=total, unit='B', unit_scale=True, desc=f"  Downloading {dest_path.name}") as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))


def convert_avi_to_mp4(source_avi: Path, target_mp4: Path):
    """Convert AVI to standard MP4 using OpenCV VideoWriter."""
    if target_mp4.exists() and target_mp4.stat().st_size > 0:
        print(f"  Target MP4 already exists: {target_mp4.name}")
        return

    cap = cv2.VideoCapture(str(source_avi))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source_avi} with OpenCV.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(target_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {target_mp4}")

    with tqdm(total=total_frames or None, unit='frame', desc=f"  Converting {source_avi.name} -> {target_mp4.name}") as bar:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
            bar.update(1)

    cap.release()
    writer.release()
    if count == 0:
        raise RuntimeError(f"No frames decoded from {source_avi}")
    print(f"  Encoded {count} frames ({w}x{h} @ {fps:.1f} fps) -> {target_mp4.name}")


def main():
    parser = argparse.ArgumentParser(description="Download and convert EPFL Passageway 1 sequence")
    parser.add_argument('--output-dir', default='data/epfl_passageway', help='Destination root directory')
    parser.add_argument('--keep-avi', action='store_true', help='Retain raw AVI files after conversion')
    args = parser.parse_args()

    root = Path(args.output_dir)
    videos_dir = root / 'videos'
    gt_dir = root / 'ground_truth'
    raw_dir = root / 'raw_avi'

    raw_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"EPFL Passageway 1 Dataset Downloader")
    print(f"Destination: {root.resolve()}")
    print(f"=======================================================\n")

    # 1. Download & convert videos
    for cam_id, url in URLS.items():
        avi_path = raw_dir / f'{cam_id}.avi'
        mp4_path = videos_dir / f'{cam_id}.mp4'

        print(f"[{cam_id}] Checking / downloading video...")
        download_file(url, avi_path)
        convert_avi_to_mp4(avi_path, mp4_path)

        if not args.keep_avi and avi_path.exists():
            avi_path.unlink()

    if not args.keep_avi and raw_dir.exists():
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    # 2. Download ground truth and calibration
    print("\n[Annotations] Downloading ground truth and calibration files...")
    download_file(GT_URL, gt_dir / 'gt_passageway1.txt')
    download_file(CALIB_URL, gt_dir / 'calibration-passageway.txt')

    print(f"\nEPFL Passageway 1 dataset ready!")
    print(f"  Videos:       {videos_dir.resolve()}")
    print(f"  Ground truth: {gt_dir.resolve()}\n")


if __name__ == '__main__':
    main()

