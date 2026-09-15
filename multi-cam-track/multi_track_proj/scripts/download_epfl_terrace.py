"""Download and convert the EPFL Terrace 1 multi-camera sequence.

Downloads synchronized multi-view AVI streams (4 cameras) from official EPFL CVLab,
converts them into MP4 files using ffmpeg, and downloads ground truth and calibration files.
"""
import argparse
import os
from pathlib import Path
import subprocess
import urllib.request
import requests
from tqdm import tqdm

URLS = {
    'cam01': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c0.avi',
    'cam02': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c1.avi',
    'cam03': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c2.avi',
    'cam04': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c3.avi',
}

GT_URL = 'https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_terrace1.txt'
CALIB_URL = 'https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-terrace.txt'


def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"File already exists: {dest_path.name}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(dest_path, 'wb') as out, tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))


def convert_avi_to_mp4(source_avi: Path, target_mp4: Path):
    """Convert AVI to MP4 using OpenCV VideoWriter (with ffmpeg fallback)."""
    if target_mp4.exists() and target_mp4.stat().st_size > 0:
        print(f"Video already exists: {target_mp4.name}")
        return

    target_mp4.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cap = cv2.VideoCapture(str(source_avi))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        writer = cv2.VideoWriter(str(target_mp4), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if writer.isOpened():
            with tqdm(total=total_frames or None, unit='frame', desc=f"Converting {source_avi.name} -> {target_mp4.name}") as bar:
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
            if count > 0:
                print(f"Successfully created: {target_mp4.name} ({count} frames @ {fps:.1f} fps)")
                return

    # Fallback to ffmpeg if installed
    print(f"Converting {source_avi.name} -> {target_mp4.name} via ffmpeg...")
    cmd = [
        'ffmpeg', '-y', '-i', str(source_avi),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
        '-an', str(target_mp4)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Conversion failed for {source_avi}:\n{res.stderr}")
    print(f"Successfully created: {target_mp4.name}")


def main():
    parser = argparse.ArgumentParser(description="Download EPFL Terrace 1 sequence")
    parser.add_argument('--output-dir', default='data/epfl_terrace', help='Root destination folder')
    parser.add_argument('--keep-avi', action='store_true', help='Keep original AVI files after conversion')
    args = parser.parse_args()

    root = Path(args.output_dir)
    videos_dir = root / 'videos'
    gt_dir = root / 'ground_truth'
    raw_dir = root / 'raw_avi'

    raw_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and convert videos
    for cam_id, url in URLS.items():
        avi_file = raw_dir / f'{cam_id}.avi'
        mp4_file = videos_dir / f'{cam_id}.mp4'

        download_file(url, avi_file)
        convert_avi_to_mp4(avi_file, mp4_file)

        if not args.keep_avi and avi_file.exists():
            avi_file.unlink()

    if not args.keep_avi and raw_dir.exists():
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    # 2. Download ground truth & calibration
    download_file(GT_URL, gt_dir / 'gt_terrace1.txt')
    download_file(CALIB_URL, gt_dir / 'calibration-terrace.txt')

    print("\nEPFL Terrace 1 sequence ready!")
    print(f"Videos:       {videos_dir.resolve()}")
    print(f"Ground truth: {gt_dir.resolve()}")


if __name__ == '__main__':
    main()
