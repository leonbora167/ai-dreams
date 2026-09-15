"""Download and prepare the PETS 2009 (S2.L1) multi-camera pedestrian dataset.

Downloads synchronized multi-view frames (Views 001, 005, 006, 008) via sparse git
checkout, converts them into standard MP4 files, downloads official ground truth
CVML annotations, parses them into MOT format and JSON, and copies camera calibration.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import cv2
from tqdm import tqdm


REPO_URL = 'https://github.com/Cosmos-Krishna/pets2009-cross-camera-reid.git'
GT_URL = 'https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/PETS-2009/data/annotations/PETS2009-S2L1.xml'

VIEW_MAP = {
    'cam01': 'View_001',
    'cam02': 'View_005',
    'cam03': 'View_006',
    'cam04': 'View_008',
}
FPS = 7.0  # Official capture rate for PETS 2009


def clone_sparse_repo(repo_dir: Path):
    """Clone repo using sparse checkout to fetch only necessary views and calibration."""
    if not repo_dir.exists():
        print(f"Cloning repository metadata from {REPO_URL}...")
        cmd = [
            'git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
            REPO_URL, str(repo_dir)
        ]
        subprocess.run(cmd, check=True)

    views_to_fetch = [f'data/{v}' for v in VIEW_MAP.values()]
    targets = ['calibration', 'csv'] + views_to_fetch

    print(f"Updating sparse-checkout for {len(targets)} targets...")
    cmd = ['git', 'sparse-checkout', 'set'] + targets
    subprocess.run(cmd, cwd=str(repo_dir), check=True)


def convert_images_to_mp4(images_dir: Path, output_mp4: Path, fps: float = FPS):
    """Convert an ordered folder of JPG frames to an MP4 video."""
    if output_mp4.exists():
        print(f"Video already exists: {output_mp4.name}")
        return

    frames = sorted(images_dir.glob('*.jpg'))
    if not frames:
        raise RuntimeError(f"No JPG frames found in {images_dir}")

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame from {frames[0]}")
    h, w, _ = first_frame.shape

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create VideoWriter for {output_mp4}")

    print(f"Encoding {len(frames)} frames into {output_mp4.name} ({w}x{h} @ {fps} fps)...")
    for frame_path in tqdm(frames, desc=output_mp4.stem):
        img = cv2.imread(str(frame_path))
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"Saved: {output_mp4}")


def download_ground_truth(gt_dir: Path):
    """Download official CVML XML annotations and convert to MOT txt & JSON format."""
    gt_dir.mkdir(parents=True, exist_ok=True)
    xml_path = gt_dir / 'PETS2009-S2L1.xml'

    if not xml_path.exists():
        print(f"Downloading ground truth XML from {GT_URL}...")
        urllib.request.urlretrieve(GT_URL, str(xml_path))

    with open(xml_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Parse CVML XML
    frame_blocks = re.findall(r'<frame number="(\d+)">(.*?)</frame>', text, re.DOTALL)
    mot_rows = []
    json_frames = {}

    for fnum_str, body in frame_blocks:
        fid0 = int(fnum_str)
        fid1 = fid0 + 1  # 1-indexed for standard MOT format
        frame_objs = []
        for oid_str, box_str in re.findall(r'<object id="(\d+)">\s*<box ([^/]+)/>', body):
            attrs = dict(re.findall(r'(\w+)="([-\d.]+)"', box_str))
            h = float(attrs['h'])
            w = float(attrs['w'])
            xc = float(attrs['xc'])
            yc = float(attrs['yc'])
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
            oid = int(oid_str)

            # MOT Challenge: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            mot_rows.append(f"{fid1},{oid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")
            frame_objs.append({
                'id': oid,
                'bbox_xyxy': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                'confidence': 1.0,
            })
        json_frames[fid0] = frame_objs

    # Save MOT txt
    mot_path = gt_dir / 'cam01_gt_mot.txt'
    with open(mot_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mot_rows) + '\n')

    # Save JSON
    json_path = gt_dir / 'cam01_gt.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_frames, f, indent=2)

    print(f"Ground truth parsed: {len(mot_rows)} boxes across {len(json_frames)} frames.")
    print(f"  MOT format:  {mot_path}")
    print(f"  JSON format: {json_path}")


def copy_calibration(repo_dir: Path, calib_dest: Path):
    """Copy homography calibration matrices and points."""
    src_calib = repo_dir / 'calibration'
    if src_calib.exists():
        calib_dest.mkdir(parents=True, exist_ok=True)
        for item in src_calib.glob('*'):
            if item.is_file():
                shutil.copy2(item, calib_dest / item.name)
        print(f"Calibration data copied to {calib_dest}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare PETS 2009 S2.L1 dataset")
    parser.add_argument('--output-dir', default='data/pets2009', help='Root directory for dataset')
    parser.add_argument('--cache-repo', default='data/pets2009_repo', help='Directory for raw git checkout')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw frame folders after encoding')
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    videos_dir = out_root / 'videos'
    gt_dir = out_root / 'ground_truth'
    calib_dir = out_root / 'calibration'
    repo_dir = Path(args.cache_repo)

    # 1. Sparse clone / fetch
    clone_sparse_repo(repo_dir)

    # 2. Convert views to MP4
    for cam_id, view_name in VIEW_MAP.items():
        view_images_dir = repo_dir / 'data' / view_name
        dest_mp4 = videos_dir / f'{cam_id}.mp4'
        convert_images_to_mp4(view_images_dir, dest_mp4, fps=FPS)

    # 3. Download and parse ground truth
    download_ground_truth(gt_dir)

    # 4. Copy calibration data
    copy_calibration(repo_dir, calib_dir)

    print("\nPETS 2009 Dataset preparation complete!")
    print(f"Videos:        {videos_dir.resolve()}")
    print(f"Ground truth:  {gt_dir.resolve()}")
    print(f"Calibration:   {calib_dir.resolve()}")


if __name__ == '__main__':
    main()

