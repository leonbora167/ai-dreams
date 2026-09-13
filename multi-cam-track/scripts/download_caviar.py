"""Download and prepare the CAVIAR (Corridor) multi-camera pedestrian dataset.

Downloads synchronized multi-camera views (Corridor and Frontal views) from the
official University of Edinburgh CAVIAR repository, converts them to MP4,
and parses official CVML XML annotations into standard MOT format and JSON.
"""
import argparse
import json
from pathlib import Path
import re
import cv2
import requests
from tqdm import tqdm

BASE_URL = 'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/'

SEQUENCES = {
    'TwoEnterShop1': {
        'cam01': {
            'video_url': f'{BASE_URL}TwoEnterShop1cor/TwoEnterShop1cor.mpg',
            'gt_url': f'{BASE_URL}TwoEnterShop1cor/c2es1gt.xml',
            'name': 'corridor',
        },
        'cam02': {
            'video_url': f'{BASE_URL}TwoEnterShop1front/TwoEnterShop1front.mpg',
            'gt_url': f'{BASE_URL}TwoEnterShop1front/f2es1gt.xml',
            'name': 'frontal',
        },
    }
}


def download_file(url: str, dest_path: Path):
    """Download a remote file with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
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


def convert_mpg_to_mp4(source_mpg: Path, target_mp4: Path, fps: float = 25.0):
    """Convert an MPG video to MP4 using OpenCV VideoWriter."""
    if target_mp4.exists() and target_mp4.stat().st_size > 0:
        print(f"  Video already exists: {target_mp4.name}")
        return

    cap = cv2.VideoCapture(str(source_mpg))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source_mpg} with OpenCV.")

    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(target_mp4), fourcc, cap_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create VideoWriter for {target_mp4}")

    with tqdm(total=total_frames or None, unit='frame', desc=f"  Encoding {source_mpg.name} -> {target_mp4.name}") as bar:
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
        raise RuntimeError(f"No frames decoded from {source_mpg}")
    print(f"  Encoded {count} frames ({w}x{h} @ {cap_fps:.1f} fps) -> {target_mp4.name}")


def parse_cvml_xml(xml_path: Path, gt_json_path: Path, gt_mot_path: Path):
    """Parse CAVIAR CVML XML annotations into JSON and standard MOT format."""
    text = xml_path.read_text(encoding='utf-8', errors='ignore')
    frame_blocks = re.findall(r'<frame number="(\d+)">(.*?)</frame>', text, re.DOTALL)

    mot_rows = []
    json_frames = {}

    for fnum_str, body in frame_blocks:
        fid0 = int(fnum_str)
        fid1 = fid0 + 1  # 1-indexed for MOTChallenge format
        frame_objs = []

        for oid_str, box_str in re.findall(r'<object id="(\d+)">.*?<box ([^/]+)/>', body, re.DOTALL):
            attrs = dict(re.findall(r'(\w+)="([-\d.]+)"', box_str))
            h = float(attrs.get('h', 0))
            w = float(attrs.get('w', 0))
            xc = float(attrs.get('xc', 0))
            yc = float(attrs.get('yc', 0))

            if h <= 0 or w <= 0:
                continue

            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
            oid = int(oid_str)

            mot_rows.append(f"{fid1},{oid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1")
            frame_objs.append({
                'id': oid,
                'bbox_xyxy': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                'confidence': 1.0,
            })

        if frame_objs:
            json_frames[fid0] = frame_objs

    # Save JSON and MOT formats
    gt_json_path.parent.mkdir(parents=True, exist_ok=True)
    gt_json_path.write_text(json.dumps(json_frames, indent=2), encoding='utf-8')
    gt_mot_path.write_text('\n'.join(mot_rows) + '\n', encoding='utf-8')

    print(f"  Parsed {len(mot_rows)} ground truth bounding boxes across {len(json_frames)} frames.")
    print(f"    JSON GT: {gt_json_path}")
    print(f"    MOT GT:  {gt_mot_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CAVIAR multi-camera dataset")
    parser.add_argument('--sequence', default='TwoEnterShop1', choices=list(SEQUENCES.keys()), help='Sequence name')
    parser.add_argument('--output-dir', default='data/caviar', help='Destination root directory')
    parser.add_argument('--keep-raw', action='store_true', help='Retain downloaded raw MPG files')
    args = parser.parse_args()

    seq_info = SEQUENCES[args.sequence]
    root = Path(args.output_dir)
    videos_dir = root / 'videos'
    gt_dir = root / 'ground_truth'
    raw_dir = root / 'raw_mpg'

    raw_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"CAVIAR Dataset Downloader: {args.sequence}")
    print(f"Destination: {root.resolve()}")
    print(f"=======================================================\n")

    for cam_id, info in seq_info.items():
        print(f"[{cam_id}: {info['name']}] Downloading and processing...")
        mpg_path = raw_dir / f"{cam_id}.mpg"
        mp4_path = videos_dir / f"{cam_id}.mp4"
        xml_path = gt_dir / f"{cam_id}_raw.xml"

        # 1. Download video
        download_file(info['video_url'], mpg_path)
        convert_mpg_to_mp4(mpg_path, mp4_path)

        if not args.keep_raw and mpg_path.exists():
            mpg_path.unlink()

        # 2. Download and parse ground truth
        download_file(info['gt_url'], xml_path)
        parse_cvml_xml(
            xml_path,
            gt_dir / f"{cam_id}_gt.json",
            gt_dir / f"{cam_id}_gt_mot.txt",
        )

    if not args.keep_raw and raw_dir.exists():
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    print(f"\nCAVIAR ({args.sequence}) dataset ready!")
    print(f"  Videos:       {videos_dir.resolve()}")
    print(f"  Ground truth: {gt_dir.resolve()}\n")


if __name__ == '__main__':
    main()

