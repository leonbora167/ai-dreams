"""Download the compact EPFL 4-person laboratory sequence.

The source is the official EPFL CVLab page. The videos are AVI files, so this
helper converts them to the MP4 files expected by the tracker. It only runs
when the user explicitly invokes it; importing this module has no side effects.
"""
import argparse
from pathlib import Path
import requests
from tqdm import tqdm


URLS = {
    'cam01': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c0.avi',
    'cam02': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c1.avi',
    'cam03': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c2.avi',
    'cam04': 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c3.avi',
}


def download(url, target):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with target.open('wb') as out, tqdm(total=total, unit='B', unit_scale=True, desc=target.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk); bar.update(len(chunk))


def convert(source, target):
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV cannot decode {source}; install/use ffmpeg and convert it to MP4 manually.')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened(): raise RuntimeError(f'Cannot create {target}')
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        writer.write(frame); count += 1
    cap.release(); writer.release()
    if count == 0: raise RuntimeError(f'No frames decoded from {source}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/videos')
    parser.add_argument('--keep-avi', action='store_true')
    args = parser.parse_args()
    target = Path(args.output); target.mkdir(parents=True, exist_ok=True)
    for camera, url in URLS.items():
        avi = target / f'{camera}.avi'; mp4 = target / f'{camera}.mp4'
        if not mp4.exists():
            if not avi.exists(): download(url, avi)
            print(f'Converting {avi.name} -> {mp4.name}')
            convert(avi, mp4)
        if avi.exists() and not args.keep_avi: avi.unlink()
    print(f'Dataset ready: {target.resolve()}')


if __name__ == '__main__':
    main()
