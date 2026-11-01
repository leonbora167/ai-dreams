import os
import urllib.request
import ssl
import sys
from pathlib import Path
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl._create_unverified_context()

SAMPLE_URLS = [
    # Open Intel IoT sample video of pedestrians indoors
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
    # Additional open-source pedestrian / tracking samples
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4",
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/classroom.mp4"
]

DEFAULT_OUTPUT = Path("data/raw_videos/sample.mp4")

def download_sample(target_path: Path = DEFAULT_OUTPUT) -> Path:
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 10000:
        print(f"Sample video already exists at: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    success = False
    for url in SAMPLE_URLS:
        print(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response, open(target_path, "wb") as out_file:
                total_length = response.getheader("content-length")
                total_bytes = int(total_length) if total_length else None
                downloaded = 0
                block_size = 64 * 1024

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_bytes:
                        done = int(50 * downloaded / total_bytes)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024*1024):.2f} MB")
                        sys.stdout.flush()
                print()
            
            if target_path.stat().st_size > 10000:
                print(f"Successfully downloaded to {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
                success = True
                break
            else:
                print("Downloaded file was too small, trying fallback...")
                if target_path.exists():
                    target_path.unlink()
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass

    if not success:
        raise RuntimeError(f"Could not download sample video from any source to {target_path}")

    return target_path

if __name__ == "__main__":
    download_sample()
