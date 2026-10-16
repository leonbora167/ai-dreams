"""Standalone Dataset Ingestion and Sampling for Anomaly Evaluation.

Downloads authentic video samples from:
  1. UCF-Crime (minhajuddinmeraj/anomalydetectiondatasetucf)
  2. XD-Violence (bypktt/xd-violence)

Downloads ~100 videos per dataset across the 7 target classes:
  - Normal
  - Violence
  - Fighting
  - Arson
  - Car Accident
  - Explosion
  - Shooting

Cleans raw cache to respect edge storage constraints, verifies each video with OpenCV,
and compiles eval/data/manifest.csv.
"""

import os
import sys
import shutil
import cv2
import pandas as pd
from tqdm import tqdm
import kagglehub
import kagglesdk
from kagglesdk.datasets.services.dataset_api_service import ApiListDatasetFilesRequest

# Target Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UCF_OUT_DIR = os.path.join(DATA_DIR, "ucf_crime")
XD_OUT_DIR = os.path.join(DATA_DIR, "xd_violence")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

TARGET_CLASSES = [
    "Normal",
    "Violence",
    "Fighting",
    "Arson",
    "Car Accident",
    "Explosion",
    "Shooting"
]


def discover_all_mp4s(owner_slug: str, dataset_slug: str, max_pages: int = 70):
    """List all .mp4 files in a Kaggle dataset via kagglesdk."""
    client = kagglesdk.KaggleClient()
    page_token = None
    all_files = []
    
    print(f"[*] Discovering video files in {owner_slug}/{dataset_slug}...")
    for _ in range(max_pages):
        req = ApiListDatasetFilesRequest()
        req.owner_slug = owner_slug
        req.dataset_slug = dataset_slug
        req.page_size = 100
        if page_token:
            req.page_token = page_token
        resp = client.datasets.dataset_api_client.list_dataset_files(req)
        for f in resp.dataset_files:
            if f.name.endswith(".mp4"):
                all_files.append((f.name, getattr(f, "total_bytes", 0)))
        page_token = resp.next_page_token
        if not page_token:
            break
            
    print(f"[+] Discovered {len(all_files)} .mp4 files in {owner_slug}/{dataset_slug}.")
    return all_files


def select_ucf_samples(all_ucf_files, max_per_class=15):
    """Categorize and select ~100 genuine UCF-Crime videos."""
    class_candidates = {cls: [] for cls in TARGET_CLASSES}
    
    for fpath, size in all_ucf_files:
        # Filter out corrupt/zero size or overly massive (>60MB) for efficient eval
        if size < 500_000 or size > 60_000_000:
            continue
            
        fpath_lower = fpath.lower()
        if "normal_videos_for_event_recognition" in fpath_lower:
            class_candidates["Normal"].append((fpath, size))
        elif "arson" in fpath_lower:
            class_candidates["Arson"].append((fpath, size))
        elif "fighting" in fpath_lower:
            class_candidates["Fighting"].append((fpath, size))
        elif "explosion" in fpath_lower:
            class_candidates["Explosion"].append((fpath, size))
        elif "roadaccidents" in fpath_lower:
            class_candidates["Car Accident"].append((fpath, size))
        elif "shooting" in fpath_lower:
            class_candidates["Shooting"].append((fpath, size))
        elif "assault" in fpath_lower or "abuse" in fpath_lower:
            class_candidates["Violence"].append((fpath, size))

    selected = []
    # Allocation targets to reach ~100
    targets = {
        "Arson": 20,
        "Fighting": 15,
        "Explosion": 15,
        "Shooting": 15,
        "Car Accident": 15,
        "Violence": 10,
        "Normal": 10,
    }
    for cls, count in targets.items():
        chosen = class_candidates[cls][:count]
        for fpath, sz in chosen:
            selected.append({"dataset": "ucf_crime", "class": cls, "path": fpath, "size": sz})
        print(f"  UCF [{cls}]: Selected {len(chosen)} / {len(class_candidates[cls])} candidates")
    
    return selected


def select_xd_samples(all_xd_files):
    """Categorize and select ~100 genuine XD-Violence videos."""
    class_candidates = {cls: [] for cls in TARGET_CLASSES}
    
    for fpath, size in all_xd_files:
        if size < 500_000 or size > 60_000_000:
            continue
            
        parts = fpath.split("/")
        if len(parts) >= 2:
            cat = parts[1].lower() if parts[0] in ["test", "train"] else parts[0].lower()
            if "normal" in cat:
                class_candidates["Normal"].append((fpath, size))
            elif "caraccident" in cat:
                class_candidates["Car Accident"].append((fpath, size))
            elif "explosion" in cat:
                class_candidates["Explosion"].append((fpath, size))
            elif "fighting" in cat:
                class_candidates["Fighting"].append((fpath, size))
            elif "shooting" in cat:
                class_candidates["Shooting"].append((fpath, size))
            elif "riot" in cat or "abuse" in cat:
                class_candidates["Violence"].append((fpath, size))

    selected = []
    targets = {
        "Car Accident": 20,
        "Explosion": 20,
        "Fighting": 20,
        "Shooting": 20,
        "Violence": 10,
        "Normal": 10,
    }
    for cls, count in targets.items():
        chosen = class_candidates[cls][:count]
        for fpath, sz in chosen:
            selected.append({"dataset": "xd_violence", "class": cls, "path": fpath, "size": sz})
        print(f"  XD [{cls}]: Selected {len(chosen)} / {len(class_candidates[cls])} candidates")
    
    return selected


def download_and_ingest(selected_list, dataset_name, handle, target_dir):
    """Download each chosen video, validate with cv2, move to destination, and clean cache."""
    os.makedirs(target_dir, exist_ok=True)
    ingested_records = []
    
    print(f"\n=======================================================")
    print(f"Starting Download & Ingestion: {dataset_name} ({len(selected_list)} videos)")
    print(f"Kaggle Handle: {handle}")
    print(f"Target Directory: {target_dir}")
    print(f"=======================================================\n")
    
    pbar = tqdm(selected_list, desc=f"Downloading {dataset_name}", unit="video")
    for item in pbar:
        rel_path = item["path"]
        cls = item["class"]
        pbar.set_postfix({"class": cls, "file": os.path.basename(rel_path)[:25]})
        
        cls_dir = os.path.join(target_dir, cls.replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)
        dest_file = os.path.join(cls_dir, os.path.basename(rel_path))
        
        # If already downloaded and verified, keep it
        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
            cached_path = dest_file
        else:
            try:
                cached_path = kagglehub.dataset_download(handle, path=rel_path)
                shutil.copy2(cached_path, dest_file)
                # Purge cache file to free up cache directory
                try:
                    if os.path.exists(cached_path) and os.path.abspath(cached_path) != os.path.abspath(dest_file):
                        os.remove(cached_path)
                except Exception:
                    pass
            except Exception as e:
                tqdm.write(f"[!] Error downloading {rel_path}: {e}")
                continue

        # Validate video using OpenCV
        cap = cv2.VideoCapture(dest_file)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        cap.release()
        
        if not ret or frame_count < 10:
            tqdm.write(f"[!] Warning: {dest_file} appears invalid or empty. Removing.")
            if os.path.exists(dest_file):
                os.remove(dest_file)
            continue
            
        duration_sec = frame_count / fps if fps > 0 else 0
        
        video_id = f"{dataset_name}_{cls.replace(' ', '_')}_{os.path.basename(dest_file)}"
        rel_local = os.path.relpath(dest_file, BASE_DIR).replace("\\", "/")
        
        ingested_records.append({
            "video_id": video_id,
            "dataset": dataset_name,
            "ground_truth": cls,
            "original_path": rel_path,
            "local_path": rel_local,
            "full_path": os.path.abspath(dest_file),
            "file_size_bytes": os.path.getsize(dest_file),
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 2),
            "resolution": f"{w}x{h}"
        })
        
    return ingested_records


def main():
    print("=" * 70)
    print("  AUTHENTIC DATASET INGESTION: UCF-CRIME & XD-VIOLENCE")
    print("=" * 70)
    
    # 1. Discover and sample UCF-Crime
    ucf_files = discover_all_mp4s("minhajuddinmeraj", "anomalydetectiondatasetucf", max_pages=50)
    ucf_selected = select_ucf_samples(ucf_files)
    
    # Ingest UCF-Crime (download 100 videos, then clean raw cache)
    ucf_records = download_and_ingest(
        ucf_selected,
        dataset_name="ucf_crime",
        handle="minhajuddinmeraj/anomalydetectiondatasetucf",
        target_dir=UCF_OUT_DIR
    )
    print(f"\n[+] Successfully ingested {len(ucf_records)} authentic UCF-Crime videos.\n")
    
    # 2. Discover and sample XD-Violence
    xd_files = discover_all_mp4s("bypktt", "xd-violence", max_pages=30)
    xd_selected = select_xd_samples(xd_files)
    
    # Ingest XD-Violence (download 100 videos, then clean raw cache)
    xd_records = download_and_ingest(
        xd_selected,
        dataset_name="xd_violence",
        handle="bypktt/xd-violence",
        target_dir=XD_OUT_DIR
    )
    print(f"\n[+] Successfully ingested {len(xd_records)} authentic XD-Violence videos.\n")
    
    # 3. Clean any remaining kagglehub archives
    cache_root = os.path.expanduser("~/.cache/kagglehub/datasets")
    print(f"[*] Pruning temporary download caches in {cache_root}...")
    for sub in ["minhajuddinmeraj/anomalydetectiondatasetucf", "bypktt/xd-violence"]:
        target_cache = os.path.join(cache_root, os.path.normpath(sub))
        if os.path.exists(target_cache):
            try:
                shutil.rmtree(target_cache)
                print(f"[+] Cleared cache: {target_cache}")
            except Exception as e:
                print(f"[-] Could not clear {target_cache}: {e}")
                
    # 4. Compile Manifest CSV
    all_records = ucf_records + xd_records
    df_manifest = pd.DataFrame(all_records)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[+] Compiled verified manifest to: {MANIFEST_PATH}")
    print(f"    Total videos ingested: {len(df_manifest)}")
    print("\nBreakdown by Dataset and Class:")
    print(df_manifest.groupby(["dataset", "ground_truth"]).size())
    print("\nDataset ingestion complete and verified!\n")


if __name__ == "__main__":
    main()

