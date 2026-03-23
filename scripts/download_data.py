"""
Download datasets for RecSys Experimental Tutorials.

Datasets:
1. Criteo Display Advertising Challenge (~11GB compressed)
   - Source: Kaggle (criteo-display-advertising-challenge)
2. Taobao User Behavior (~1GB compressed)
   - Source: Kaggle (marwa80/userbehavior)
3. Ali-CCP (Alibaba Click and Conversion Prediction)
   - Source: Tianchi (dataset/408)

Usage:
    python scripts/download_data.py [--dataset {criteo,taobao,aliccp,all}]
"""

import os
import sys
import argparse
import subprocess
import zipfile
import gzip
import shutil
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def check_kaggle_api():
    """Check if kaggle API is available."""
    try:
        import kaggle
        return True
    except Exception:
        print("Error: Kaggle API not configured.")
        print("1. pip install kaggle")
        print("2. Go to https://www.kaggle.com/settings -> Create New Token")
        print("3. Place kaggle.json in ~/.kaggle/")
        return False


def download_criteo():
    """Download Criteo Display Advertising Challenge dataset."""
    dest = DATA_DIR / "criteo"
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "train.txt").exists() or (dest / "train.csv").exists():
        print("[Criteo] Dataset already exists, skipping download.")
        return

    print("[Criteo] Downloading from Kaggle...")
    print("[Criteo] Note: This dataset is ~11GB compressed, ~40GB uncompressed.")

    try:
        subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "criteo-display-advertising-challenge",
             "-p", str(dest)],
            check=True
        )

        # Extract
        for f in dest.glob("*.zip"):
            print(f"[Criteo] Extracting {f.name}...")
            with zipfile.ZipFile(f, 'r') as zf:
                zf.extractall(dest)
            f.unlink()

        print("[Criteo] Download complete.")

    except subprocess.CalledProcessError:
        print("[Criteo] Kaggle download failed.")
        print("[Criteo] Alternative: Download manually from")
        print("  https://www.kaggle.com/c/criteo-display-advertising-challenge/data")
        print(f"  and extract to {dest}")


def download_taobao():
    """Download Taobao User Behavior dataset."""
    dest = DATA_DIR / "taobao"
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "UserBehavior.csv").exists():
        print("[Taobao] Dataset already exists, skipping download.")
        return

    print("[Taobao] Downloading from Kaggle...")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "marwa80/userbehavior",
             "-p", str(dest)],
            check=True
        )

        # Extract
        for f in dest.glob("*.zip"):
            print(f"[Taobao] Extracting {f.name}...")
            with zipfile.ZipFile(f, 'r') as zf:
                zf.extractall(dest)
            f.unlink()

        # Handle .gz file if present
        for f in dest.glob("*.csv.gz"):
            print(f"[Taobao] Decompressing {f.name}...")
            with gzip.open(f, 'rb') as gz_in:
                with open(f.with_suffix('').with_suffix('.csv'), 'wb') as out:
                    shutil.copyfileobj(gz_in, out)
            f.unlink()

        print("[Taobao] Download complete.")

    except subprocess.CalledProcessError:
        print("[Taobao] Kaggle download failed.")
        print("[Taobao] Alternative: Download manually from")
        print("  https://www.kaggle.com/datasets/marwa80/userbehavior")
        print(f"  and extract to {dest}")


def download_aliccp():
    """Download Ali-CCP dataset."""
    dest = DATA_DIR / "aliccp"
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.glob("*.csv")) or any(dest.glob("*.txt")):
        print("[Ali-CCP] Dataset already exists, skipping download.")
        return

    print("[Ali-CCP] Downloading from Tianchi...")
    print("[Ali-CCP] Note: You may need to register at tianchi.aliyun.com")

    try:
        # Try kaggle first (some mirrors exist)
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "aliwisdom/alibaba-click-and-conversion-prediction",
             "-p", str(dest)],
            check=True
        )

        for f in dest.glob("*.zip"):
            print(f"[Ali-CCP] Extracting {f.name}...")
            with zipfile.ZipFile(f, 'r') as zf:
                zf.extractall(dest)
            f.unlink()

        print("[Ali-CCP] Download complete.")

    except subprocess.CalledProcessError:
        print("[Ali-CCP] Kaggle download failed.")
        print("[Ali-CCP] Please download manually from:")
        print("  https://tianchi.aliyun.com/dataset/408")
        print(f"  and extract to {dest}")
        print("")
        print("[Ali-CCP] Alternative Kaggle sources:")
        print("  https://www.kaggle.com/datasets/aliwisdom/alibaba-click-and-conversion-prediction")


def main():
    parser = argparse.ArgumentParser(description="Download RecSys datasets")
    parser.add_argument(
        "--dataset",
        choices=["criteo", "taobao", "aliccp", "all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()

    if not check_kaggle_api():
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("criteo", "all"):
        download_criteo()
    if args.dataset in ("taobao", "all"):
        download_taobao()
    if args.dataset in ("aliccp", "all"):
        download_aliccp()

    print("\nDone! Data directory structure:")
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
