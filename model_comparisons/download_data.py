"""
Download AeroPath (Zenodo zip) + two KaggleHub datasets into separate folders.

Requirements:
  pip install kagglehub

Notes:
- KaggleHub requires Kaggle credentials (e.g., %USERPROFILE%\.kaggle\kaggle.json on Windows).
- AeroPath is downloaded from Zenodo and extracted into its folder.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import kagglehub


def copytree_into(src: Path, dst: Path) -> None:
    """Copy *contents* of src directory into dst directory."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def download_kagglehub(dataset_slug: str, target_dir: Path) -> Path:
    """Download KaggleHub dataset to cache, then copy into target_dir."""
    cache_path = Path(kagglehub.dataset_download(dataset_slug))
    copytree_into(cache_path, target_dir)
    return cache_path


def main() -> None:
    base_dir = Path("data")

    aeropath_dir = base_dir / "AeroPath"
    bctv_dir = base_dir / "BCTV_abdomen"  # lssz1275/abdomen
    msd_dir = base_dir / "MSD_lung"       # vivekprajapati2048/medical-segmentation-decathlon-lung

    # Ensure folders exist
    aeropath_dir.mkdir(parents=True, exist_ok=True)
    bctv_dir.mkdir(parents=True, exist_ok=True)
    msd_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Downloading BCTV (lssz1275/abdomen) -> {bctv_dir}")
    bctv_cache = download_kagglehub("lssz1275/abdomen", bctv_dir)
    print(f"      Done. (KaggleHub cache: {bctv_cache})")

    print(f"[3/3] Downloading MSD Lung (vivekprajapati2048/medical-segmentation-decathlon-lung) -> {msd_dir}")
    msd_cache = download_kagglehub("vivekprajapati2048/medical-segmentation-decathlon-lung", msd_dir)
    print(f"      Done. (KaggleHub cache: {msd_cache})")

    print("\nAll datasets downloaded into:")
    print(f"  {base_dir.resolve()}")
    print("\nFolder layout:")
    print(f"  - {aeropath_dir.name}/")
    print(f"  - {bctv_dir.name}/")
    print(f"  - {msd_dir.name}/")


if __name__ == "__main__":
    main()
