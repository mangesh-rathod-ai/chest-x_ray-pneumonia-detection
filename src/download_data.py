"""Download the Kaggle "Chest X-Ray Images (Pneumonia)" dataset into data/raw/chest_xray/.

    pip install kagglehub
    python -m src.download_data

If this fails (no internet / Kaggle login required), download the zip manually from
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
and unzip it so that these folders exist:

    data/raw/chest_xray/train/{NORMAL,PNEUMONIA}
    data/raw/chest_xray/test/{NORMAL,PNEUMONIA}
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import config
from .utils import count_images_per_class, find_dataset_root

KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"


def main():
    dest = config.DATA_DIR
    if (dest / "train").is_dir() and (dest / "test").is_dir():
        print(f"Dataset already present at {dest}")
    else:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit("Install kagglehub first:  pip install kagglehub")

        print(f"Downloading {KAGGLE_SLUG} (about 1.2 GB) ...")
        downloaded = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        root = find_dataset_root(downloaded)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            root, dest, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"), dirs_exist_ok=True
        )
        print(f"Copied dataset to {dest}")

    for split in ("train", "val", "test"):
        if (dest / split).is_dir():
            print(f"  {split:<5}: {count_images_per_class(dest / split)}")


if __name__ == "__main__":
    main()
