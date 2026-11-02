"""Create a tiny SYNTHETIC dataset with the same folder layout as the real one.

The images are fake (noise + blobs) - they are only for checking that the whole
pipeline (train -> evaluate -> predict -> Grad-CAM -> app) runs end to end on your
machine before you download the 1.2 GB real dataset. Never report results from it.

    python -m src.make_dummy_data --out data/dummy
    python -m src.train --data-dir data/dummy --weights none --epochs 1 --finetune-epochs 1 --img-size 96
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _make_image(rng: np.random.Generator, pneumonia: bool, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype("float32")
    lungs = np.exp(-(((xx - size / 2) / (size * 0.35)) ** 2 + ((yy - size / 2) / (size * 0.45)) ** 2))
    img = 50 + 60 * lungs + rng.normal(0, 8, (size, size))
    if pneumonia:  # brighter cloudy patches inside the "lungs"
        for _ in range(int(rng.integers(2, 5))):
            cx, cy = rng.integers(int(size * 0.25), int(size * 0.75), size=2)
            r = int(rng.integers(size // 12, size // 6))
            img += 80 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)))
    return np.clip(img, 0, 255).astype("uint8")


def make_dummy_dataset(out_dir, n_train: int = 60, n_test: int = 24, seed: int = 0) -> Path:
    """Write train/val/test folders. Pneumonia is the majority class in train (like the real data)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    plan = {
        "train": {"NORMAL": n_train, "PNEUMONIA": int(n_train * 1.5)},
        "val": {"NORMAL": 4, "PNEUMONIA": 4},
        "test": {"NORMAL": n_test, "PNEUMONIA": n_test},
    }
    for split, classes in plan.items():
        for cls, n in classes.items():
            folder = out_dir / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = _make_image(rng, pneumonia=(cls == "PNEUMONIA"))
                Image.fromarray(img).save(folder / f"{cls.lower()}_{i:04d}.jpeg")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a tiny synthetic dataset for smoke tests.")
    p.add_argument("--out", default="data/dummy")
    p.add_argument("--n-train", type=int, default=60, help="NORMAL images in train (PNEUMONIA = 1.5x)")
    p.add_argument("--n-test", type=int, default=24, help="images per class in test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = make_dummy_dataset(args.out, args.n_train, args.n_test, args.seed)
    print(f"Synthetic dataset written to {out}")


if __name__ == "__main__":
    main()
