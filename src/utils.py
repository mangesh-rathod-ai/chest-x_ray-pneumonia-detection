"""Small helpers that do not need TensorFlow (so scripts like download_data stay fast)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config


def find_dataset_root(path) -> Path:
    """Return the folder that directly contains ``train/`` and ``test/``.

    The Kaggle zip is often nested (chest_xray/chest_xray/train/...) and may contain a
    __MACOSX folder, so we look a few levels down instead of trusting one fixed path.
    """
    path = Path(path)
    for candidate in (path, path / "chest_xray"):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for train_dir in sorted(path.rglob("train")):
        if (
            train_dir.is_dir()
            and (train_dir.parent / "test").is_dir()
            and "__MACOSX" not in train_dir.parts
        ):
            return train_dir.parent
    raise FileNotFoundError(
        f"Could not find a dataset with train/ and test/ folders under: {path}\n"
        "Run `python -m src.download_data` or see the README for manual download steps."
    )


def count_images_per_class(split_dir) -> dict[str, int]:
    """Count images in <split_dir>/<CLASS>/ for every class."""
    counts = {}
    for cls in config.CLASS_NAMES:
        folder = Path(split_dir) / cls
        counts[cls] = (
            sum(1 for f in folder.iterdir() if f.suffix.lower() in config.IMG_EXTS)
            if folder.is_dir()
            else 0
        )
    return counts


def compute_class_weights(split_dir) -> dict[int, float]:
    """Balanced class weights {0: w_normal, 1: w_pneumonia} from folder counts."""
    counts = count_images_per_class(split_dir)
    n = [counts[c] for c in config.CLASS_NAMES]
    total = sum(n)
    if min(n) == 0:
        raise ValueError(f"Every class needs at least one training image, got: {counts}")
    return {i: total / (len(n) * n[i]) for i in range(len(n))}


def select_threshold(y_true, y_prob, min_recall: float = config.MIN_RECALL) -> float:
    """Pick the decision threshold with the lowest false-positive rate whose recall >= min_recall.

    For a screening-style task, missing a sick patient (false negative) is worse than
    flagging a healthy one, so we fix a recall target on the *validation* set and
    then minimise false alarms. Falls back to 0.5 if the target cannot be reached.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= min_recall)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmin(fpr[ok])]
    return float(np.clip(thresholds[best], 0.01, 0.99))


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_history(histories: list[dict], out_path, finetune_start: int | None = None) -> None:
    """Plot loss / accuracy / AUC / recall over all training phases and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged: dict[str, list] = {}
    for h in histories:
        for key, values in h.items():
            merged.setdefault(key, []).extend(values)

    panels = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "ROC-AUC"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    for ax, (key, title) in zip(axes, panels):
        if key in merged:
            ax.plot(merged[key], label="train")
        if f"val_{key}" in merged:
            ax.plot(merged[f"val_{key}"], label="validation")
        if finetune_start is not None:
            ax.axvline(finetune_start - 0.5, color="gray", linestyle="--", label="fine-tuning starts")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
