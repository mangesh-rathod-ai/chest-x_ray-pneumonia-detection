"""Evaluate the trained model on the untouched test set.

    python -m src.evaluate
    python -m src.evaluate --threshold 0.35

Reports metrics at the default 0.5 threshold AND at the recall-tuned threshold that was
chosen on the validation set during training. Outputs go to ``outputs/``:
metrics.json, classification_report.txt, confusion_matrix.png, roc_curve.png,
test_predictions.csv and error_analysis.png (false negatives / false positives).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config, data
from .predict import load_model_and_config
from .train import collect_predictions
from .utils import save_json


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _plot_confusion_matrices(y_true, y_prob, thresholds: dict, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.2 * len(thresholds), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, thr) in zip(axes, thresholds.items()):
        cm = confusion_matrix(y_true, (y_prob >= thr).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], config.CLASS_NAMES)
        ax.set_yticks([0, 1], config.CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} (threshold = {thr:.3f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_roc(y_true, y_prob, thresholds: dict, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    for name, thr in thresholds.items():
        pred = (y_prob >= thr).astype(int)
        t_fpr = ((pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
        t_tpr = ((pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        ax.scatter([t_fpr], [t_tpr], s=60, zorder=3, label=f"{name} ({thr:.2f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall / sensitivity)")
    ax.set_title("ROC curve - test set")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_errors(df: pd.DataFrame, out_path, per_group: int = 8):
    """Grid of the most confident mistakes: false negatives (missed pneumonia) on top, false positives below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fn = df[(df.y_true == 1) & (df.y_pred == 0)].sort_values("prob").head(per_group)
    fp = df[(df.y_true == 0) & (df.y_pred == 1)].sort_values("prob", ascending=False).head(per_group)
    rows = [("False negatives (pneumonia missed)", fn), ("False positives (normal flagged)", fp)]
    fig, axes = plt.subplots(2, per_group, figsize=(2.3 * per_group, 6))
    for r, (title, group) in enumerate(rows):
        for c in range(per_group):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(group):
                row = group.iloc[c]
                ax.imshow(Image.open(row.path).convert("L"), cmap="gray")
                ax.set_title(f"p={row.prob:.2f}", fontsize=9)
        axes[r, 0].text(0, -0.12, title, transform=axes[r, 0].transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the model on the test split.")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--model-path", default=str(config.MODEL_PATH))
    p.add_argument("--config-path", default=str(config.CONFIG_PATH))
    p.add_argument("--output-dir", default=str(config.OUTPUTS_DIR))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the tuned threshold stored in model_config.json")
    args = p.parse_args(argv)

    model, cfg = load_model_and_config(args.model_path, args.config_path)
    tuned = args.threshold if args.threshold is not None else cfg["threshold"]
    test_ds, file_paths = data.load_test(args.data_dir, cfg["img_size"], args.batch_size)
    y_true, y_prob = collect_predictions(model, test_ds)
    y_true = y_true.astype(int)

    thresholds = {"default": 0.5}
    if abs(tuned - 0.5) > 1e-6:
        thresholds["recall-tuned"] = tuned
    results = {name: compute_metrics(y_true, y_prob, thr) for name, thr in thresholds.items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / "metrics.json")

    report_lines = []
    for name, thr in thresholds.items():
        report = classification_report(
            y_true, (y_prob >= thr).astype(int), target_names=config.CLASS_NAMES, digits=4, zero_division=0
        )
        report_lines.append(f"=== {name} threshold = {thr:.3f} ===\n{report}")
    (out_dir / "classification_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    _plot_confusion_matrices(y_true, y_prob, thresholds, out_dir / "confusion_matrix.png")
    if len(np.unique(y_true)) > 1:
        _plot_roc(y_true, y_prob, thresholds, out_dir / "roc_curve.png")

    if file_paths and len(file_paths) == len(y_true):
        df = pd.DataFrame({
            "path": file_paths, "y_true": y_true, "prob": y_prob,
            "y_pred": (y_prob >= tuned).astype(int),
        })
        df.to_csv(out_dir / "test_predictions.csv", index=False)
        _plot_errors(df, out_dir / "error_analysis.png")

    for name, m in results.items():
        print(f"\n--- {name} (threshold {m['threshold']:.3f}) ---")
        print("\n".join(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}"
                        for k, v in m.items() if k != "threshold"))
    print(f"\nSaved reports and figures to {out_dir}/")
    return results


if __name__ == "__main__":
    main()
