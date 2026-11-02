"""Single-image inference (used by the Streamlit app and from the command line).

Usage:
    python -m src.predict --image path/to/xray.jpeg --gradcam --out outputs/prediction.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from . import config
from .gradcam import make_gradcam_heatmap, overlay_heatmap
from .utils import load_json


def load_model_and_config(model_path=config.MODEL_PATH, config_path=config.CONFIG_PATH):
    """Load the trained model and its saved settings (image size, threshold, ...)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Train one first: python -m src.train"
        )
    model = keras.saving.load_model(model_path, compile=False)
    cfg = {"img_size": config.IMG_SIZE, "threshold": 0.5, "class_names": config.CLASS_NAMES}
    if Path(config_path).exists():
        cfg.update(load_json(config_path))
    return model, cfg


def preprocess_image(image, img_size: int = config.IMG_SIZE):
    """Open an image (path or PIL.Image) and prepare it exactly like the training pipeline.

    Returns (display_uint8 [H, W, 3], batch_float32 [1, H, W, 3] in 0-255).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize(rgb, (img_size, img_size), method="bilinear").numpy()
    display = np.clip(resized, 0, 255).astype(np.uint8)
    return display, resized[None].astype(np.float32)


def predict_image(
    model,
    image,
    img_size: int = config.IMG_SIZE,
    threshold: float = 0.5,
    with_gradcam: bool = False,
    alpha: float = 0.4,
) -> dict:
    """Predict one image.

    ``probability`` is P(PNEUMONIA) - the model's confidence score. ``label`` is
    PNEUMONIA when probability >= threshold, otherwise NORMAL.
    """
    display, batch = preprocess_image(image, img_size)
    probability = float(model(batch, training=False).numpy().ravel()[0])
    label_idx = int(probability >= threshold)
    result = {
        "label": config.CLASS_NAMES[label_idx],
        "probability": probability,
        "threshold": float(threshold),
        "image": display,
    }
    if with_gradcam:
        heatmap, _ = make_gradcam_heatmap(model, batch, target_class=label_idx)
        overlay, heatmap_rgb = overlay_heatmap(display, heatmap, alpha=alpha)
        result.update(heatmap=heatmap, heatmap_rgb=heatmap_rgb, overlay=overlay)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray image.")
    parser.add_argument("--image", required=True, help="Path to the X-ray image")
    parser.add_argument("--model-path", default=str(config.MODEL_PATH))
    parser.add_argument("--config-path", default=str(config.CONFIG_PATH))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: value saved during training)")
    parser.add_argument("--gradcam", action="store_true", help="Also create a Grad-CAM figure")
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "prediction.png"))
    args = parser.parse_args(argv)

    model, cfg = load_model_and_config(args.model_path, args.config_path)
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]
    result = predict_image(
        model, args.image, img_size=cfg["img_size"], threshold=threshold, with_gradcam=args.gradcam
    )

    print(f"Prediction            : {result['label']}")
    print(f"P(PNEUMONIA)          : {result['probability']:.4f}")
    print(f"Decision threshold    : {result['threshold']:.3f}")

    if args.gradcam:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        axes[0].imshow(result["image"]); axes[0].set_title("X-ray")
        axes[1].imshow(result["heatmap_rgb"]); axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(result["overlay"]); axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{result['label']}  (P(pneumonia) = {result['probability']:.1%})")
        fig.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        plt.close(fig)
        print(f"Saved figure          : {args.out}")


if __name__ == "__main__":
    main()
