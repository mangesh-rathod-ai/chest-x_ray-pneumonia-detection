"""Grad-CAM: which regions of the X-ray pushed the model towards its decision?

Works on any model built by ``models.build_model`` because it relies on the named
parts ``preprocess`` -> ``backbone`` -> ``head``.
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(model, img_batch, target_class: int | None = None):
    """Compute a Grad-CAM heatmap.

    Args:
        model: model from ``build_model`` (or loaded from a .keras file).
        img_batch: float array of shape (1, H, W, 3) with raw 0-255 pixel values.
        target_class: 1 = explain "PNEUMONIA", 0 = explain "NORMAL".
            Default: the class the model predicts (at threshold 0.5).

    Returns:
        (heatmap, probability) - heatmap is a 2-D float array in [0, 1] at feature-map
        resolution; probability is P(PNEUMONIA).
    """
    preprocess = model.get_layer("preprocess")
    backbone = model.get_layer("backbone")
    head = model.get_layer("head")

    img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps = backbone(preprocess(img_batch), training=False)
        tape.watch(feature_maps)  # needed when the backbone is frozen
        prob = head(feature_maps, training=False)[:, 0]
        if target_class is None:
            target_class = int(float(prob[0]) >= 0.5)
        score = prob if target_class == 1 else 1.0 - prob

    grads = tape.gradient(score, feature_maps)
    channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))            # one weight per channel
    cam = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)   # weighted sum of maps
    cam = tf.nn.relu(cam).numpy()
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    return cam, float(prob[0])


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """Blend a heatmap onto an RGB uint8 image.

    Returns (overlay_rgb, heatmap_rgb), both uint8 with the same size as ``image_rgb``.
    """
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(heatmap.astype("float32"), (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    colored_bgr = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = alpha * heatmap_rgb.astype("float32") + (1.0 - alpha) * image_rgb.astype("float32")
    return np.clip(blended, 0, 255).astype(np.uint8), heatmap_rgb
