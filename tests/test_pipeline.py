"""End-to-end smoke tests on a tiny synthetic dataset (no downloads, no GPU, no ImageNet weights).

Run:  pytest -q
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, evaluate, models, predict, train  # noqa: E402
from src.gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402
from src.make_dummy_data import make_dummy_dataset  # noqa: E402
from src.utils import (  # noqa: E402
    compute_class_weights,
    find_dataset_root,
    select_threshold,
)

IMG = 96  # small images keep the tests fast


@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory):
    return make_dummy_dataset(tmp_path_factory.mktemp("dummy"), n_train=24, n_test=10)


def test_find_dataset_root_handles_nested_layout(dummy_root, tmp_path):
    nested = tmp_path / "chest_xray" / "chest_xray"
    shutil.copytree(dummy_root, nested)
    assert find_dataset_root(tmp_path) == nested
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path / "does_not_exist")


def test_class_weights_favour_minority_class(dummy_root):
    w = compute_class_weights(dummy_root / "train")
    assert w[0] > w[1]  # NORMAL is the minority class in the dummy data


def test_select_threshold_reaches_recall_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    p = np.r_[rng.beta(2, 5, 200), rng.beta(5, 2, 200)]
    thr = select_threshold(y, p, min_recall=0.95)
    recall = ((p >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert recall >= 0.95
    assert select_threshold(np.zeros(10), np.random.rand(10)) == 0.5  # single class fallback


@pytest.mark.parametrize("backbone", config.BACKBONES)
def test_model_forward_and_gradcam(backbone):
    model = models.build_model(backbone, IMG, weights=None)
    batch = np.random.uniform(0, 255, (2, IMG, IMG, 3)).astype("float32")
    out = model(batch, training=False).numpy()
    assert out.shape == (2, 1) and ((out >= 0) & (out <= 1)).all()

    heatmap, prob = make_gradcam_heatmap(model, batch[:1])
    assert heatmap.ndim == 2 and 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0.0 <= prob <= 1.0
    overlay, heat_rgb = overlay_heatmap(batch[0].astype("uint8"), heatmap)
    assert overlay.shape == (IMG, IMG, 3) and heat_rgb.shape == (IMG, IMG, 3)


def test_full_pipeline_train_evaluate_predict(dummy_root, tmp_path):
    model_path = tmp_path / "model.keras"
    cfg_path = tmp_path / "model_config.json"
    out_dir = tmp_path / "outputs"
    common = ["--data-dir", str(dummy_root), "--img-size", str(IMG), "--batch-size", "8"]

    train.main(
        common
        + ["--backbone", "mobilenetv2", "--weights", "none", "--epochs", "1",
           "--finetune-epochs", "1", "--finetune-layers", "10",
           "--model-path", str(model_path), "--config-path", str(cfg_path),
           "--output-dir", str(out_dir)]
    )
    assert model_path.exists() and cfg_path.exists()
    assert (out_dir / "training_history.png").exists()
    cfg = json.loads(cfg_path.read_text())
    assert 0.01 <= cfg["threshold"] <= 0.99 and cfg["img_size"] == IMG

    results = evaluate.main(
        ["--data-dir", str(dummy_root), "--batch-size", "8", "--model-path", str(model_path),
         "--config-path", str(cfg_path), "--output-dir", str(out_dir)]
    )
    assert "default" in results and 0.0 <= results["default"]["accuracy"] <= 1.0
    for name in ("metrics.json", "confusion_matrix.png", "roc_curve.png",
                 "test_predictions.csv", "error_analysis.png"):
        assert (out_dir / name).exists(), name

    model, loaded_cfg = predict.load_model_and_config(model_path, cfg_path)
    sample = next((dummy_root / "test" / "PNEUMONIA").glob("*.jpeg"))
    result = predict.predict_image(
        model, sample, img_size=loaded_cfg["img_size"], threshold=loaded_cfg["threshold"],
        with_gradcam=True,
    )
    assert result["label"] in config.CLASS_NAMES
    assert 0.0 <= result["probability"] <= 1.0
    assert result["overlay"].shape == (IMG, IMG, 3)
