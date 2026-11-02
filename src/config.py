"""Central place for paths and default hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where the Kaggle "Chest X-Ray Images (Pneumonia)" dataset is expected
DATA_DIR = ROOT / "data" / "raw" / "chest_xray"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "medical_model.keras"
CONFIG_PATH = MODELS_DIR / "model_config.json"

# Class order matters: label 0 = NORMAL, label 1 = PNEUMONIA.
# The network has ONE sigmoid output = P(PNEUMONIA).
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
POSITIVE_CLASS = "PNEUMONIA"

BACKBONES = ("efficientnetb0", "mobilenetv2", "baseline_cnn")

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15          # fraction of train/ held out for validation
MIN_RECALL = 0.95         # target sensitivity used to pick the decision threshold

IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}
