# Medical Image Disease Detection: Chest X-ray Pneumonia

A deep-learning system that classifies chest X-rays as **NORMAL** or **PNEUMONIA**, reports a
confidence score, and explains each prediction with a **Grad-CAM** heatmap. It uses transfer
learning (EfficientNetB0 / MobileNetV2), evaluates with medical-appropriate metrics (recall,
F1, ROC-AUC), and ships as a **Streamlit** web app.

> **Disclaimer:** an educational / portfolio project. It is not a medical device and must not be
> used for diagnosis or treatment decisions.

```
X-ray -> preprocessing -> augmentation -> CNN / transfer learning -> P(pneumonia)
      -> recall-tuned threshold -> Grad-CAM heatmap -> Streamlit app
```

## Project structure

```
Medical-Disease-Detection/
├── app/app.py                 Streamlit web app
├── data/
│   ├── raw/chest_xray/        dataset goes here (not committed)
│   └── processed/             reserved for derived data
├── models/                    medical_model.keras + model_config.json (created by training)
├── notebooks/                 01 EDA, 02 preprocessing, 03 baseline CNN, 04 transfer learning, 05 evaluation
├── outputs/                   curves, confusion matrix, ROC, error analysis (created by scripts)
├── src/
│   ├── config.py              paths and defaults
│   ├── data.py                tf.data pipeline + augmentation
│   ├── models.py              baseline CNN, EfficientNetB0, MobileNetV2
│   ├── train.py               two-phase training, class weights, threshold tuning
│   ├── evaluate.py            test-set metrics, plots, error analysis
│   ├── predict.py             single-image inference (CLI + used by the app)
│   ├── gradcam.py             Grad-CAM heatmap + overlay
│   ├── download_data.py       dataset download helper
│   ├── make_dummy_data.py     tiny synthetic dataset for smoke tests
│   └── utils.py
├── tests/test_pipeline.py     end-to-end smoke tests
├── requirements.txt
└── README.md
```

## Quick start

Python 3.10 to 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Check the install (no dataset needed, about 1 minute)

```bash
python -m src.make_dummy_data --out data/dummy
python -m src.train --data-dir data/dummy --weights none --epochs 1 --finetune-epochs 1 --img-size 96
python -m src.evaluate --data-dir data/dummy
pytest -q
```

The dummy images are synthetic, so ignore these numbers. This only proves every stage runs.
Delete `models/medical_model.keras`, `models/model_config.json` and `outputs/` afterwards so they
don't get mixed up with your real model.

### 2. Get the real dataset

Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(about 5,860 images).

```bash
python -m src.download_data
```

If that fails, download the zip from Kaggle manually and unzip it so these folders exist:

```
data/raw/chest_xray/train/{NORMAL,PNEUMONIA}
data/raw/chest_xray/test/{NORMAL,PNEUMONIA}
```

(The nested `chest_xray/chest_xray/` layout from the zip is detected automatically.)

### 3. Train

```bash
python -m src.train                           # EfficientNetB0, 10 + 10 epochs
python -m src.train --backbone mobilenetv2    # lighter and faster
python -m src.train --backbone baseline_cnn --epochs 15   # CNN from scratch, for comparison
```

The first run with `--weights imagenet` (default) downloads the pretrained weights, so it needs internet.
**Use a GPU if you can**: on a CPU each epoch takes several minutes. Free option: Google Colab
(Runtime -> Change runtime type -> GPU), then:

```python
!git clone <your-repo-url> && %cd Medical-Disease-Detection
!pip install -q kagglehub opencv-python-headless
!python -m src.download_data
!python -m src.train
```

Training does the following:

1. Holds out 15% of `train/` as validation (the official `val/` folder has only 16 images).
2. **Phase 1**: trains a new head on the frozen ImageNet backbone, with class weights for the imbalance.
3. **Phase 2**: fine-tunes the top 30 backbone layers (BatchNorm frozen) with a tiny learning rate.
4. Uses early stopping on validation AUC and learning-rate reduction on plateau.
5. Chooses the decision threshold on the **validation** set so recall reaches at least 95% with the fewest false alarms.
6. Saves `models/medical_model.keras`, `models/model_config.json` and `outputs/training_history.png`.

### 4. Evaluate on the test set

```bash
python -m src.evaluate
```

Writes to `outputs/`: `metrics.json`, `classification_report.txt`, `confusion_matrix.png`,
`roc_curve.png`, `test_predictions.csv`, `error_analysis.png`. Metrics are reported at threshold 0.5
and at the recall-tuned threshold.

### 5. Predict one image

```bash
python -m src.predict --image path/to/xray.jpeg --gradcam
```

### 6. Run the web app

```bash
streamlit run app/app.py
```

Upload a JPG or PNG X-ray to see the prediction, pneumonia probability, and Grad-CAM heatmap.
The sidebar lets you move the decision threshold and the heatmap opacity.

## Deploy to Streamlit Community Cloud

1. Train locally so `models/medical_model.keras` and `models/model_config.json` exist. The model is about 18 MB, so commit it (`.gitignore` already allows these two files).
2. Push the repo to GitHub (do **not** commit `data/raw`).
3. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo with main file path `app/app.py`.
4. Under Advanced settings choose Python 3.11 or 3.12.

## Results

Fill this in after training. Don't copy numbers from other projects.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN (threshold 0.5) | | | | | |
| EfficientNetB0 (threshold 0.5) | | | | | |
| EfficientNetB0 (recall-tuned) | | | | | |

## Design decisions

- **Recall over accuracy.** Missing pneumonia is worse than a false alarm, so the threshold is tuned on validation data for a recall target.
- **No horizontal flips.** A flipped chest X-ray puts the heart on the right, which is rare in real data.
- **No threshold tuning on test data.** The test set is touched once, for the final report.
- **One preprocessing path.** Each model contains its own rescaling layer, so training, evaluation and the app see identical inputs.
- **Grad-CAM for every model.** All models share the `preprocess -> backbone -> head` structure.

## Known limitations (worth stating in interviews)

- **Possible patient leakage.** The Kaggle dataset has several images per patient with no patient IDs, so a random validation split can put the same patient on both sides. Validation scores may be optimistic.
- **Test-set shift.** The official test split is known to behave differently from train, with more false positives on NORMAL images than validation suggests. Use `error_analysis.png` to investigate.
- **Single source.** Pediatric images from one hospital. The model has not been validated on other scanners, ages, or populations.
- **Bacterial vs. viral** pneumonia are both labeled PNEUMONIA.
- **Heatmaps are not explanations of disease.** Check that the heat sits on the lungs and not on text markers or borders.

## Resume bullet

**Medical Image Disease Detection** | TensorFlow/Keras, EfficientNet, Grad-CAM, Streamlit
Built a chest X-ray pneumonia classifier using transfer learning and augmentation with class-weighted
training; tuned the decision threshold on validation data for a recall target and evaluated with
precision, recall, F1 and ROC-AUC. Added Grad-CAM explainability and error analysis, and deployed an
interactive Streamlit app. (Add your measured test recall / AUC here.)

## Next steps

Multi-class diseases (COVID, TB), EfficientNet vs MobileNet comparison table, hyperparameter search
(Keras Tuner / Optuna), a FastAPI endpoint, and a Dockerfile.
