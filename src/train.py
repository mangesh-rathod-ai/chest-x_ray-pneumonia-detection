
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import keras
import numpy as np

from . import config, data, models
from .utils import (
    compute_class_weights,
    count_images_per_class,
    plot_history,
    save_json,
    select_threshold,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a chest X-ray pneumonia classifier."
    )

    p.add_argument(
        "--data-dir",
        default=str(config.DATA_DIR),
    )

    p.add_argument(
        "--backbone",
        default="efficientnetb0",
        choices=config.BACKBONES,
    )

    p.add_argument(
        "--weights",
        default="none",
        choices=["imagenet", "none"],
        help='Pretrained weights: "imagenet" or "none"',
    )

    p.add_argument(
        "--img-size",
        type=int,
        default=config.IMG_SIZE,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
    )

    # CHANGED: Default training epochs reduced from 10 to 5.
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Phase 1 epochs (default: 5)",
    )

    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="Phase 2 epochs (0 = skip)",
    )

    p.add_argument(
        "--finetune-layers",
        type=int,
        default=30,
        help="Backbone layers to unfreeze",
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience",
    )

    p.add_argument(
        "--min-recall",
        type=float,
        default=config.MIN_RECALL,
        help="Minimum validation recall target",
    )

    p.add_argument(
        "--no-class-weights",
        action="store_true",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
    )

    p.add_argument(
        "--model-path",
        default=str(config.MODEL_PATH),
    )

    p.add_argument(
        "--config-path",
        default=str(config.CONFIG_PATH),
    )

    p.add_argument(
        "--output-dir",
        default=str(config.OUTPUTS_DIR),
    )

    return p.parse_args(argv)


def make_metrics():
    return [
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
        keras.metrics.AUC(
            name="auc"
        ),
    ]


def make_callbacks(patience: int):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def collect_predictions(model, dataset):
    y_true = []
    y_prob = []

    for images, labels in dataset:
        predictions = model(
            images,
            training=False,
        ).numpy().ravel()

        y_prob.append(predictions)
        y_true.append(
            labels.numpy().ravel()
        )

    return (
        np.concatenate(y_true),
        np.concatenate(y_prob),
    )


def main(argv=None):
    args = parse_args(argv)

    # Validate training arguments.
    if args.epochs < 1:
        raise ValueError(
            "--epochs must be at least 1."
        )

    if args.finetune_epochs < 0:
        raise ValueError(
            "--finetune-epochs cannot be negative."
        )

    if args.finetune_layers < 1:
        raise ValueError(
            "--finetune-layers must be at least 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    if not 0.0 <= args.min_recall <= 1.0:
        raise ValueError(
            "--min-recall must be between 0 and 1."
        )

    keras.utils.set_random_seed(
        args.seed
    )

    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA CLASSIFIER")
    print("=" * 60)

    print(
        f"Backbone       : {args.backbone}"
    )

    print(
        f"Training epochs: {args.epochs}"
    )

    print(
        f"Fine-tune epochs: {args.finetune_epochs}"
    )

    print(
        f"Image size     : {args.img_size}"
    )

    print(
        f"Batch size     : {args.batch_size}"
    )

    print(
        f"Initial weights: {args.weights}"
    )

    print("\nLoading data ...")

    train_ds, val_ds, root = data.load_train_val(
        args.data_dir,
        args.img_size,
        args.batch_size,
        config.VAL_SPLIT,
        args.seed,
    )

    train_counts = count_images_per_class(
        root / "train"
    )

    class_weight = (
        None
        if args.no_class_weights
        else compute_class_weights(
            root / "train"
        )
    )

    print(
        f"Train folder counts: {train_counts}"
    )

    print(
        f"Class weights: {class_weight}"
    )

    pretrained = (
        args.backbone in models.PRETRAINED
        and args.weights != "none"
    )

    model = models.build_model(
        backbone_name=args.backbone,
        img_size=args.img_size,
        dropout=args.dropout,
        weights=args.weights,
        freeze_backbone=pretrained,
    )

    model.summary(
        line_length=100
    )

    histories = []
    finetune_start = None

    # Compile the model for Phase 1.
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.lr
        ),
        loss="binary_crossentropy",
        metrics=make_metrics(),
    )

    print(
        "\n=== Phase 1: Training ==="
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=make_callbacks(
            args.patience
        ),
    )

    histories.append(
        history.history
    )

    # Phase 2: Fine-tuning is optional.
    if (
        pretrained
        and args.finetune_epochs > 0
    ):
        n_trainable = models.unfreeze_top_layers(
            model,
            args.finetune_layers,
        )

        print(
            f"\n=== Phase 2: Fine-tuning "
            f"({n_trainable} layers) ==="
        )

        # Recompile after changing trainable layers.
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=args.finetune_lr
            ),
            loss="binary_crossentropy",
            metrics=make_metrics(),
        )

        finetune_start = len(
            history.history["loss"]
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(
                args.patience
            ),
        )

        histories.append(
            history2.history
        )

    # Select the classification threshold using validation data.
    print(
        "\nCollecting validation predictions ..."
    )

    y_val, p_val = collect_predictions(
        model,
        val_ds,
    )

    threshold = select_threshold(
        y_val,
        p_val,
        args.min_recall,
    )

    print(
        f"\nSelected threshold: {threshold:.3f}"
    )

    # Save the trained model.
    model_path = Path(
        args.model_path
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The model is saved for inference.
    # Compile it with metrics again if evaluation is needed later.
    model.compile(
        loss="binary_crossentropy"
    )

    model.save(
        model_path
    )

    # Save training outputs.
    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_history(
        histories,
        out_dir / "training_history.png",
        finetune_start,
    )

    # Save configuration and threshold.
    save_json(
        {
            "backbone": args.backbone,
            "weights": args.weights,
            "img_size": args.img_size,
            "class_names": config.CLASS_NAMES,
            "threshold": float(threshold),
            "min_recall_target": args.min_recall,
            "trained_at": dt.datetime.now().isoformat(
                timespec="seconds"
            ),
            "train_class_counts": train_counts,
            "args": vars(args),
        },
        args.config_path,
    )

    print(
        f"\nSaved model -> {model_path}"
    )

    print(
        f"Saved config -> {args.config_path}"
    )

    print(
        f"Saved curves -> "
        f"{out_dir / 'training_history.png'}"
    )

    print(
        "\nTraining completed successfully."
    )

    return model


if __name__ == "__main__":
    main()

