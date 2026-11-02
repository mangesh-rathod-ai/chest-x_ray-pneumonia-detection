from __future__ import annotations

import keras
from keras import layers

from . import config


PRETRAINED = ("efficientnetb0", "mobilenetv2")


def _preprocess_layer(backbone_name: str) -> layers.Layer:
    backbone_name = backbone_name.lower()

    if backbone_name == "mobilenetv2":
        return layers.Rescaling(
            1.0 / 127.5,
            offset=-1.0,
            name="preprocess",
        )

    if backbone_name == "baseline_cnn":
        return layers.Rescaling(
            1.0 / 255.0,
            name="preprocess",
        )

    return layers.Rescaling(
        1.0,
        name="preprocess",
    )


def _baseline_cnn_backbone(input_shape) -> keras.Model:
    inputs = keras.Input(
        shape=input_shape,
        name="backbone_input",
    )

    x = inputs

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            activation="relu",
        )(x)

        x = layers.MaxPooling2D()(x)

    return keras.Model(
        inputs,
        x,
        name="backbone",
    )


def _pretrained_backbone(
    name: str,
    input_shape,
    weights,
) -> keras.Model:

    if name == "efficientnetb0":
        return keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            name="backbone",
        )

    if name == "mobilenetv2":
        return keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            name="backbone",
        )

    raise ValueError(
        f"Unsupported backbone: {name}"
    )


def _head(
    feature_shape,
    dropout: float,
) -> keras.Sequential:

    return keras.Sequential(
        [
            keras.Input(
                shape=feature_shape
            ),

            layers.GlobalAveragePooling2D(),

            layers.BatchNormalization(),

            layers.Dropout(dropout),

            layers.Dense(
                128,
                activation="relu",
            ),

            layers.Dropout(dropout),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="head",
    )


def build_model(
    backbone_name: str = "efficientnetb0",
    img_size: int = config.IMG_SIZE,
    dropout: float = 0.3,
    weights: str | None = None,
    freeze_backbone: bool = False,
) -> keras.Model:

    backbone_name = backbone_name.lower()

    if backbone_name not in config.BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from {config.BACKBONES}."
        )

    if isinstance(weights, str):
        if weights.lower() == "none":
            weights = None

    input_shape = (
        img_size,
        img_size,
        3,
    )

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = _preprocess_layer(
        backbone_name
    )(inputs)

    if backbone_name == "baseline_cnn":

        backbone = _baseline_cnn_backbone(
            input_shape
        )

        backbone.trainable = True

    else:

        backbone = _pretrained_backbone(
            backbone_name,
            input_shape,
            weights,
        )

        backbone.trainable = not freeze_backbone

    x = backbone(x)

    feature_shape = backbone.output_shape[1:]

    head = _head(
        feature_shape,
        dropout,
    )

    outputs = head(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"pneumonia_{backbone_name}",
    )


def unfreeze_top_layers(
    model: keras.Model,
    n_layers: int = 30,
) -> int:

    if n_layers < 1:
        raise ValueError(
            "n_layers must be at least 1."
        )

    backbone = model.get_layer(
        "backbone"
    )

    backbone.trainable = True

    n_layers = min(
        n_layers,
        len(backbone.layers),
    )

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-n_layers:]:
        layer.trainable = True

    for layer in backbone.layers:
        if isinstance(
            layer,
            layers.BatchNormalization,
        ):
            layer.trainable = False

    return sum(
        1
        for layer in backbone.layers
        if layer.trainable
    )