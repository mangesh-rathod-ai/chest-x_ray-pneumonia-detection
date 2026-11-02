from pathlib import Path
import keras

model_path = Path("models/medical_model.keras")

if model_path.exists():
    print("Model file exists!")

    model = keras.models.load_model(
        model_path,
        compile=False,
    )

    print("Model loaded successfully!")
    model.summary()
else:
    print("Model file not found!")