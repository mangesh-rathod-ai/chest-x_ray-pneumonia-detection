from pathlib import Path

import keras
import numpy as np
from PIL import Image

# Paths
MODEL_PATH = Path("models/medical_model.keras")
IMAGE_PATH = Path("test_image.jpg")

# Load the saved model without compiling it.
model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Saved model loaded successfully!")

# Check that the test image exists.
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}. "
        "Place a test chest X-ray image in the project root."
    )

# Load and resize the image.
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

# Convert the image into a NumPy array.
image_array = np.asarray(
    image,
    dtype=np.float32,
)

# Add the batch dimension.
image_array = np.expand_dims(
    image_array,
    axis=0,
)

# Generate prediction.
prediction = model.predict(
    image_array,
    verbose=0,
)

probability = float(prediction.reshape(-1)[0])

print(f"Model output: {probability:.4f}")
print("Prediction completed successfully!")
