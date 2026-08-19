"""Jour 3/4 - Prediction sur une image quelconque (pas seulement CIFAR-10)."""

import numpy as np
from PIL import Image
from tensorflow import keras

from data import CLASS_NAMES
from train import MODEL_PATH


def load_trained_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Le modele attend exactement des images 32x32 RGB normalisees [0, 1],
    # comme celles utilisees a l'entrainement.
    image = image.convert("RGB").resize((32, 32))
    array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)  # ajoute la dimension "batch"


def predict(model, image: Image.Image):
    x = preprocess_image(image)
    probabilities = model.predict(x, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    return CLASS_NAMES[top_index], float(probabilities[top_index]), probabilities


if __name__ == "__main__":
    import sys

    model = load_trained_model()
    image = Image.open(sys.argv[1])
    label, confidence, _ = predict(model, image)
    print(f"Prediction: {label} ({confidence:.1%})")
