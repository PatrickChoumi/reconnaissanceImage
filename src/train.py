"""Jour 2/3 - Entrainement du modele.

Usage:
    python src/train.py --epochs 15
"""

import argparse
import os

from tensorflow import keras

from data import load_data
from model import build_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.keras")


def main(epochs: int):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    # EarlyStopping arrete l'entrainement si le modele arrete de progresser
    # sur les donnees de validation, pour eviter de perdre du temps / overfitter.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nPrecision finale sur le jeu de test: {test_acc:.2%}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args.epochs)
