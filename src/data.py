"""Jour 1 - Chargement et preparation des donnees.

Dataset: CIFAR-10 (integre a Keras, pas besoin de le telecharger a la main).
60 000 images couleur 32x32, 10 classes: avion, voiture, oiseau, chat, cerf,
chien, grenouille, cheval, bateau, camion.
"""

import tensorflow as tf

CLASS_NAMES = [
    "avion", "voiture", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Les pixels vont de 0 a 255. On les ramene entre 0 et 1: le reseau
    # converge plus vite et plus stablement sur des petites valeurs.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # y_train/y_test sont des indices de classe (ex: 3 = "chat"). On les
    # aplatit car Keras les fournit en colonnes (N, 1).
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Exemple de label: {y_train[0]} -> {CLASS_NAMES[y_train[0]]}")
