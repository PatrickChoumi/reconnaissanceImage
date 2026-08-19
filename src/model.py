"""Jour 2 - Architecture du CNN (reseau de neurones convolutif).

Un CNN apprend a reconnaitre des motifs visuels (bords, textures, formes)
par couches successives: les premieres couches detectent des details simples,
les suivantes combinent ces details en concepts plus abstraits (roue, oreille,
aile...) jusqu'a la classification finale.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Chaque bloc Conv2D + MaxPooling extrait des motifs puis reduit
        # la taille de l'image pour ne garder que l'information utile.
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # On "aplatit" la grille de motifs 2D en un vecteur, pour pouvoir
        # la passer a des couches classiques (Dense) qui font la decision.
        layers.Flatten(),
        layers.Dropout(0.4),  # eteint 40% des neurones a chaque pas d'entrainement, contre l'overfitting
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),  # softmax = probabilite pour chacune des 10 classes
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # adaptee aux labels sous forme d'entiers (0-9)
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
