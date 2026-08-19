# Reconnaissance d'image (CNN)

Projet d'apprentissage : reconnaitre des objets dans une image (chat, chien,
voiture, avion...) avec un reseau de neurones convolutif (CNN), entraine avec
TensorFlow/Keras, servi via une petite interface Streamlit.

## Stack choisie (et pourquoi)

- **TensorFlow / Keras** plutot que PyTorch : API plus haut niveau, moins de
  code "boilerplate", adaptee a un premier CNN.
- **CIFAR-10** comme dataset : 60 000 images 32x32 deja labellisees dans 10
  classes (avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval,
  bateau, camion), integre a Keras (`tf.keras.datasets.cifar10`). Pas besoin
  de collecter/labelliser des images soi-meme, ce qui serait irrealiste en 5
  jours.
- **Streamlit** pour l'interface : une UI web fonctionnelle en ~30 lignes de
  Python pur, pas besoin de HTML/JS.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
src/data.py     -> chargement + normalisation des donnees (Jour 1)
src/model.py    -> architecture du CNN (Jour 2)
src/train.py    -> boucle d'entrainement (Jour 2-3)
src/predict.py  -> prediction sur une image quelconque (Jour 3-4)
app/app.py      -> interface Streamlit (Jour 4)
models/         -> modele entraine (genere, pas versionne dans git)
```

## Utilisation

```bash
# 1. Entrainer le modele (telecharge CIFAR-10 automatiquement la 1ere fois)
python src/train.py --epochs 15

# 2. Tester en ligne de commande sur une image
python src/predict.py chemin/vers/mon_image.jpg

# 3. Lancer l'interface web
streamlit run app/app.py
```

Le premier lancement de `train.py` telecharge CIFAR-10 (~170 Mo) et prend
quelques minutes sur CPU pour une quinzaine d'epochs. Le modele entraine est
sauvegarde dans `models/cnn_model.keras` (non versionne, voir `.gitignore`).

## Feuille de route - 5 jours

**Jour 1 - Bases & donnees** (`src/data.py`)
- Comprendre : qu'est-ce qu'un tenseur, une image en RGB, un label ?
- Charger CIFAR-10, visualiser quelques images avec matplotlib.
- Notion cle : normalisation (pixels 0-255 -> 0-1).

**Jour 2 - Premier CNN** (`src/model.py`, `src/train.py`)
- Comprendre : convolution, filtre, pooling, overfitting, dropout.
- Construire et entrainer un premier modele, observer accuracy train vs test.

**Jour 3 - Ameliorer le modele**
- Data augmentation (rotation, flip, zoom) pour reduire l'overfitting.
- Optionnel : transfer learning avec MobileNetV2 pre-entraine, pour une
  meilleure precision sans beaucoup plus de code.
- `src/predict.py` : tester le modele sur des photos perso (pas juste CIFAR-10).

**Jour 4 - Interface** (`app/app.py`)
- Streamlit : upload d'image, affichage de la prediction + graphique de
  confiance par classe.

**Jour 5 - Deploiement local & polish**
- Verifier les cas limites (mauvaise image, mauvais format).
- `streamlit run app/app.py` accessible sur le reseau local.
- Pistes d'extension : plus de classes, webcam en direct, export mobile.

## Notes

Le code est commente pour expliquer le "pourquoi" de chaque choix
(normalisation, dropout, softmax...), chaque fichier correspondant a une
etape de la feuille de route ci-dessus.
