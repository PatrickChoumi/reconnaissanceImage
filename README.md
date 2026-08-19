# Reconnaissance d'image (CNN)

Projet d'apprentissage : reconnaitre des objets dans une image (chat, chien,
voiture, avion...) avec un reseau de neurones convolutif (CNN), entraine avec
TensorFlow/Keras, servi via une petite interface Streamlit.

## Stack choisie (et pourquoi)

- **TensorFlow / Keras** plutot que PyTorch : API plus haut niveau, moins de
  code "boilerplate", ideal pour un premier CNN. Une fois les concepts
  maitrises (Jour 1-3), le meme modele peut etre reecrit en PyTorch pour
  comparer les deux, si le temps le permet.
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

> Note : cet environnement d'execution n'a pas acces a `cs.toronto.edu` (le
> serveur qui heberge CIFAR-10), donc l'entrainement doit se faire sur ta
> machine en local, ou` l'acces internet est libre. Le code lui-meme a ete
> verifie ici (build du modele, boucle d'entrainement testee avec des donnees
> factices, pretraitement d'image) : il n'y a plus qu'a lancer `train.py`
> chez toi pour obtenir un vrai modele.

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
- Optionnel : transfer learning avec MobileNetV2 pre-entraine (bien meilleure
  precision, peu de code en plus) si le temps le permet.
- `src/predict.py` : tester le modele sur des photos perso (pas juste CIFAR-10).

**Jour 4 - Interface** (`app/app.py`)
- Streamlit : upload d'image, affichage de la prediction + graphique de
  confiance par classe.

**Jour 5 - Deploiement local & polish**
- Verifier les cas limites (mauvaise image, mauvais format).
- `streamlit run app/app.py` accessible sur le reseau local.
- Bilan : ce qu'on a appris, pistes pour aller plus loin (plus de classes,
  webcam en direct, export mobile...).

## Comment on va travailler

Chaque jour correspond a un fichier ci-dessus. Le code est deja pose avec des
commentaires qui expliquent le "pourquoi" de chaque choix (normalisation,
dropout, softmax...) pour servir de support d'apprentissage. Dis-moi a quel
jour tu veux commencer et on approfondit ensemble : je peux t'expliquer un
concept en detail, modifier l'architecture, deboguer une erreur d'entrainement,
ou ajouter une fonctionnalite (webcam, plus de classes, transfer learning...).
