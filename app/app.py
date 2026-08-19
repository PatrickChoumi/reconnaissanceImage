"""Jour 4 - Interface utilisateur (Streamlit).

Lancer avec: streamlit run app/app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import load_trained_model, predict  # noqa: E402

st.set_page_config(page_title="Reconnaissance d'image", page_icon="🔍")
st.title("🔍 Reconnaissance d'image")
st.write("Depose une image (chat, chien, voiture, avion...) et le modele devine ce que c'est.")


@st.cache_resource
def get_model():
    return load_trained_model()


uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image envoyee", use_container_width=True)

    try:
        model = get_model()
    except (OSError, ValueError):
        st.error("Aucun modele entraine trouve. Lance d'abord: python src/train.py")
    else:
        label, confidence, probabilities = predict(model, image)
        st.subheader(f"Prediction : **{label}** ({confidence:.1%} de confiance)")

        from data import CLASS_NAMES
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probabilities)})
