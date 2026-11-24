"""
Streamlit prototype: upload an oral image, get a prediction + Grad-CAM heatmap.

Run with:
    streamlit run app/app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402

st.set_page_config(page_title="Oral Disease Detection", layout="centered")
st.title("Oral Disease Detection using AI")
st.write(
    "Upload an intraoral photo or X-ray. The model classifies it as "
    "healthy, cavity, gum disease, or oral lesion, and shows a Grad-CAM "
    "heatmap explaining which region drove the prediction."
)

MODEL_PATH = "models/final_model.keras"  # produced by src/train.py
CLASS_NAMES = ["healthy", "cavity", "gum_disease", "oral_lesion"]
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    st.image(img, caption="Uploaded image", width=300)

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found at models/final_model.keras. "
            "Run `python src/train.py` on your dataset first."
        )
    else:
        model = load_model()
        preds = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        st.subheader(f"Prediction: {CLASS_NAMES[pred_idx]}")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, preds)})

        heatmap, _ = make_gradcam_heatmap(np.expand_dims(img_array, 0).astype("float32"), model)
        overlay = overlay_heatmap((img_array * 255).astype(np.uint8), heatmap)
        st.image(overlay, caption="Grad-CAM: regions influencing the prediction", width=300)
