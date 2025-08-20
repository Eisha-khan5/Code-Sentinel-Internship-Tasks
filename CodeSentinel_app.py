# ==== Suppress TensorFlow Logs ====
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==== Imports ====
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ==== Streamlit App ====
st.set_page_config(page_title="TensorFlow + Streamlit App", layout="centered")

st.title("🍊 VGG16 Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.image(uploaded_file, caption="Uploaded Image", use_container_width=False, width=300)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Convert image to numpy array
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.write("✅ Image preprocessed and ready for model!")
# Load your trained VGG16 model
model = tf.keras.models.load_model("./models/model.keras")

# Make predictions
preds = model.predict(img_array)[0]  # get first row since batch=1

# Load labels.json
import json
with open("labels.json", "r") as f:
    labels = json.load(f)

# Get top 3 predictions
top_indices = preds.argsort()[-3:][::-1]  # sort and take top 3
st.subheader("🔍 Predictions")
for i, idx in enumerate(top_indices):
    label = labels[str(idx)] if str(idx) in labels else f"Class {idx}"
    score = preds[idx]
    st.write(f"**{i+1}. {label}** ({score:.2%})")
