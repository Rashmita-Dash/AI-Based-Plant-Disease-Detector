
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random

st.set_page_config(page_title="AI Plant Disease Detector", layout="wide")
st.title("🌱 AI-Based Plant Disease Detector")
st.write("Upload a photo of your plant or leaf to identify diseases and get treatment guidance.")

MODEL_PATH = "models/plant_disease_model.h5"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ AI Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        model = None
else:
    st.warning("⚠️ Model file not found. Running in demo mode with simulated results.")

CLASS_NAMES = [
    "Healthy Leaf",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Potato - Leaf Spot",
    "Apple - Scab",
    "Corn - Common Rust",
]

TREATMENTS = {
    "Healthy Leaf": {
        "status": "✅ Your plant looks healthy!",
        "chemical": "No treatment needed.",
        "organic": "Maintain regular watering and sunlight exposure."
    },
    "Tomato - Early Blight": {
        "status": "⚠️ Early Blight Detected.",
        "chemical": "Use Mancozeb or Chlorothalonil spray.",
        "organic": "Apply neem oil weekly and remove infected leaves."
    },
    "Tomato - Late Blight": {
        "status": "⚠️ Late Blight Detected.",
        "chemical": "Use copper-based fungicides.",
        "organic": "Use baking soda spray and improve air circulation."
    },
    "Potato - Leaf Spot": {
        "status": "⚠️ Leaf Spot Found.",
        "chemical": "Apply Azoxystrobin fungicide.",
        "organic": "Use compost tea and avoid overhead watering."
    },
    "Apple - Scab": {
        "status": "⚠️ Apple Scab Identified.",
        "chemical": "Use Captan or Sulfur-based fungicide.",
        "organic": "Prune infected leaves and use lime-sulfur spray."
    },
    "Corn - Common Rust": {
        "status": "⚠️ Common Rust Detected.",
        "chemical": "Use Propiconazole-based fungicide.",
        "organic": "Rotate crops and ensure good field sanitation."
    }
}

uploaded_file = st.file_uploader("📤 Upload Plant or Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze with AI"):
        with st.spinner("Analyzing image..."):
            # Prepare image for model
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            if model:
                # Predict using AI model
                predictions = model.predict(img_array)
                pred_class = CLASS_NAMES[np.argmax(predictions[0])]
            else:
                # Simulated prediction (demo mode)
                pred_class = random.choice(CLASS_NAMES)

            result = TREATMENTS.get(pred_class, {})
            st.success(f"🪴 Prediction: **{pred_class}**")
            st.info(result.get("status", "Analysis complete."))

            st.subheader("💊 Treatment Suggestions")
            st.write(f"**Chemical:** {result.get('chemical', 'N/A')}")
            st.write(f"**Organic:** {result.get('organic', 'N/A')}")

st.markdown("---")
st.subheader("🤖 Chatbot Assistance")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask about plant care, watering, or prevention tips...")

if user_input:
    response = ""
    q = user_input.lower()

    if "water" in q:
        response = "💧 Water your plants early in the morning. Avoid overwatering!"
    elif "fertilizer" in q:
        response = "🌿 Use compost or organic fertilizer once every two weeks."
    elif "prevent" in q:
        response = "🛡️ Ensure proper air circulation and avoid leaf wetness."
    elif "sunlight" in q:
        response = "☀️ Most plants need at least 6 hours of sunlight daily."
    elif "thank" in q:
        response = "😊 You're welcome! Happy gardening!"
    else:
        response = "🤔 I'm still learning about that. Try asking about watering, fertilizers, or diseases."

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

st.markdown("---")
st.markdown("🌿 **THANK YOU **")
