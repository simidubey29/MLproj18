import streamlit as st
import pickle
import os
from utils import get_spam_probability

# -------------------------------
# Load Model (Render Safe)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "model1.pkl")
    vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# UI Design
# -------------------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>📩 SMS Spam Detector</h1>
    <p style='text-align: center;'>AI-powered spam detection system</p>
    """,
    unsafe_allow_html=True
)

# Input
message = st.text_area("✉️ Enter your message")

# Button
if st.button("🚀 Predict"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Transform
        vect_msg = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(vect_msg)[0]
        prob = get_spam_probability(model, vect_msg)

        # Result
        if prediction == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Not Spam (Safe Message)")

        # -------------------------------
        # Probability Bar 📊
        # -------------------------------
        st.markdown("### 📊 Spam Probability")
        st.progress(int(prob * 100))

        st.write(f"🔎 Confidence: **{prob*100:.2f}% Spam**")