import sys
import os
import streamlit as st
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text

# Load model + vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #ff4b4b;
            text-align: center;
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown("<div class='main-title'>🎬 Movie Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enter a movie review and see if it's Positive or Negative</div>", unsafe_allow_html=True)

# --- Input ---
review = st.text_area("✍️ Write your review below:", height=150)

# --- Prediction ---
if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == "positive":
            st.success("✅ Positive Review 🎉\nThe model thinks this review is **positive**.")
        else:
            st.error("❌ Negative Review 😔\nThe model thinks this review is **negative**.")
    else:
        st.warning("⚠️ Please type a review before analyzing.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses **TF-IDF + Logistic Regression**  
    to classify IMDb reviews as **Positive** or **Negative**.  
    
    🔧 Built with [Streamlit](https://streamlit.io)  
    📊 Dataset: IMDb Reviews  
    """
)
