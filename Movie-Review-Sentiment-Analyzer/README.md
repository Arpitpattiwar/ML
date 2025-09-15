# 🎬 Movie Sentiment Analyzer

A simple **end-to-end NLP project** that predicts if a movie review is **positive** or **negative** using TF-IDF + Logistic Regression.  
Deployed with **Streamlit** for an interactive UI.

## 🚀 Features
- Preprocesses raw text (cleaning, stopword removal).
- Vectorizes using **TF-IDF**.
- Trains a baseline **Logistic Regression model**.
- Interactive web app with Streamlit.

## 📊 Dataset
Using the [IMDb Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).  
Place `imdb_reviews.csv` in the `data/` folder.

## ⚙️ Setup
```bash
git clone https://github.com/ArpitPattiwar/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer
pip install -r requirements.txt
