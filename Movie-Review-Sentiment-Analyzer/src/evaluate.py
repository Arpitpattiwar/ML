import joblib
from preprocess import clean_text

def predict_review(review: str):
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    return prediction

if __name__ == "__main__":
    sample_review = "The movie was bad, but enjoyed it!"
    result = predict_review(sample_review)
    print(f"Review: {sample_review}")
    print(f"Predicted Sentiment: {result}")
