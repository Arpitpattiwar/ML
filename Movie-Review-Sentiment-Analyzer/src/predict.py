import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from preprocess import clean_text

def evaluate_model(data_path="data/imdb_reviews.csv"):
    # Load dataset
    df = pd.read_csv(data_path)
    df["review"] = df["review"].apply(clean_text)

    # Load model + vectorizer
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Transform
    X_vec = vectorizer.transform(df["review"])
    y_true = df["sentiment"]

    # Predict
    y_pred = model.predict(X_vec)

    # Metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label="positive"))
    print("Recall:", recall_score(y_true, y_pred, pos_label="positive"))
    print("F1 Score:", f1_score(y_true, y_pred, pos_label="positive"))

if __name__ == "__main__":
    evaluate_model()
