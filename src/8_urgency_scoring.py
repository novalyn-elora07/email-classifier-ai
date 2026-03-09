import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.urgency_keywords import keyword_urgency

model = joblib.load("models/urgency_model.pkl")
vectorizer = joblib.load("models/urgency_vectorizer.pkl")


def predict_urgency(text):

    keyword_result = keyword_urgency(text)

    X = vectorizer.transform([text])
    ml_result = model.predict(X)[0]

    if keyword_result == "high":
        return "high"

    return ml_result
