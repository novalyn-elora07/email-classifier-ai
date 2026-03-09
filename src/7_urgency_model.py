import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/urgency_dataset.csv")

X = df["text"]
y = df["urgency"]

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/urgency_model.pkl")
joblib.dump(vectorizer, "models/urgency_vectorizer.pkl")

print("Urgency model trained and saved!")