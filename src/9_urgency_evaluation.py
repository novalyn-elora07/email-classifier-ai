import pandas as pd
import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/processed/urgency_dataset.csv")

X = df["text"]
y = df["urgency"]

model = joblib.load("models/urgency_model.pkl")
vectorizer = joblib.load("models/urgency_vectorizer.pkl")

X_vectorized = vectorizer.transform(X)

predictions = model.predict(X_vectorized)

print("Confusion Matrix")
print(confusion_matrix(y, predictions))

print("\nClassification Report")
print(classification_report(y, predictions))