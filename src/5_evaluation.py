import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/test.csv")

X = df[["text"]]
y = df["category"]

# Load trained model
model = joblib.load("models/logistic_model.pkl")

# Predict
predictions = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, predictions)
print("Model Accuracy:", accuracy)

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y, predictions))
