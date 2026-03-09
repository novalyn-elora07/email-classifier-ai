import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test dataset
df = pd.read_csv("data/test.csv")

# Features and labels
X = df[["text"]]       # email text
y = df[["category"]]       # actual category

# Load trained model
model = joblib.load("models/baseline_model.pkl")

# Make predictions
predictions = model.predict(X)

# Evaluate model
accuracy = accuracy_score(y, predictions)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y, predictions))
