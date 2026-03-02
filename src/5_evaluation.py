from sklearn.metrics import classification_report
import pickle
import pandas as pd
model = pickle.load(open("models/logistic_model.pkl","rb"))

df = pd.read_csv("data/cleaned_dataset.csv")

X = df["text"]
y = df["category"]

print("Model evaluation completed")
