import joblib
import pandas as pd
from transformers import pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text
from src.urgency_model import HybridUrgencyModel

class EmailPredictor:
    def __init__(self, use_advanced=False):
        self.use_advanced = use_advanced
        self.urgency_model = HybridUrgencyModel()
        
        try:
            self.urgency_model.load("models/urgency_hybrid.pkl")
        except FileNotFoundError:
            print("Urgency ML model not found. Using Rule-based only.")
            
        if self.use_advanced:
            try:
                self.hf_pipeline = pipeline("text-classification", model="models/distilbert", tokenizer="models/distilbert")
            except Exception as e:
                print(f"Failed to load DistilBERT: {e}. Falling back to baseline.")
                self.use_advanced = False
                
        if not self.use_advanced:
            try:
                self.lr_model = joblib.load("models/logistic_regression.pkl")
                self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
            except Exception as e:
                print(f"Warning: Baseline models not generated yet! Run train_model.py first. Error: {e}")
                self.lr_model = None

    def predict(self, raw_text: str):
        cleaned = clean_text(raw_text)
        urgency = self.urgency_model.predict(cleaned, mode="hybrid")
        
        category = "Unknown"
        confidence = 0.0
        
        if self.use_advanced:
            res = self.hf_pipeline(raw_text, truncation=True, max_length=128)[0]
            category = str(res['label'])
            confidence = res['score']
        elif self.lr_model:
            X = self.vectorizer.transform([cleaned])
            pred_class = self.lr_model.predict(X)[0]
            probs = self.lr_model.predict_proba(X)[0]
            confidence = float(max(probs))
            category = str(pred_class)

        return {
            "category": category,
            "category_confidence": confidence,
            "urgency": urgency
        }

if __name__ == "__main__":
    predictor = EmailPredictor(use_advanced=False)
    test_emails = [
        "The server is completely down and not working, I need help immediately!",
        "Thanks for getting back to me so quickly.",
        "I'd like to request a new laptop for my work.",
        "Buy cheap viagra and rolex watches here!"
    ]
    for email in test_emails:
        print(f"Input: {email}")
        print(f"Result: {predictor.predict(email)}\n")
