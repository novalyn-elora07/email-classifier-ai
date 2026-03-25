import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

URGENT_KEYWORDS = [r'\burgent\b', r'\basap\b', r'\bnot working\b', r'\bimmediately\b', r'\bemergency\b', r'\bcritical\b', r'\bblocked\b']
MEDIUM_KEYWORDS = [r'\bsoon\b', r'\bimportant\b', r'\bissue\b', r'\bhelp\b', r'\brequest\b', r'\bneed\b']

class HybridUrgencyModel:
    def __init__(self):
        self.ml_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.is_trained = False
        
    def rule_based_urgency(self, text: str) -> str:
        text = str(text).lower()
        if any(re.search(kw, text) for kw in URGENT_KEYWORDS):
            return "High"
        elif any(re.search(kw, text) for kw in MEDIUM_KEYWORDS):
            return "Medium"
        else:
            return "Low"
            
    def _create_synthetic_labels(self, texts: list) -> list:
        # Simulated training data from heuristic rule mapping
        return [self.rule_based_urgency(t) for t in texts]

    def train_ml(self, texts: list):
        print("Creating synthetic labels for urgency training...")
        labels = pd.Series(self._create_synthetic_labels(texts))
        
        # Only train if we have more than one class variance
        if len(labels.unique()) <= 1:
            print("Not enough variance to train ML model, falling back to rule-based only.")
            return
            
        print("Training ML Urgency model (LogisticRegression)...")
        X = self.vectorizer.fit_transform(texts)
        self.ml_model.fit(X, labels)
        self.is_trained = True
        
    def save(self, model_path: str = "models/urgency_hybrid.pkl"):
        print(f"Saving urgency model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            "model": self.ml_model,
            "vectorizer": self.vectorizer,
            "is_trained": self.is_trained
        }, model_path)
        
    def load(self, model_path: str = "models/urgency_hybrid.pkl"):
        print(f"Loading urgency model from {model_path}...")
        data = joblib.load(model_path)
        self.ml_model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.is_trained = data["is_trained"]
        
    def predict(self, text: str, mode: str = "hybrid") -> str:
        rule_pred = self.rule_based_urgency(text)
        
        if mode == "rule" or not self.is_trained:
            return rule_pred
            
        # ML prediction
        try:
            X = self.vectorizer.transform([str(text)])
            ml_pred = self.ml_model.predict(X)[0]
            ml_probs = self.ml_model.predict_proba(X)
            ml_confidence = max(ml_probs[0])
            
            if mode == "ml":
                return ml_pred
                
            # Hybrid logic: If rule says High, trust it. If ML says High with high confidence, trust it.
            if rule_pred == "High":
                return "High"
            if ml_pred == "High" and ml_confidence > 0.7:
                return "High"
            if rule_pred == "Medium":
                return "Medium"
                
            return ml_pred
        except Exception:
            return rule_pred

if __name__ == "__main__":
    # Test sequence
    classifier = HybridUrgencyModel()
    dummy_texts = [
        "Help me urgently! Fix this ASAP.", 
        "Just wanted to say thanks for the service.", 
        "This is somewhat important, the system is not working well."
    ]
    classifier.train_ml(dummy_texts)
    print("\nTesting Hybrid Urgency Engine:")
    for t in dummy_texts:
        print(f"'{t}' -> {classifier.predict(t, mode='hybrid')}")
