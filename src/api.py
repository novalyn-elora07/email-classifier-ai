from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import EmailPredictor

app = FastAPI(title="Smart Email Classifier API", version="1.0")

# Preload predictor globally
try:
    # Use advanced=True if you want DistilBERT by default
    predictor = EmailPredictor(use_advanced=False)
except Exception as e:
    predictor = None

class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    category: str
    urgency: str

@app.post("/predict", response_model=EmailResponse)
def predict_email(request: EmailRequest):
    if not predictor:
        return {"category": "SystemNotTrained", "urgency": "SystemNotTrained"}
    
    result = predictor.predict(request.email)
    return {
        "category": result["category"],
        "urgency": result["urgency"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "ready": predictor is not None}
