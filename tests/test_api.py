import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import app
from src.preprocessing import clean_text

client = TestClient(app)

def test_clean_text():
    raw = "<b>Hello!</b> Check this out http://link.com"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "hello" in cleaned.lower()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    response = client.post("/predict", json={"email": "This is a test email message for complaint."})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "urgency" in data
