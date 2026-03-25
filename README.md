# AI Powered Smart Email Classifier for Enterprises

An end-to-end AI system that classifies enterprise emails into categories (Complaint, Request, Feedback, Spam) and detects their urgency levels (High, Medium, Low). 

This project includes a FastAPI backend, a Streamlit dashboard, and an automated preprocessing and training pipeline leveraging `scikit-learn` and Hugging Face `transformers`.

## Project Structure

```
email-classifier-project/
├── data/                  # Downloaded raw datasets / cleaned data
├── notebooks/             # Exploratory data analysis / PoC notebooks
├── src/                   # Source code
│   ├── preprocessing.py   # Dataset loading, cleaning, tokenization
│   ├── train_model.py     # Training baseline ML and advanced NLP models
│   ├── predict.py         # Inference functions
│   ├── urgency_model.py   # Rule-based & heuristic urgency detection
│   └── api.py             # FastAPI backend
├── app/                   # Web app code
│   └── streamlit_app.py   # Streamlit dashboard
├── models/                # Saved model weights and tokenizers
├── tests/                 # Unit tests (pytest)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Setup Instructions

### 1. Environment Setup
It is highly recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
To download data, clean it, and train the baseline (and advanced) models:

```bash
# Run preprocessing and model training
python src/train_model.py
```
*(Note: Training the DistilBERT model may take some time depending on hardware)*

### 3. Run the API Server
Start the FastAPI endpoint to serve predictions:

```bash
uvicorn src.api:app --reload
```
The API will be accessible at: `http://localhost:8000`. You can view the swagger UI documentation at `http://localhost:8000/docs`.

### 4. Run the Streamlit Dashboard
To run the interactive UI dashboard:

```bash
streamlit run app/streamlit_app.py
```
The dashboard typically opens at `http://localhost:8501`.

## Deployment

To deploy this, simply define a Dockerfile or use the standard `requirements.txt` via platforms like Heroku, Render, AWS Elastic Beanstalk, or a conventional VM setup.
