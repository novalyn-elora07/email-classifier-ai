import os
import pandas as pd
import numpy as np
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Removing HTML tags
    2. Lowercasing
    3. Removing special characters
    4. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Lowercase text
    text = text.lower()
    
    # 3. Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Remove stopwords
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    
    text = ' '.join(words)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data():
    """
    Loads the dataset from Hugging Face and applies preprocessing.
    Returns train and test dataframes.
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("jason23322/high-accuracy-email-classifier")
    
    # Convert to pandas dataframe (using train split as everything is likely there)
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        splits = list(dataset.keys())
        df = dataset[splits[0]].to_pandas()
        
    print(f"Initial shape: {df.shape}")
    print(f"Columns found: {df.columns.tolist()}")
    
    # Map text and label columns dynamically
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    
    if 'category' in df.columns:
        label_col = 'category'
    elif 'label' in df.columns:
        label_col = 'label'
    elif 'label_text' in df.columns:
        label_col = 'label_text'
    else:
        label_col = df.columns[1]
        
    print(f"Using text column: {text_col}, label column: {label_col}")
    
    print("Cleaning text...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    # For robust classification, ensure no NaNs
    df = df.dropna(subset=['clean_text', label_col])
    
    print("Splitting into train/test (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df, text_col, label_col

if __name__ == "__main__":
    train, test, text_col, label_col = load_and_preprocess_data()
    print("\nSample cleaned text:")
    print(train[['clean_text', label_col]].head())
    
    # Save a small sample for fast local testing
    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train_sample.csv", index=False)
    test.to_csv("data/test_sample.csv", index=False)
    print("Saved samples to data/ directory.")
