import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from src.preprocessing import load_and_preprocess_data
from src.urgency_model import HybridUrgencyModel

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

def train_baselines(train_df, test_df, text_col, label_col):
    print("\nTraining Baseline Models...")
    vectorizer = TfidfVectorizer(max_features=10000)
    
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df[label_col].astype(str)
    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df[label_col].astype(str)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    evaluate_model(y_test, lr_preds, "Logistic Regression")
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    evaluate_model(y_test, nb_preds, "Naive Bayes")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(nb, "models/naive_bayes.pkl")
    print("Baseline models saved to models/")
    return lr, vectorizer

def train_distilbert(train_df, test_df, text_col, label_col):
    print("\nFine-tuning DistilBERT...")
    os.environ["WANDB_DISABLED"] = "true"
    
    # Map labels
    labels = train_df[label_col].astype(str).unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    train_df['label_id'] = train_df[label_col].astype(str).map(label2id)
    test_df['label_id'] = test_df[label_col].astype(str).map(label2id)
    
    # Limit to 500 samples so local execution is fast
    train_subset = train_df.sample(min(500, len(train_df)), random_state=42)
    test_subset = test_df.sample(min(100, len(test_df)), random_state=42)
    
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_subset['clean_text'].tolist(), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(test_subset['clean_text'].tolist(), truncation=True, padding=True, max_length=64)
    
    class EmailDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = EmailDataset(train_encodings, train_subset['label_id'].tolist())
    test_dataset = EmailDataset(test_encodings, test_subset['label_id'].tolist())
    
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), label2id=label2id, id2label=id2label)
    
    training_args = TrainingArguments(
        output_dir='./models/results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./models/logs',
        logging_steps=10,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    
    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    print("DistilBERT model saved to models/distilbert/")
    
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(-1)
    y_true = preds.label_ids
    evaluate_model(y_true, y_pred, "DistilBERT")

def train_urg_model(train_df, text_col):
    model = HybridUrgencyModel()
    model.train_ml(train_df['clean_text'].tolist()[:5000])
    model.save("models/urgency_hybrid.pkl")

if __name__ == "__main__":
    train_df, test_df, text_col, label_col = load_and_preprocess_data()
    train_urg_model(train_df, text_col)
    train_baselines(train_df, test_df, text_col, label_col)
    train_distilbert(train_df, test_df, text_col, label_col)
