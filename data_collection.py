from datasets import load_dataset
import pandas as pd

# Paste your HuggingFace token below
token = "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"

# Load dataset
dataset = load_dataset(
    "jason23322/high-accuracy-email-classifier",
    token=token
)

print("Dataset Loaded Successfully")

# Convert to pandas
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Save locally
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset Saved Successfully")