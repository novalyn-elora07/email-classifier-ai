from datasets import load_dataset
import pandas as pd

# Load dataset from HuggingFace
dataset = load_dataset("jason23322/high-accuracy-email-classifier")

print(dataset)

# Convert to pandas
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

# Save locally
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Dataset downloaded successfully")