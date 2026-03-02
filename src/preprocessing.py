import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/train.csv")

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub('<.*?>',' ',text)

    # Remove special characters
    text = re.sub('[^a-zA-Z ]',' ',text)

    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv("data/cleaned_dataset.csv", index=False)

print("Preprocessing Completed")
