import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

X = df["text"]
y = df["category"]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predictions
predictions = lr_model.predict(X_test)

print(classification_report(y_test, predictions))

# Save model
pickle.dump(lr_model, open("models/logistic_model.pkl","wb"))

print("Baseline model trained")
import pandas as pd

df = pd.read_csv("data/train.csv")
print(df.columns)
