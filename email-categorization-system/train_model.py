"""Train, compare and save the email classifier.

Run:  python train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text

DATASET_PATH = "dataset/emails.csv"
MODEL_DIR = "model"
LABELS = ["Personal", "Spam", "Work"]  # fixed order for the confusion matrix

# 1. Load the dataset
data = pd.read_csv(DATASET_PATH)
data = data.dropna(subset=["text", "category"]).drop_duplicates(subset=["text"])
print("Dataset loaded successfully.")
print("Number of emails:", len(data))
print(data["category"].value_counts().to_string(), "\n")

# 2. Clean the text
data["clean_text"] = data["text"].apply(clean_text)

X = data["clean_text"]
y = data["category"]

# 3. Stratified train/test split (keeps the class ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. TF-IDF vectorization (fit on training data only to avoid leakage)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Candidate models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=10, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = (model, accuracy)

    print(name)
    print("-" * len(name))
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification report (precision, recall, F1-score):")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted)")
    print("Order:", LABELS)
    print(confusion_matrix(y_test, predictions, labels=LABELS), "\n")

# 6. Pick the better model (ties go to Logistic Regression, which gives better probabilities)
best_name = max(results, key=lambda n: (results[n][1], n == "Logistic Regression"))
best_model = results[best_name][0]
print(f"{best_name} selected (accuracy {results[best_name][1]:.2f}).\n")

# 7. Save the model and vectorizer
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "email_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("Model saved successfully.")
print(" -", os.path.join(MODEL_DIR, "email_classifier.pkl"))
print(" -", os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
