"""Flask web app for the Email Categorization System.

Run:  python app.py   ->  http://127.0.0.1:5000
"""

import os

import joblib
from flask import Flask, jsonify, render_template, request

from preprocess import clean_text

MODEL_PATH = os.path.join("model", "email_classifier.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")
MAX_EMAIL_LENGTH = 10000

app = Flask(__name__)

# Load the trained model and vectorizer once, when the app starts
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    raise SystemExit("Model files not found. Run `python train_model.py` first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "email" not in data:
        return jsonify({"error": "Invalid request. Send JSON like {\"email\": \"...\"}."}), 400

    email_text = data["email"]
    if not isinstance(email_text, str):
        return jsonify({"error": "Email text must be a string."}), 400

    email_text = email_text.strip()
    if not email_text:
        return jsonify({"error": "Email text is required."}), 400
    if len(email_text) > MAX_EMAIL_LENGTH:
        return jsonify({"error": f"Email is too long (max {MAX_EMAIL_LENGTH} characters)."}), 400

    cleaned = clean_text(email_text)
    if not cleaned:
        return jsonify({"error": "The email has no meaningful words to analyse."}), 400

    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    best = probabilities.argmax()

    return jsonify({
        "category": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]), 2),
        "probabilities": {
            str(label): round(float(p), 4) for label, p in zip(model.classes_, probabilities)
        },
    })


if __name__ == "__main__":
    app.run(debug=True)
