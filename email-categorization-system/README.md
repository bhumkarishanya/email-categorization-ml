# Email Categorization System

A machine-learning web app that classifies an email as **Spam**, **Work** or **Personal** and shows the prediction probability.

**Stack:** Python · Flask · NLTK · Scikit-learn (TF-IDF + Logistic Regression) · Pandas · NumPy · Joblib · HTML/CSS/JavaScript · Jupyter

```
email-categorization-system/
├── dataset/emails.csv            101 fictional labelled emails (35 Spam, 35 Work, 31 Personal)
├── model/                        saved model + vectorizer (re-created by train_model.py)
│   ├── email_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/email_classification.ipynb   step-by-step experiments with charts
├── templates/index.html          web page
├── static/css/style.css          styling
├── static/js/script.js           calls POST /predict and shows the result
├── app.py                        Flask server (GET /, POST /predict)
├── train_model.py                trains, compares and saves the model
├── preprocess.py                 reusable text-cleaning function
├── requirements.txt
└── README.md
```

## 1. Setup (Windows PowerShell)

```powershell
cd email-categorization-system
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

macOS/Linux: `python3 -m venv venv && source venv/bin/activate`

```powershell
pip install -r requirements.txt
```

## 2. Train the model

```powershell
python train_model.py
```

This loads the CSV, cleans the text, builds TF-IDF features, trains Logistic Regression and Multinomial Naive Bayes, prints accuracy / classification report / confusion matrix for both, then saves the better one to `model/`. The first run downloads NLTK data (`punkt`, `stopwords`); if you are offline the code falls back to a built-in stop-word list automatically.

## 3. Run the web app

```powershell
python app.py
```

Open http://127.0.0.1:5000

### API

```
POST /predict
Content-Type: application/json

{ "email": "Please attend the project meeting at 3 PM." }
```

Response:

```json
{
  "category": "Work",
  "confidence": 0.96,
  "probabilities": { "Personal": 0.0209, "Spam": 0.0234, "Work": 0.9558 }
}
```

Errors return HTTP 400 with `{ "error": "..." }` (empty text, invalid JSON, missing `email` field, non-string value, text over 10,000 characters).

PowerShell test:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"email":"Win a free prize now"}'
```

## 4. Jupyter Notebook

```powershell
jupyter notebook
```

Open `notebook/email_classification.ipynb` and run all cells. It walks through data inspection, cleaning, TF-IDF, both models, confusion matrices, accuracy comparison and the top words per class.

## 5. What each file does

| File | Purpose |
|---|---|
| `dataset/emails.csv` | Training data with `text` and `category` columns. All emails are fictional. |
| `preprocess.py` | `clean_text()`: lowercase, strip URLs and email addresses, remove special characters, tokenize, remove stop words. Used by both training and the web app so they always agree. |
| `train_model.py` | Full training pipeline. Stratified 75/25 split, TF-IDF (1-2 word n-grams), trains both models, selects the best by accuracy, saves with Joblib. |
| `app.py` | Loads the saved model once at startup. `GET /` serves the page; `POST /predict` cleans the text, vectorizes it and returns the category plus probabilities. |
| `templates/index.html` | Input box, Classify/Clear buttons, example emails, result card. |
| `static/css/style.css` | Responsive layout; red = Spam, blue = Work, green = Personal. |
| `static/js/script.js` | Sends the email to `/predict`, shows the badge, confidence meter and per-class breakdown, handles errors. |
| `notebook/email_classification.ipynb` | Experiments and visualisations. |

## 6. How the ML works

1. **Cleaning:** reduce each email to its meaningful lowercase words (stop words like "the", "is" carry no signal).
2. **TF-IDF:** converts text to numbers. *Term Frequency* counts how often a word appears in an email; *Inverse Document Frequency* down-weights words that appear in almost every email. Distinctive words such as "prize" or "meeting" get high weights. This project also uses bigrams (two-word phrases such as "free prize").
3. **Logistic Regression (multiclass):** learns one weight per word per class. For a new email it computes a score for each class and converts the scores to probabilities with softmax. The highest probability is the prediction and also the "confidence".
4. **Multinomial Naive Bayes:** applies Bayes' theorem, assuming words are independent given the class. It is very fast and works well for small text datasets.
5. **Model selection:** both are scored on the same held-out test set; the better one is saved (ties go to Logistic Regression because its probabilities are better calibrated).

**Metrics:** accuracy = correct predictions / total; precision = of the emails predicted as a class, how many were right; recall = of the real emails of a class, how many were found; F1 = harmonic mean of precision and recall.

**Current results:** about 85% accuracy on the 26-email test set. With so few samples that number is noisy; the biggest improvement you can make is more data (hundreds or thousands of real-style emails), especially for Personal vs Work, which share words like "meeting" and "tomorrow".

## 7. Interview questions and answers

**Q: Why TF-IDF instead of plain word counts?**
Counts overweight common words. TF-IDF scales each word by how rare it is across all emails, so informative words dominate.

**Q: Why fit the vectorizer only on the training data?**
Fitting on all data leaks information from the test set into training (data leakage) and makes the evaluation too optimistic.

**Q: Why stratified splitting?**
It keeps the same Spam/Work/Personal proportions in the train and test sets, which matters with small, slightly imbalanced data.

**Q: Why compare Logistic Regression and Naive Bayes?**
They are strong, fast baselines for text. Naive Bayes needs little data; Logistic Regression usually does better with more data and gives well-behaved probabilities.

**Q: What does "confidence" mean?**
It is the model's predicted probability for the winning class. It is not a guarantee of correctness, and probabilities can be over- or under-confident on unusual text.

**Q: Why use the same `clean_text` in training and in the app?**
The model only sees cleaned text during training, so predictions must be preprocessed identically (otherwise training-serving skew).

**Q: Why save the vectorizer as well as the model?**
The model's weights are tied to the vocabulary and IDF values learned during training. Without the same vectorizer the input features would not match.

**Q: How would you improve it?**
More and more varied data, cross-validation, hyperparameter tuning (`GridSearchCV`), class weights, character n-grams for obfuscated spam, and transformer embeddings (e.g. BERT) for better meaning understanding. Also add header features such as sender domain and subject.

**Q: How would you handle an email that fits no category?**
Add a confidence threshold: if the top probability is below, say, 0.5, return "Uncertain" and route it for manual review.

**Q: How would you deploy it?**
Run Flask behind Gunicorn (and a reverse proxy like Nginx) or in a Docker container, loading the saved model at startup; retrain offline and ship new `.pkl` files.

## 8. Resume description

**Email Categorization System** | Python, Flask, Scikit-learn, NLTK, Pandas, HTML/CSS/JS
- Built an NLP pipeline (cleaning, tokenization, stop-word removal, TF-IDF with bigrams) to classify emails into Spam, Work and Personal.
- Trained and compared Logistic Regression and Multinomial Naive Bayes using stratified splits; evaluated with accuracy, precision, recall, F1 and confusion matrices; persisted the best model with Joblib.
- Developed a Flask REST API (`POST /predict`) with input validation and a responsive web UI that shows the predicted category and per-class probabilities.
- Documented experiments in a Jupyter notebook with visualisations of data distribution, model comparison and top predictive words.

## 9. Notes and limitations

- The dataset is small and fictional, intended for learning. Do not treat the accuracy as production-grade.
- `app.py` runs Flask in debug mode for development. Turn `debug` off for any real deployment.
