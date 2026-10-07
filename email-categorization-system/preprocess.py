"""Reusable text preprocessing for the Email Categorization System.

Pipeline: lowercase -> remove URLs -> remove email addresses -> remove special
characters -> tokenize -> remove stop words -> join back into one string.

NLTK is used when its data (punkt / stopwords) is available. If the data is
missing and cannot be downloaded (for example, no internet), the module falls
back to a built-in stop-word list and a regex tokenizer so the project still runs.
"""

import re

import nltk

# Words that carry little meaning for classification (used as the offline fallback).
_FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "s", "t", "ll",
    "d", "m", "re", "ve",
}


def _load_nltk_resources():
    """Try to load NLTK stop words and the tokenizer. Return (stop_words, tokenizer)."""
    try:
        for pkg in ("punkt", "punkt_tab", "stopwords"):
            nltk.download(pkg, quiet=True)
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words("english"))
        word_tokenize("test sentence")  # raises LookupError if punkt data is missing
        return stop_words, word_tokenize
    except Exception:  # offline, blocked network, missing data, etc.
        return _FALLBACK_STOP_WORDS, lambda text: re.findall(r"[a-z]+", text)


STOP_WORDS, _tokenize = _load_nltk_resources()


def clean_text(text):
    """Clean one email string and return space-separated, meaningful tokens."""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep only letters and spaces (removes digits and special characters)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and remove stop words and one-letter tokens
    tokens = [w for w in _tokenize(text) if w not in STOP_WORDS and len(w) > 1]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Congratulations!!! You WON $1000. Click https://example.com now or mail win@spam.com"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
