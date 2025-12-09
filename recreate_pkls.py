# recreate_pkls.py
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spam.csv")
TEXT_COL = "text"
LABEL_COL = "label"

# If spam.csv missing, create a small sample dataset for testing
if not os.path.exists(CSV_PATH):
    print(f"[INFO] {CSV_PATH} not found — creating a small sample dataset for testing.")
    sample = [
        {"text": "Free entry! You won a prize. Click here", "label": "spam"},
        {"text": "Call me when you are free", "label": "ham"},
        {"text": "Congratulations! Claim your free gift now", "label": "spam"},
        {"text": "Are we meeting tomorrow for lunch?", "label": "ham"},
        {"text": "Win cash now by entering the contest", "label": "spam"},
        {"text": "Please review the attached file", "label": "ham"},
    ]
    df = pd.DataFrame(sample)
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Sample dataset written to {CSV_PATH}")
else:
    print(f"[INFO] Found dataset at {CSV_PATH}. Loading it...")
    df = pd.read_csv(CSV_PATH)

# Verify columns
if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise RuntimeError(f"CSV must contain columns '{TEXT_COL}' and '{LABEL_COL}'. Found: {list(df.columns)}")

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].tolist()

print("[INFO] Fitting TfidfVectorizer and training MultinomialNB...")
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vec.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

vpath = os.path.join(BASE_DIR, "vectorizer1.pkl")
mpath = os.path.join(BASE_DIR, "model1.pkl")

with open(vpath, "wb") as f:
    pickle.dump(vec, f)
with open(mpath, "wb") as f:
    pickle.dump(model, f)

print("[SUCCESS] Saved vectorizer ->", vpath)
print("[SUCCESS] Saved model ->", mpath)
