"""Exports the three trained models as small JSON files the website runs in the browser.

Run from the repo root:  python website/build_models.py
"""
import json, re, string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
RS = 42

# ---------- Task 1: Iris, KNN (k=5) on standardised features ----------
iris = load_iris(as_frame=True)
X, y = iris.data.values, iris.target_names[iris.target]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
sc = StandardScaler().fit(Xtr)
knn = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
acc = accuracy_score(yte, knn.predict(sc.transform(Xte)))
json.dump({"k": 5, "features": iris.feature_names, "mean": sc.mean_.round(6).tolist(),
           "scale": sc.scale_.round(6).tolist(), "X": Xtr.round(2).tolist(), "y": list(ytr),
           "test_accuracy": round(acc, 4)}, open(OUT / "iris.json", "w"))
print("iris test acc", acc)

# ---------- Task 4: Spam, TF-IDF + Multinomial Naive Bayes ----------
# Same cleaning as the notebook except stemming, so the browser can reproduce it exactly.
STOP = set(ENGLISH_STOP_WORDS)
def clean(t):
    t = t.lower()
    t = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", t)
    t = re.sub(r"\S+@\S+", " emailtoken ", t)
    t = re.sub(r"\d{5,}", " numtoken ", t)
    t = t.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    t = re.sub(r"\d+", " ", t)
    return " ".join(w for w in t.split() if w not in STOP and len(w) > 1)

df = pd.read_csv("DataScience-Task4-EmailSpamDetection/data/spam.csv").drop_duplicates("message")
df["clean"] = df["message"].apply(clean)
tr, te = train_test_split(df, test_size=0.2, random_state=RS, stratify=df["label"])
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
Xtr, Xte = tf.fit_transform(tr["clean"]), tf.transform(te["clean"])
nb = MultinomialNB(alpha=0.1).fit(Xtr, tr["label"])
p = nb.predict(Xte)
print("spam acc", accuracy_score(te["label"], p), "f1", f1_score(te["label"], p, pos_label="spam"))
vocab = tf.get_feature_names_out()
ham_i, spam_i = list(nb.classes_).index("ham"), list(nb.classes_).index("spam")
json.dump({
    "stopwords": sorted(STOP),
    "terms": {t: [round(float(tf.idf_[i]), 5), round(float(nb.feature_log_prob_[spam_i, i] - nb.feature_log_prob_[ham_i, i]), 5)]
              for i, t in enumerate(vocab)},
    "prior": round(float(nb.class_log_prior_[spam_i] - nb.class_log_prior_[ham_i]), 6),
    "test_accuracy": round(accuracy_score(te["label"], p), 4),
}, open(OUT / "spam.json", "w"), separators=(",", ":"))

# sanity check: replicate the browser maths in numpy
def score(msg):
    toks = clean(msg).split()
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    counts = pd.Series(grams).value_counts() if grams else pd.Series(dtype=float)
    vec = {g: (1 + np.log(c)) * tf.idf_[tf.vocabulary_[g]] for g, c in counts.items() if g in tf.vocabulary_}
    norm = np.sqrt(sum(v * v for v in vec.values())) or 1
    return nb.class_log_prior_[spam_i] - nb.class_log_prior_[ham_i] + sum(
        v / norm * (nb.feature_log_prob_[spam_i, tf.vocabulary_[g]] - nb.feature_log_prob_[ham_i, tf.vocabulary_[g]]) for g, v in vec.items())
manual = np.array(["spam" if score(m) > 0 else "ham" for m in te["message"]])
print("browser-maths agreement", (manual == p).mean())

# ---------- Task 5: Sales, degree-2 polynomial regression ----------
ad = pd.read_csv("DataScience-Task5-SalesPrediction/data/Advertising.csv")
X, y = ad[["TV", "Radio", "Newspaper"]], ad["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
poly = PolynomialFeatures(degree=2, include_bias=False).fit(Xtr)
lr = LinearRegression().fit(poly.transform(Xtr), ytr)
r2 = r2_score(yte, lr.predict(poly.transform(Xte)))
print("sales r2", r2)
json.dump({"terms": poly.get_feature_names_out(["TV", "Radio", "Newspaper"]).tolist(),
           "coef": lr.coef_.round(10).tolist(), "intercept": round(float(lr.intercept_), 8),
           "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X}, "test_r2": round(r2, 4)},
          open(OUT / "sales.json", "w"))
