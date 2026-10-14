import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path

RANDOM_STATE = 42

print("Loading MeAJOR...")
df = pd.read_csv("data/meajor.csv")
df = df.dropna(subset=["body", "label"])
df["body"] = df["body"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df["body"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
)

print(f"Train: {len(X_train)}  Test: {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

print("\n=== Email Model Results ===")
print(classification_report(y_test, y_pred, target_names=["benign", "phishing"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/email_model.pkl")
joblib.dump(vectorizer, "models/email_vectorizer.pkl")
print("Saved to models/")
