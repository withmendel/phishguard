import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from pathlib import Path

def extract_features(url: str) -> dict:
    url = str(url)
    return {
        "length": len(url),
        "num_dots": url.count("."),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special": sum(not c.isalnum() and c != "." for c in url),
        "has_https": 1 if url.startswith("https") else 0,
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_at": 1 if "@" in url else 0,
        "has_double_slash": 1 if "//" in url[8:] else 0,
        "subdomain_count": url.split("//")[-1].split("/")[0].count(".") if "//" in url else url.split("/")[0].count("."),
        "path_depth": url.count("/"),
        "query_params": url.count("="),
        "tld_len": len(url.split(".")[-1].split("/")[0]) if "." in url else 0,
    }

print("Loading PhiUSIIL...")
df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df[["URL", "label"]].dropna()

print("Extracting lexical features...")
features = df["URL"].apply(extract_features).apply(pd.Series)
X = features.values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== URL Model Results ===")
print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/url_model.pkl")
print("Saved to models/url_model.pkl")
