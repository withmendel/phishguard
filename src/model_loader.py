from pathlib import Path
import joblib
import re
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

class PhishGuard:
    def __init__(self):
        self.email_model = joblib.load(MODEL_DIR / "email_model.pkl")
        self.email_vectorizer = joblib.load(MODEL_DIR / "email_vectorizer.pkl")
        self.url_model = joblib.load(MODEL_DIR / "url_model.pkl")

    def _url_features(self, url: str) -> np.ndarray:
        url = str(url)
        feats = [
            len(url), url.count("."), sum(c.isdigit() for c in url),
            sum(not c.isalnum() and c != "." for c in url),
            1 if url.startswith("https") else 0,
            1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
            1 if "@" in url else 0,
            1 if "//" in url[8:] else 0,
            url.split("//")[-1].split("/")[0].count(".") if "//" in url else url.split("/")[0].count("."),
            url.count("/"), url.count("="),
            len(url.split(".")[-1].split("/")[0]) if "." in url else 0,
        ]
        return np.array(feats).reshape(1, -1)

    def predict(self, email_text: str, url: str) -> dict:
        email_vec = self.email_vectorizer.transform([email_text])
        email_prob = float(self.email_model.predict_proba(email_vec)[0, 1])

        url_feats = self._url_features(url)
        url_prob = float(self.url_model.predict_proba(url_feats)[0, 1])

        combined = 0.6 * email_prob + 0.4 * url_prob

        return {
            "verdict": "phishing" if combined >= 0.5 else "legitimate",
            "combined_score": round(combined, 4),
            "email_probability": round(email_prob, 4),
            "url_probability": round(url_prob, 4),
        }

_bundle = None

def get_bundle() -> PhishGuard:
    global _bundle
    if _bundle is None:
        _bundle = PhishGuard()
    return _bundle
