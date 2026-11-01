from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from model_loader import get_bundle
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Rate limiting: 5 requests per minute per IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"],
    storage_uri="memory://"
)

def require_api_key():
    api_key = request.headers.get("X-API-Key")
    expected = os.environ.get("APP_SECRET", "changeme")
    if not api_key or api_key != expected:
        return False
    return True

@app.route("/health", methods=["GET"])
@limiter.exempt
def health():
    return jsonify({
        "status": "ok",
        "version": "1.1.0"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not require_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Body must be JSON object"}), 400

    email_text = data.get("email_text", "")
    url = data.get("url", "")

    if not email_text and not url:
        return jsonify({"error": "Provide at least 'email_text' or 'url'"}), 400

    try:
        result = get_bundle().predict(email_text, url)
        return jsonify(result), 200
    except Exception:
        return jsonify({"error": "internal error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
