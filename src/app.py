from flask import Flask, jsonify, request
from model_loader import get_bundle

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
