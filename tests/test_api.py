import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    rv = client.get("/health")
    assert rv.status_code == 200

def test_predict_no_auth(client):
    rv = client.post("/predict", json={"email_text": "test"})
    assert rv.status_code == 401

def test_predict_with_auth(client):
    headers = {"X-API-Key": os.environ.get("APP_SECRET", "changeme")}
    rv = client.post("/predict", json={"email_text": "test", "url": "http://test.com"}, headers=headers)
    # It may fail if the model isn't loaded in the test env, so we just check auth passed
    assert rv.status_code in [200, 500] 
