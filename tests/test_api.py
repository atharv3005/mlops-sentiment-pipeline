import pytest
from fastapi.testclient import TestClient
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifacts missing — run `python src/data_prep.py && python src/train.py` first",
)


@pytest.fixture
def client():
    from serve import app  # imported lazily so the skip check above runs first
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_label_and_confidence(client):
    resp = client.post("/predict", json={"text": "I absolutely loved this, fantastic!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("positive", "negative")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_text(client):
    resp = client.post("/predict", json={"text": "   !!! ???  "})
    assert resp.status_code == 400
