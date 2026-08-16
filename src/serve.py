"""
Serving layer. Loads the locally pickled model (fast, no MLflow server
dependency) and exposes a /predict endpoint via FastAPI. Every request
is appended to data/prediction_logs.jsonl, which monitor_drift.py later
reads to compare live traffic against the training-time baseline.
"""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import ROOT, clean_text

MODEL_PATH = ROOT / "models" / "model.pkl"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.pkl"
LOG_PATH = ROOT / "data" / "prediction_logs.jsonl"

model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python src/train.py` "
            "(after `python src/data_prep.py`) first, or `dvc repro`."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    yield
    # nothing to clean up on shutdown


app = FastAPI(title="Sentiment Classifier API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty text after cleaning")

    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]
    proba = model.predict_proba(X).max()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "text_length": len(cleaned.split()),
                    "predicted_label": label,
                    "confidence": float(proba),
                }
            )
            + "\n"
        )

    return PredictResponse(label=label, confidence=float(proba))