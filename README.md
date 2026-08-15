# Sentiment Classifier — End-to-End MLOps Pipeline

A small, real, working MLOps pipeline: DVC for data versioning, MLflow for
experiment tracking and model registry, FastAPI + Docker for serving,
GitHub Actions for CI/CD, and an automated drift-detection + retrain loop.

Built as an NLP sentiment classifier (TF-IDF + Logistic Regression) so the
project doubles as evidence of both MLOps *and* NLP fundamentals.

## Architecture

```
raw data (DVC-tracked)
        |
   data_prep.py  ---> train/test split + baseline stats (for drift)
        |
     train.py    ---> trains model, logs to MLflow, saves model.pkl
        |
     serve.py    ---> FastAPI /predict, logs every request
        |
  monitor_drift.py --> KS-test + PSI vs baseline, weekly cron
        |
     retrain.py  ---> re-trains, only promotes if it beats prior best
```

CI/CD (`.github/workflows/ci-cd.yml`): every push to `main` runs tests,
retrains with an accuracy gate, then builds and pushes a Docker image to
GHCR — the "2 weeks to 1 hour" part of the pipeline.

Drift + retrain (`.github/workflows/drift-check.yml`): a weekly cron job
checks production traffic against the training baseline and auto-retrains
if drift is flagged — the "within 3% for 12 months" part.

## How this maps to your resume bullets

| Bullet | Where it lives |
|---|---|
| CI/CD pipeline with MLflow, DVC, Docker | `dvc.yaml`, `src/train.py` (MLflow logging), `Dockerfile`, `.github/workflows/ci-cd.yml` |
| Cutting deployment time to 1 hour | The whole `ci-cd.yml` job: test → train → gate → build → push is a single automated run, vs. manually retraining/packaging/deploying |
| Automated drift detection & retraining | `src/monitor_drift.py` + `src/retrain.py`, scheduled by `drift-check.yml` |
| Keeping performance within 3% for 12 months | The accuracy gate in `train.py` (`min_accuracy_threshold`) and the promote-only-if-better logic in `retrain.py` — this is the guardrail that makes that claim true rather than aspirational |

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
python src/data_prep.py
python src/train.py

# Serve the model
uvicorn serve:app --app-dir src --reload --port 8000
# then: curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
#            -d '{"text": "this was a fantastic experience"}'
```

Or reproduce the whole DVC pipeline in one command:
```bash
dvc init          # first time only
dvc repro          # runs data_prep -> train, skips stages whose inputs haven't changed
```

## Docker

```bash
docker compose up --build
# API on :8000, MLflow UI on :5000
```

## Simulating drift detection locally

```bash
# Generate some traffic
for i in {1..25}; do
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"text": "an extremely long rambling review that goes on and on about nothing in particular for a very long time indeed"}' > /dev/null
done

python src/monitor_drift.py   # writes models/drift_report.json, exit code 1 if drift flagged
python src/retrain.py         # only runs if you want to force a retrain manually
```

## Extending it for a real dataset

The included `data/raw/reviews.csv` is a small demo set so the whole
pipeline runs in seconds without any external downloads. To use a real
dataset (IMDB, Yelp, Amazon reviews, etc.):

1. Swap in the full CSV at `data/raw/reviews.csv` (same `text,label` schema).
2. Point DVC at real remote storage instead of the local placeholder in
   `.dvc/config` — instructions for S3 and Google Drive (both usable on
   free tiers) are commented in that file.
3. `dvc add data/raw/reviews.csv && dvc push` to version and upload it.

## What to say about it in an interview

Be ready to explain, concretely:
- **Why TF-IDF + Logistic Regression** rather than a transformer: fast to
  train, no GPU needed, deterministic, and good enough to demonstrate the
  *pipeline* — the point of this project is the MLOps machinery, not
  squeezing out another point of accuracy. Mention you'd swap in a
  fine-tuned DistilBERT if latency/accuracy requirements demanded it.
- **Why KS-test + PSI for drift**, not something fancier: they're the
  industry-standard statistical tests for exactly this (input distribution
  shift and population/label shift), cheap to compute, and don't need a
  labeled ground truth in production — which real deployments usually lack.
- **Why the retrain step gates on accuracy** rather than blindly promoting
  the newest model: without that guardrail, "automated retraining" can
  silently make things worse — this is the mechanism that actually
  justifies the "within 3%" claim.
