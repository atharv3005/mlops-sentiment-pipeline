"""
Stage 2 of the pipeline: train the sentiment classifier, log the run to
MLflow (params + metrics + model artifact), register it in the MLflow
Model Registry, and also drop a plain pickle to models/ so serve.py can
run without a live MLflow server if needed.

Exits with a non-zero status if accuracy falls below the threshold in
params.yaml — this is what lets CI fail a bad training run before it
ever gets deployed.
"""
import json
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import ROOT, load_params

MODEL_NAME = "sentiment-classifier"


def main():
    all_params = load_params()
    p = all_params["train"]

    train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")

    vectorizer = TfidfVectorizer(
        max_features=p["max_features"],
        ngram_range=(p["ngram_range_min"], p["ngram_range_max"]),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train, y_test = train_df["label"], test_df["label"]

    mlflow.set_tracking_uri(mlflow.get_tracking_uri())  # honors $MLFLOW_TRACKING_URI if set
    mlflow.set_experiment("sentiment-classifier")

    with mlflow.start_run() as run:
        clf = LogisticRegression(
            C=p["C"], max_iter=p["max_iter"], random_state=p["random_state"]
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, pos_label="positive"),
            "precision": precision_score(y_test, preds, pos_label="positive"),
            "recall": recall_score(y_test, preds, pos_label="positive"),
        }

        mlflow.log_params(p)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Run {run.info.run_id} -> {metrics}")

        # Local artifacts for the Docker-served API (works even without
        # a reachable MLflow tracking server, e.g. in a lightweight demo).
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(clf, models_dir / "model.pkl")
        joblib.dump(vectorizer, models_dir / "vectorizer.pkl")
        with open(models_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    if metrics["accuracy"] < p["min_accuracy_threshold"]:
        print(
            f"FAILED: accuracy {metrics['accuracy']:.3f} is below "
            f"threshold {p['min_accuracy_threshold']}"
        )
        sys.exit(1)

    print("Training run passed the accuracy gate.")


if __name__ == "__main__":
    main()
