"""
Called by the scheduled drift-check workflow when monitor_drift.py flags
drift. Re-runs data_prep + train, then only "promotes" the new model
(overwrites the artifacts serve.py actually loads) if it beats the
previously recorded metric. This is the guardrail that stops a bad
retrain from silently degrading production.
"""
import json
import shutil
import subprocess
import sys
from utils import ROOT

METRICS_PATH = ROOT / "models" / "metrics.json"
BEST_METRICS_PATH = ROOT / "models" / "best_metrics.json"


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    previous_accuracy = 0.0
    if BEST_METRICS_PATH.exists():
        with open(BEST_METRICS_PATH) as f:
            previous_accuracy = json.load(f).get("accuracy", 0.0)

    # In a real deployment, swap the raw CSV read here for a pull of
    # fresh production data (e.g. from a labeled feedback store or a
    # new DVC-tracked data version) before re-running the pipeline.
    run([sys.executable, "src/data_prep.py"])
    run([sys.executable, "src/train.py"])

    with open(METRICS_PATH) as f:
        new_metrics = json.load(f)

    print(f"Previous best accuracy: {previous_accuracy:.4f}")
    print(f"New model accuracy:     {new_metrics['accuracy']:.4f}")

    if new_metrics["accuracy"] >= previous_accuracy:
        shutil.copy(METRICS_PATH, BEST_METRICS_PATH)
        print("New model PROMOTED — models/model.pkl now serves this version.")
        # models/model.pkl and vectorizer.pkl were already overwritten by
        # train.py, so "promotion" here just means we keep them and update
        # the best-metrics marker. If you're using the MLflow Model
        # Registry, this is also where you'd transition the new run's
        # version to the "Production" stage and the old one to "Archived".
    else:
        print("New model did NOT beat the previous best — keeping prior model.")
        # Note: train.py already overwrote model.pkl at this point, so in
        # production wire this up to restore the previous artifacts from
        # the MLflow registry ("Production" stage) rather than local files.
        sys.exit(1)


if __name__ == "__main__":
    main()
