"""
Compares recent production predictions (data/prediction_logs.jsonl,
written by serve.py) against the training-time baseline
(data/processed/baseline_stats.json, written by data_prep.py).

Two checks, both standard in production ML monitoring:
  1. Kolmogorov-Smirnov test on text-length distribution -> catches
     input drift (e.g. users suddenly sending much longer/shorter text).
  2. Population Stability Index (PSI) on predicted-class balance ->
     catches concept/label drift in what the model is outputting.

Writes drift_report.json and exits 1 if drift is flagged, so this can
gate a CI job or a scheduled GitHub Action directly.
"""
import json
import sys
import numpy as np
from scipy.stats import ks_2samp

from utils import ROOT, load_params

BASELINE_PATH = ROOT / "data" / "processed" / "baseline_stats.json"
LOG_PATH = ROOT / "data" / "prediction_logs.jsonl"
REPORT_PATH = ROOT / "models" / "drift_report.json"


def psi(baseline_dist: dict, current_dist: dict, eps: float = 1e-4) -> float:
    """Population Stability Index across the label categories."""
    labels = set(baseline_dist) | set(current_dist)
    score = 0.0
    for label in labels:
        b = baseline_dist.get(label, eps) or eps
        c = current_dist.get(label, eps) or eps
        score += (c - b) * np.log(c / b)
    return float(score)


def main():
    thresholds = load_params()["drift"]

    if not BASELINE_PATH.exists():
        print("No baseline stats found — run data_prep.py first.")
        sys.exit(2)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    if not LOG_PATH.exists():
        print("No prediction logs yet — nothing to check.")
        sys.exit(0)

    with open(LOG_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) < thresholds["min_predictions_for_check"]:
        print(
            f"Only {len(records)} predictions logged, need "
            f"{thresholds['min_predictions_for_check']} — skipping check."
        )
        sys.exit(0)

    current_lengths = [r["text_length"] for r in records]
    baseline_lengths = baseline["text_lengths_sample"]

    ks_stat, ks_pvalue = ks_2samp(baseline_lengths, current_lengths)
    length_drift = ks_pvalue < thresholds["text_length_ks_pvalue_threshold"]

    current_counts = {}
    for r in records:
        current_counts[r["predicted_label"]] = current_counts.get(r["predicted_label"], 0) + 1
    total = sum(current_counts.values())
    current_dist = {k: v / total for k, v in current_counts.items()}

    psi_score = psi(baseline["class_distribution"], current_dist)
    class_drift = psi_score > thresholds["class_balance_psi_threshold"]

    drift_detected = length_drift or class_drift

    report = {
        "n_predictions_checked": len(records),
        "text_length_ks_pvalue": ks_pvalue,
        "text_length_drift_flagged": length_drift,
        "class_distribution_current": current_dist,
        "class_distribution_baseline": baseline["class_distribution"],
        "psi_score": psi_score,
        "class_drift_flagged": class_drift,
        "drift_detected": drift_detected,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    sys.exit(1 if drift_detected else 0)


if __name__ == "__main__":
    main()
