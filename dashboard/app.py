"""
Streamlit dashboard for the sentiment classifier pipeline.

Two panels:
  1. MLflow run history — every training run's params/metrics, pulled
     live from the MLflow tracking server.
  2. Drift monitoring — the latest models/drift_report.json written by
     src/monitor_drift.py, showing whether production traffic has
     drifted from the training baseline.

Run locally:
    export MLFLOW_TRACKING_URI=http://localhost:5000   # if using docker-compose's mlflow service
    streamlit run dashboard/app.py

Works even without a reachable MLflow server or without a drift report
yet — each panel degrades to an informative message instead of crashing.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

METRICS_PATH = ROOT / "models" / "metrics.json"
DRIFT_REPORT_PATH = ROOT / "models" / "drift_report.json"
EXPERIMENT_NAME = "sentiment-classifier"

st.set_page_config(page_title="Sentiment Classifier — MLOps Dashboard", layout="wide")
st.title("Sentiment Classifier — MLOps Dashboard")


# ---------------------------------------------------------------- sidebar --
tracking_uri = st.sidebar.text_input(
    "MLflow tracking URI",
    value=st.session_state.get("tracking_uri", "http://localhost:5000"),
)
st.session_state["tracking_uri"] = tracking_uri


# ------------------------------------------------------- current model ----
st.header("Current Model")
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name.capitalize(), f"{value:.3f}")
else:
    st.info("No metrics.json yet — run `python src/train.py` first.")


# ------------------------------------------------------ MLflow run history -
st.header("MLflow Run History")
try:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        st.warning(f"No experiment named '{EXPERIMENT_NAME}' found yet at {tracking_uri}.")
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        if not runs:
            st.info("Experiment exists but has no runs yet.")
        else:
            rows = []
            for r in runs:
                row = {"run_id": r.info.run_id[:8], "start_time": pd.to_datetime(r.info.start_time, unit="ms")}
                row.update(r.data.metrics)
                row.update({f"param_{k}": v for k, v in r.data.params.items()})
                rows.append(row)
            df = pd.DataFrame(rows).sort_values("start_time")

            st.dataframe(df, use_container_width=True)

            if "accuracy" in df.columns:
                st.subheader("Accuracy over runs")
                st.line_chart(df.set_index("start_time")["accuracy"])

            # Registered model versions + stages, if any are registered
            try:
                versions = client.search_model_versions(f"name='sentiment-classifier'")
                if versions:
                    st.subheader("Model Registry")
                    reg_df = pd.DataFrame(
                        [
                            {"version": v.version, "stage": v.current_stage, "run_id": v.run_id[:8]}
                            for v in versions
                        ]
                    ).sort_values("version", ascending=False)
                    st.dataframe(reg_df, use_container_width=True)
            except Exception:
                pass  # registry not set up — fine, just skip this section

except Exception as e:
    st.warning(
        f"Couldn't reach MLflow at `{tracking_uri}` ({e}). "
        "Start it with `docker compose up mlflow` or point the sidebar "
        "field at your tracking server."
    )


# --------------------------------------------------------- drift monitor --
st.header("Drift Monitoring")
if DRIFT_REPORT_PATH.exists():
    with open(DRIFT_REPORT_PATH) as f:
        report = json.load(f)

    status = "DRIFT DETECTED" if report["drift_detected"] else "No drift"
    (st.error if report["drift_detected"] else st.success)(status)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Text-length KS p-value", f"{report['text_length_ks_pvalue']:.4f}")
        st.caption("Below threshold = input distribution has shifted")
    with col2:
        st.metric("Class-balance PSI score", f"{report['psi_score']:.4f}")
        st.caption("Above threshold = predicted label balance has shifted")

    st.subheader("Predicted class distribution: baseline vs. current")
    dist_df = pd.DataFrame(
        {
            "baseline": report["class_distribution_baseline"],
            "current": report["class_distribution_current"],
        }
    ).fillna(0)
    st.bar_chart(dist_df)

    st.caption(f"Based on {report['n_predictions_checked']} recent predictions")
else:
    st.info(
        "No drift report yet. Generate some traffic against the API, then run "
        "`python src/monitor_drift.py`."
    )
