"""MLflow experiment tracking integration (no-op when MLflow is unavailable)."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_training_run(metrics: dict, params: dict | None = None) -> str | None:
    """Log a training run to MLflow and return the run ID, or None if MLflow is unavailable."""
    try:
        import mlflow

        with mlflow.start_run() as run:
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if Path("model.joblib").exists():
                mlflow.log_artifact("model.joblib")
            if Path("metrics.json").exists():
                mlflow.log_artifact("metrics.json")

            run_id = run.info.run_id
            logger.info("MLflow run logged: %s metrics=%s", run_id, metrics)
            return run_id
    except ImportError:
        logger.debug("MLflow not installed — training run not logged to experiment tracker")
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
    return None


def get_best_run_metrics() -> dict:
    """Retrieve the best run's metrics from the local MLflow store if available."""
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            return {}

        runs = client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
            order_by=["metrics.auc_mean DESC"],
            max_results=1,
        )
        if runs:
            return dict(runs[0].data.metrics)
    except Exception as exc:
        logger.debug("MLflow query failed: %s", exc)

    # Fall back to local metrics.json
    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}
