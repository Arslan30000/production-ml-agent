"""Train the baseline and record real metrics, artifacts, and DVC provenance in MLflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from training.train import train


def _command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def dataset_provenance(dataset: Path) -> dict[str, str]:
    """Fail closed if the working data differs from the committed DVC revision."""
    if dataset.as_posix() != "data/raw/equipment.csv":
        raise ValueError("only the approved DVC dataset can be logged")
    commit = _command("git", "rev-parse", "HEAD")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "data/raw/equipment.csv.dvc"],
                      check=False, timeout=10).returncode != 0:
        raise ValueError("DVC pointer differs from the committed Git revision")
    pointer = _command("git", "show", "HEAD:data/raw/equipment.csv.dvc")
    match = re.search(r"(?m)^- md5: ([0-9a-f]{32})$", pointer)
    if match is None:
        raise ValueError("committed DVC pointer has no dataset hash")
    status = json.loads(_command("dvc", "status", "--json"))
    if status:
        raise ValueError("DVC outputs or data have uncommitted changes")
    return {"dataset_git_commit": commit, "dataset_dvc_md5": match.group(1)}


def log_experiment(*, seed: int, experiment_name: str, tracking_uri: str,
                   dataset: Path = Path("data/raw/equipment.csv")) -> dict:
    from mlflow.tracking import MlflowClient

    provenance = dataset_provenance(dataset)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    run = client.create_run(experiment_id, tags={
        **provenance,
        "model_type": "logistic_regression_baseline",
    })
    run_id = run.info.run_id
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            metrics_path = root / "metrics.json"
            metrics = train(dataset, model_path, metrics_path, seed=seed)
            client.log_param(run_id, "seed", str(seed))
            client.log_param(run_id, "train_rows", str(metrics["train_rows"]))
            client.log_param(run_id, "test_rows", str(metrics["test_rows"]))
            for key in ("accuracy", "precision", "recall", "f1"):
                client.log_metric(run_id, key, metrics[key])
            client.set_tag(run_id, "dataset_sha256", metrics["dataset_sha256"])
            client.log_artifact(run_id, str(model_path), "model")
            client.log_artifact(run_id, str(metrics_path), "evaluation")
        client.set_terminated(run_id, status="FINISHED")
        return {"run_id": run_id, "experiment_id": experiment_id, "metrics": metrics,
                "provenance": provenance}
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="equipment-anomaly-baseline")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    print(json.dumps(log_experiment(seed=args.seed, experiment_name=args.experiment,
                                    tracking_uri=args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
