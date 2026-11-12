"""Airflow 3 DAG: validate, preprocess, train, evaluate/log, quality gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task


@dag(
    dag_id="equipment_training",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)},
    tags=["mlops", "training"],
)
def equipment_training():
    @task
    def validation() -> dict:
        import hashlib
        import os
        from pathlib import Path
        from airflow.sdk import get_current_context
        from training.train import load_dataset
        from training.log_experiment import dataset_provenance

        conf = get_current_context()["dag_run"].conf or {}
        if conf.get("inject_failure") == "validation":
            raise ValueError("Injected dataset validation failure")
        root = Path(os.environ["MLOPS_PROJECT_ROOT"])
        os.chdir(root)
        path = Path("data/raw/equipment.csv")
        features, labels = load_dataset(path)
        provenance = dataset_provenance(path)
        return {**provenance, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(labels), "positive_rows": sum(labels)}

    @task
    def preprocessing(validated: dict) -> dict:
        import os
        from pathlib import Path
        from training.train import load_dataset

        # The baseline trainer fits its scaler on training rows only, avoiding test leakage.
        if validated["rows"] < 20 or validated["positive_rows"] == 0:
            raise ValueError("Dataset is insufficient for training")
        root = Path(os.environ["MLOPS_PROJECT_ROOT"])
        features, _ = load_dataset(root / "data/raw/equipment.csv")
        ranges = [{"min": min(row[i] for row in features),
                   "max": max(row[i] for row in features)} for i in range(3)]
        return {**validated, "scaling": "training_split_only", "feature_ranges": ranges}

    @task
    def training(prepared: dict) -> dict:
        import hashlib
        import os
        from pathlib import Path
        from airflow.sdk import get_current_context
        from training.train import train

        context = get_current_context()
        conf = context["dag_run"].conf or {}
        seed = int(conf.get("seed", 42))
        if not 0 <= seed <= 100000:
            raise ValueError("seed out of range")
        root = Path(os.environ["MLOPS_PROJECT_ROOT"])
        os.chdir(root)
        run_key = hashlib.sha256(context["run_id"].encode()).hexdigest()[:16]
        output = root / "airflow-output" / run_key
        model_path, metrics_path = output / "model.json", output / "metrics.json"
        metrics = train(Path("data/raw/equipment.csv"), model_path, metrics_path, seed=seed)
        if metrics["dataset_sha256"] != prepared["dataset_sha256"]:
            raise ValueError("Dataset changed after validation")
        return {**prepared, "seed": seed, "model_path": str(model_path),
                "metrics_path": str(metrics_path), "f1": metrics["f1"]}

    @task
    def evaluation_mlflow(trained: dict) -> dict:
        import os
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        name = "equipment-airflow-training"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        run = client.create_run(experiment_id, tags={
            "dataset_git_commit": trained["dataset_git_commit"],
            "dataset_dvc_md5": trained["dataset_dvc_md5"],
            "dataset_sha256": trained["dataset_sha256"],
            "source": "airflow",
        })
        run_id = run.info.run_id
        try:
            import json
            from pathlib import Path
            metrics = json.loads(Path(trained["metrics_path"]).read_text())
            client.log_param(run_id, "seed", str(trained["seed"]))
            for key in ("f1", "accuracy", "precision", "recall"):
                client.log_metric(run_id, key, metrics[key])
            client.log_artifact(run_id, trained["model_path"], "model")
            client.log_artifact(run_id, trained["metrics_path"], "evaluation")
            client.set_terminated(run_id, status="FINISHED")
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
        return {**trained, "mlflow_run_id": run_id}

    @task
    def quality_gate(evaluated: dict) -> dict:
        from airflow.sdk import get_current_context

        conf = get_current_context()["dag_run"].conf or {}
        threshold = float(conf.get("minimum_f1", 0.75))
        if not 0 <= threshold <= 1:
            raise ValueError("quality threshold out of range")
        if conf.get("inject_failure") == "quality_gate" or evaluated["f1"] < threshold:
            raise ValueError(f"Quality gate failed: F1 {evaluated['f1']:.3f}, minimum {threshold:.3f}")
        return {"approved": True, "f1": evaluated["f1"],
                "mlflow_run_id": evaluated["mlflow_run_id"]}

    quality_gate(evaluation_mlflow(training(preprocessing(validation()))))


equipment_training()
