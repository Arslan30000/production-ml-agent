"""Read experiment evidence through the supported MLflow client API."""

from __future__ import annotations

import re
from typing import Any


class MlflowToolError(RuntimeError):
    """MLflow evidence could not be retrieved."""


METRIC = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,99}$")


def _run_record(run: Any) -> dict:
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


class MlflowTools:
    def __init__(self, client: Any):
        self.client = client

    def list_experiments(self) -> dict:
        try:
            experiments = self.client.search_experiments()
            return {"experiments": [
                {"id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage}
                for e in experiments
            ]}
        except Exception as exc:
            raise MlflowToolError("MLflow experiments unavailable") from exc

    def list_runs(self, experiment_id: str, *, limit: int = 20) -> dict:
        if not experiment_id or not 1 <= limit <= 100:
            raise ValueError("invalid experiment ID or limit")
        try:
            runs = self.client.search_runs([experiment_id], max_results=limit,
                                           order_by=["attributes.start_time DESC"])
            return {"experiment_id": experiment_id, "runs": [_run_record(r) for r in runs]}
        except Exception as exc:
            raise MlflowToolError("MLflow runs unavailable") from exc

    def run_details(self, run_id: str) -> dict:
        if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
            raise ValueError("invalid run ID")
        try:
            run = self.client.get_run(run_id)
            artifacts = self.client.list_artifacts(run_id)
            return {**_run_record(run), "artifacts": [
                {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
                for a in artifacts
            ]}
        except Exception as exc:
            raise MlflowToolError("MLflow run details unavailable") from exc

    def compare_runs(self, run_ids: list[str], *, metric: str = "f1") -> dict:
        if not 2 <= len(run_ids) <= 20 or len(set(run_ids)) != len(run_ids):
            raise ValueError("provide 2 to 20 distinct run IDs")
        if not METRIC.fullmatch(metric):
            raise ValueError("invalid metric")
        runs = [self.run_details(run_id) for run_id in run_ids]
        return {"metric": metric, "runs": runs,
                "note": "Compare data versions and quality gates before production promotion."}

    def compare_experiments(self, experiment_ids: list[str], *, metric: str = "f1",
                            limit: int = 20) -> dict:
        if not 2 <= len(experiment_ids) <= 10 or len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("provide 2 to 10 distinct experiment IDs")
        return {"metric": metric,
                "experiments": [{"experiment_id": experiment_id,
                                 "selection": self.best_run(experiment_id, metric=metric, limit=limit)}
                                for experiment_id in experiment_ids],
                "caveat": "Check dataset versions and evaluation design before comparing scores."}

    def best_run(self, experiment_id: str, *, metric: str = "f1", limit: int = 20,
                 higher_is_better: bool = True) -> dict:
        if not METRIC.fullmatch(metric):
            raise ValueError("invalid metric")
        recent = self.list_runs(experiment_id, limit=limit)["runs"]
        candidates = [r for r in recent if r["status"] == "FINISHED"
                      and isinstance(r["metrics"].get(metric), (int, float))]
        if not candidates:
            return {"metric": metric, "candidate": None, "considered": len(recent),
                    "reason": "No finished run has this recorded metric."}
        selected = sorted(candidates, key=lambda r: (
            -r["metrics"][metric] if higher_is_better else r["metrics"][metric],
            -r["start_time"], r["run_id"]))[0]
        return {"metric": metric, "candidate": selected, "considered": len(recent),
                "reason": "Highest recorded metric among finished recent runs." if higher_is_better
                else "Lowest recorded metric among finished recent runs.",
                "production_ready": None,
                "caveat": "Metric rank alone does not establish production readiness."}

    def model_versions(self, model_name: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", model_name):
            raise ValueError("invalid model name")
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return {"model_name": model_name, "versions": [
                {"version": v.version, "run_id": v.run_id, "status": v.status,
                 "source": v.source} for v in versions
            ]}
        except Exception as exc:
            raise MlflowToolError("Model registry unavailable") from exc


TOOL_SCHEMAS = [
    {"name": "mlflow_list_experiments", "inputs": {}},
    {"name": "mlflow_list_runs", "inputs": {"experiment_id": "string", "limit": "integer"}},
    {"name": "mlflow_run_details", "inputs": {"run_id": "string"}},
    {"name": "mlflow_compare_runs", "inputs": {"run_ids": "array[string]", "metric": "string"}},
    {"name": "mlflow_compare_experiments", "inputs": {"experiment_ids": "array[string]", "metric": "string"}},
    {"name": "mlflow_best_run", "inputs": {"experiment_id": "string", "metric": "string"}},
    {"name": "mlflow_model_versions", "inputs": {"model_name": "string"}},
]
