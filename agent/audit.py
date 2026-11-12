"""Deterministic four-service check for the mandatory integration scenario."""

from __future__ import annotations

import uuid
import json
import logging
import time
from typing import Any


def production_audit(dvc: Any, mlflow: Any, airflow: Any, kubernetes: Any,
                     *, experiment_id: str, request_id: str | None = None) -> dict:
    request_id = request_id or uuid.uuid4().hex
    checks = (
        ("mlflow_best_run", lambda: mlflow.best_run(experiment_id, metric="f1")),
        ("dvc_current_version", dvc.current_version),
        ("k8s_deployed_version", kubernetes.deployed_version),
        ("k8s_diagnose", kubernetes.diagnose),
        ("airflow_recent_runs", airflow.recent_runs),
    )
    evidence = {}
    for name, function in checks:
        started = time.monotonic()
        try:
            evidence[name] = {"ok": True, "data": function()}
        except Exception:
            evidence[name] = {"ok": False, "error": "service_unavailable"}
        finally:
            logging.getLogger("production_ml_agent").info(json.dumps({
                "request_id": request_id, "tool_name": name,
                "status": "ok" if evidence[name]["ok"] else "error",
                "duration_ms": round((time.monotonic() - started) * 1000, 2)}))

    facts, unknowns = [], []
    best = evidence["mlflow_best_run"]
    dvc_result = evidence["dvc_current_version"]
    deployed = evidence["k8s_deployed_version"]
    health = evidence["k8s_diagnose"]
    workflows = evidence["airflow_recent_runs"]
    candidate = best["data"].get("candidate") if best["ok"] else None
    if candidate:
        facts.append(f"Best recorded F1 candidate: run {candidate['run_id']} "
                     f"(F1 {candidate['metrics']['f1']:.3f}).")
        tags = candidate.get("tags", {})
        if tags.get("dataset_git_commit") and tags.get("dataset_dvc_md5"):
            facts.append("Candidate dataset reference: Git " + tags["dataset_git_commit"] +
                         ", DVC MD5 " + tags["dataset_dvc_md5"] + ".")
        else:
            unknowns.append("The selected run has no complete Git/DVC dataset reference.")
    else:
        unknowns.append("No verified finished MLflow candidate with F1 was found.")
    if dvc_result["ok"]:
        facts.append("Current DVC checkout: Git " + dvc_result["data"]["git_commit"] + ".")
    else:
        unknowns.append("Current DVC checkout could not be verified.")
    if deployed["ok"]:
        for item in deployed["data"].get("deployments", []):
            facts.append(f"Deployment {item['name']}: {item['available']}/{item['desired']} "
                         "replicas available; image " + ", ".join(item["images"]) + ".")
        if not deployed["data"].get("deployments"):
            unknowns.append("No approved deployment was returned.")
    else:
        unknowns.append("Kubernetes deployment version and replicas are unknown.")
    if health["ok"]:
        unhealthy = health["data"].get("unhealthy", [])
        facts.append(f"Kubernetes diagnosis reports {len(unhealthy)} unhealthy Pod(s).")
    else:
        unknowns.append("Kubernetes Pod health is unknown.")
    if workflows["ok"]:
        runs = workflows["data"].get("dag_runs", [])
        if runs:
            latest = runs[0]
            failure = (latest.get("conf") or {}).get("inject_failure")
            suffix = f" (injected {failure} test)" if failure else ""
            facts.append(f"Latest Airflow training run {latest['dag_run_id']}: "
                         f"{latest['state']}{suffix}.")
            normal = next((run for run in runs if not (run.get("conf") or {}).get(
                "inject_failure")), None)
            if normal and normal is not latest:
                facts.append(f"Latest non-injected run {normal['dag_run_id']}: "
                             f"{normal['state']}.")
        else:
            unknowns.append("No recent Airflow runs were returned.")
    else:
        unknowns.append("Airflow training status is unknown.")

    # The container image stores a model JSON, but its MLflow run ID is not in the
    # Deployment. A successful rollout cannot prove it contains the selected run.
    unknowns.append("The deployed image is not linked to the selected MLflow run ID.")
    recommendation = ("Review any failed run and its failed task before promotion. "
                      "The injected failures are tests, not evidence of model degradation. "
                      "Retraining needs a freshness policy, real quality failure, or degradation evidence; "
                      "it is not started here.")
    return {"request_id": request_id, "facts": facts, "unknowns": unknowns,
            "recommendation": recommendation, "evidence": evidence,
            "retraining_triggered": False}

