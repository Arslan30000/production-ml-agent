"""Approved Airflow 3 public API operations with bearer authentication."""

from __future__ import annotations

import json
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class AirflowToolError(RuntimeError):
    pass


IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:~+-]{1,200}$")
APPROVED_DAG = "equipment_training"


def _part(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid Airflow identifier")
    return quote(value, safe="")


class AirflowTools:
    def __init__(self, base_url: str, token: str, *, transport: Callable | None = None,
                 timeout: int = 15):
        if not base_url.startswith(("http://", "https://")) or not token:
            raise ValueError("Airflow URL and token are required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.transport = transport or self._http

    def _http(self, method: str, path: str, body: dict | None = None):
        payload = json.dumps(body).encode() if body is not None else None
        request = Request(self.base_url + "/api/v2" + path, data=payload, method=method,
                          headers={"Authorization": "Bearer " + self.token,
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"text": data[:4000]}
        except (HTTPError, URLError, TimeoutError) as exc:
            # Never include response bodies or the bearer token in agent output.
            raise AirflowToolError("Airflow request failed or timed out") from exc

    def list_dags(self) -> dict:
        return self.transport("GET", "/dags?limit=100")

    def run_status(self, run_id: str) -> dict:
        return self.transport("GET", f"/dags/{APPROVED_DAG}/dagRuns/{_part(run_id)}")

    def recent_runs(self) -> dict:
        return self.transport("GET", f"/dags/{APPROVED_DAG}/dagRuns?limit=20&order_by=-run_after")

    def trigger_training(self, *, confirmed: bool, conf: dict | None = None) -> dict:
        if not confirmed:
            raise PermissionError("Triggering training requires explicit confirmation")
        if conf and (set(conf) - {"seed", "minimum_f1", "inject_failure"}):
            raise ValueError("unapproved DAG configuration")
        if conf:
            seed = conf.get("seed", 42)
            threshold = conf.get("minimum_f1", 0.75)
            failure = conf.get("inject_failure")
            if type(seed) is not int or not 0 <= seed <= 100000:
                raise ValueError("seed must be an integer from 0 to 100000")
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                raise ValueError("minimum_f1 must be between 0 and 1")
            if failure is not None and failure not in ("validation", "quality_gate"):
                raise ValueError("invalid failure injection")
        return self.transport("POST", f"/dags/{APPROVED_DAG}/dagRuns",
                              {"logical_date": None, "conf": conf or {}})

    def task_instances(self, run_id: str) -> dict:
        return self.transport("GET", f"/dags/{APPROVED_DAG}/dagRuns/{_part(run_id)}/taskInstances")

    def failed_tasks(self, run_id: str) -> dict:
        tasks = self.task_instances(run_id).get("task_instances", [])
        return {"run_id": run_id, "failed": [task for task in tasks if task.get("state") == "failed"]}

    def task_log(self, run_id: str, task_id: str, try_number: int = 1) -> dict:
        if not 1 <= try_number <= 20:
            raise ValueError("invalid try number")
        return self.transport("GET", f"/dags/{APPROVED_DAG}/dagRuns/{_part(run_id)}/"
                              f"taskInstances/{_part(task_id)}/logs/{try_number}")


TOOL_SCHEMAS = [
    {"name": "airflow_list_dags", "inputs": {}},
    {"name": "airflow_recent_runs", "inputs": {}},
    {"name": "airflow_run_status", "inputs": {"run_id": "string"}},
    {"name": "airflow_trigger_training", "inputs": {"confirmed": "boolean", "conf": "object"}},
    {"name": "airflow_failed_tasks", "inputs": {"run_id": "string"}},
    {"name": "airflow_task_log", "inputs": {"run_id": "string", "task_id": "string"}},
]
