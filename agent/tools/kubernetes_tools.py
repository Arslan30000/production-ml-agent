"""Namespace-limited Kubernetes diagnostics using fixed kubectl commands."""

import json
import re
import subprocess
from typing import Callable


NAMESPACE = "mlops-agent"
NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


class KubernetesToolError(RuntimeError):
    pass


class KubernetesTools:
    def __init__(self, *, runner: Callable = subprocess.run, timeout: int = 15):
        self.runner, self.timeout = runner, timeout

    def _run(self, args: list[str], *, json_output: bool = True):
        try:
            result = self.runner(["kubectl", "-n", NAMESPACE, *args], capture_output=True,
                                 text=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KubernetesToolError("Kubernetes request unavailable or timed out") from exc
        if result.returncode:
            raise KubernetesToolError(f"kubectl failed (exit {result.returncode})")
        if not json_output:
            return result.stdout[:8000]
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise KubernetesToolError("kubectl returned invalid JSON") from exc

    def workloads(self) -> dict:
        pods = self._run(["get", "pods", "-l", "app=equipment-api", "-o", "json"])
        deployments = self._run(["get", "deployments", "-o", "json"])
        deployments["items"] = [item for item in deployments.get("items", [])
                                if item.get("metadata", {}).get("name") == "equipment-api"]
        return {"pods": pods, "deployments": deployments}

    def services(self) -> dict:
        return self._run(["get", "services", "-l", "app=equipment-api", "-o", "json"])

    def pod_logs(self, pod: str) -> dict:
        if not NAME.fullmatch(pod):
            raise ValueError("invalid Pod name")
        return {"pod": pod, "logs": self._run(["logs", pod, "-c", "api", "--tail=100"],
                                                json_output=False)}

    def diagnose(self) -> dict:
        pods = self.workloads()["pods"]["items"]
        unhealthy = []
        for pod in pods:
            statuses = pod.get("status", {}).get("containerStatuses", [])
            if pod.get("status", {}).get("phase") != "Running" or not statuses or any(
                not status.get("ready") or status.get("restartCount", 0) > 0 for status in statuses
            ):
                unhealthy.append({"name": pod["metadata"]["name"],
                                  "phase": pod.get("status", {}).get("phase"),
                                  "container_statuses": statuses})
        events = self._run(["get", "events", "--sort-by=.lastTimestamp", "-o", "json"])
        return {"unhealthy": unhealthy, "events": events.get("items", [])[-30:]}

    def deployed_version(self) -> dict:
        deployments = self.workloads()["deployments"].get("items", [])
        config = self._run(["get", "configmap", "equipment-api-config", "-o", "json"])
        return {"app_version": config.get("data", {}).get("APP_VERSION"),
                "deployments": [{"name": d["metadata"]["name"],
                                 "desired": d["spec"].get("replicas"),
                                 "available": d.get("status", {}).get("availableReplicas", 0),
                                 "images": [c["image"] for c in d["spec"]["template"]["spec"]["containers"]]}
                                for d in deployments]}

    def restart_deployment(self, name: str, *, confirmed: bool) -> dict:
        if not confirmed:
            raise PermissionError("Cluster changes require explicit confirmation")
        if name != "equipment-api":
            raise PermissionError("Deployment is not approved for restart")
        return {"result": self._run(["rollout", "restart", f"deployment/{name}"], json_output=False)}


TOOL_SCHEMAS = [
    {"name": "k8s_workloads", "inputs": {}},
    {"name": "k8s_services", "inputs": {}},
    {"name": "k8s_pod_logs", "inputs": {"pod": "string"}},
    {"name": "k8s_diagnose", "inputs": {}},
    {"name": "k8s_deployed_version", "inputs": {}},
    {"name": "k8s_restart_deployment", "inputs": {"name": "string", "confirmed": "boolean"}},
]

