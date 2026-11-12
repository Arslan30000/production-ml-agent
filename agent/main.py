"""Natural-language MLOps agent and deterministic four-service audit CLI."""

import argparse
import json
import logging
import os
from pathlib import Path

from agent.audit import production_audit
from agent.orchestrator import Agent, build_tools
from agent.provider import ChatProvider, ProviderError
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DvcTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MlflowTools


class UnavailableAirflow:
    def __getattr__(self, name):
        def unavailable(*args, **kwargs):
            raise RuntimeError("Airflow API token missing")
        return unavailable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Natural-language request")
    parser.add_argument("--audit", action="store_true", help="Run the four-service evidence check without an LLM")
    parser.add_argument("--experiment", default="1", help="MLflow experiment ID for --audit")
    parser.add_argument("--approve-training", action="store_true")
    parser.add_argument("--approve-restart", action="store_true")
    parser.add_argument("--trace", type=Path, help="Local JSONL tool metadata (no credentials)")
    args = parser.parse_args()
    if not args.audit and not args.question:
        parser.error("supply a question or --audit")
    logger = logging.getLogger("production_ml_agent")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.trace, encoding="utf-8") if args.trace else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        from mlflow.tracking import MlflowClient
        dvc = DvcTools(Path.cwd())
        mlflow = MlflowTools(MlflowClient(tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")))
        token = os.environ.get("AIRFLOW_API_TOKEN", "")
        airflow = AirflowTools(os.environ.get("AIRFLOW_API_URL", "http://127.0.0.1:8080"),
                               token) if token else UnavailableAirflow()
        kubernetes = KubernetesTools()
        if args.audit:
            result = production_audit(dvc, mlflow, airflow, kubernetes,
                                      experiment_id=args.experiment)
        else:
            approvals = set()
            if args.approve_training:
                approvals.add("training")
            if args.approve_restart:
                approvals.add("restart")
            provider = ChatProvider()
            agent = Agent(provider, build_tools(dvc, mlflow, airflow, kubernetes),
                          approved_actions=approvals)
            result = agent.run(args.question)
        print(json.dumps(result, indent=2, default=str))
    except (ValueError, ProviderError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
