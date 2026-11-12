"""Inspect the MLflow tracking server using the same tools the agent will call."""

from __future__ import annotations

import argparse
import json
import os

from agent.tools.mlflow_tools import MlflowToolError, MlflowTools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["experiments", "runs", "details", "best", "compare", "compare-experiments", "versions"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    from mlflow.tracking import MlflowClient

    tools = MlflowTools(MlflowClient(tracking_uri=args.tracking_uri))
    try:
        if args.operation == "experiments" and not args.values:
            result = tools.list_experiments()
        elif args.operation == "runs" and len(args.values) == 1:
            result = tools.list_runs(args.values[0], limit=args.limit)
        elif args.operation == "details" and len(args.values) == 1:
            result = tools.run_details(args.values[0])
        elif args.operation == "best" and len(args.values) == 1:
            result = tools.best_run(args.values[0], metric=args.metric, limit=args.limit)
        elif args.operation == "compare" and len(args.values) >= 2:
            result = tools.compare_runs(args.values, metric=args.metric)
        elif args.operation == "compare-experiments" and len(args.values) >= 2:
            result = tools.compare_experiments(args.values, metric=args.metric, limit=args.limit)
        elif args.operation == "versions" and len(args.values) == 1:
            result = tools.model_versions(args.values[0])
        else:
            parser.error("wrong number of values for this operation")
        print(json.dumps(result, indent=2))
    except (MlflowToolError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
