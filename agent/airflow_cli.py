"""Inspect or trigger the approved Airflow training DAG."""

import argparse
import json
import os

from agent.tools.airflow_tools import AirflowTools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["dags", "runs", "status", "tasks", "failed", "log", "trigger"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minimum-f1", type=float)
    parser.add_argument("--inject-failure", choices=["validation", "quality_gate"])
    args = parser.parse_args()
    tools = AirflowTools(os.environ.get("AIRFLOW_API_URL", "http://127.0.0.1:8080"),
                         os.environ.get("AIRFLOW_API_TOKEN", ""))
    if args.operation == "dags":
        result = tools.list_dags()
    elif args.operation == "runs":
        result = tools.recent_runs()
    elif args.operation == "trigger":
        conf = {key: value for key, value in {
            "seed": args.seed, "minimum_f1": args.minimum_f1,
            "inject_failure": args.inject_failure}.items() if value is not None}
        result = tools.trigger_training(confirmed=args.confirm, conf=conf)
    elif len(args.values) == 1 and args.operation in ("status", "tasks", "failed"):
        result = {"status": tools.run_status, "tasks": tools.task_instances,
                  "failed": tools.failed_tasks}[args.operation](args.values[0])
    elif len(args.values) == 2 and args.operation == "log":
        result = tools.task_log(*args.values)
    else:
        parser.error("wrong arguments")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
