"""Inspect the approved Kubernetes namespace."""

import argparse
import json

from agent.tools.kubernetes_tools import KubernetesTools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["workloads", "services", "diagnose", "version", "logs", "restart"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    tools = KubernetesTools()
    if args.operation == "workloads":
        result = tools.workloads()
    elif args.operation == "services":
        result = tools.services()
    elif args.operation == "diagnose":
        result = tools.diagnose()
    elif args.operation == "version":
        result = tools.deployed_version()
    elif args.operation == "logs" and args.name:
        result = tools.pod_logs(args.name)
    elif args.operation == "restart" and args.name:
        result = tools.restart_deployment(args.name, confirmed=args.confirm)
    else:
        parser.error("Pod or deployment name required")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
