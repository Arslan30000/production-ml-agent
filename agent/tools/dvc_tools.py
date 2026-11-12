"""Approved, structured DVC inspection tools for the agent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable


class ToolError(RuntimeError):
    """A tool could not obtain trustworthy information."""


Runner = Callable[..., subprocess.CompletedProcess[str]]
REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^-]*$")


class DvcTools:
    """Run only fixed, read-only DVC/Git commands in one configured repository."""

    def __init__(self, repo: Path, *, runner: Runner = subprocess.run, timeout: int = 20):
        self.repo = repo.resolve()
        self.runner = runner
        self.timeout = timeout
        if not self.repo.is_dir():
            raise ValueError("repository directory does not exist")

    def _run(self, argv: list[str]) -> str:
        try:
            result = self.runner(
                argv, cwd=self.repo, capture_output=True, text=True,
                check=False, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"{argv[0]} request timed out") from exc
        except OSError as exc:
            raise ToolError(f"{argv[0]} is unavailable") from exc
        if result.returncode:
            # CLI stderr can contain paths or credentials. Do not expose it to the agent.
            raise ToolError(f"{argv[0]} command failed (exit {result.returncode})")
        return result.stdout.strip()

    def _json(self, argv: list[str]):
        output = self._run(argv)
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise ToolError(f"{argv[0]} returned invalid JSON") from exc

    def _commit(self, ref: str) -> str:
        if not REF.fullmatch(ref) or ref.startswith("-") or ".." in ref:
            raise ValueError("invalid Git reference")
        commit = self._run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"])
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
            raise ToolError("Git returned an invalid commit ID")
        return commit

    def list_datasets(self) -> dict:
        """List DVC-tracked entries under the approved data directory."""
        return {"datasets": self._json(["dvc", "list", "--dvc-only", "--json", "-R", ".", "data"])}

    def current_version(self) -> dict:
        """Return the recorded Git revision, DVC hashes, and workspace state."""
        return {
            "git_commit": self._commit("HEAD"),
            "tracked": self._json(
                ["dvc", "list", "--dvc-only", "--json", "--show-hash", "-R", ".", "data"]
            ),
            "workspace_changes": self._json(["dvc", "status", "--json"]),
        }

    def compare_versions(self, old_ref: str, new_ref: str) -> dict:
        """Compare DVC-tracked data at two existing Git commits."""
        old = self._commit(old_ref)
        new = self._commit(new_ref)
        return {"old_commit": old, "new_commit": new,
                "changes": self._json(["dvc", "diff", "--json", "--show-hash", old, new])}

    def detect_changes(self) -> dict:
        """Inspect current tracked data without changing it."""
        return {"changes": self._json(["dvc", "status", "--json"])}

    def reproduction_plan(self, ref: str) -> dict:
        """Give a verified revision and commands for an isolated checkout."""
        commit = self._commit(ref)
        return {
            "git_commit": commit,
            "commands": [
                f"git worktree add --detach ../dataset-review {commit}",
                "cd ../dataset-review",
                "dvc pull",
                "dvc checkout",
            ],
            "note": "Requires access to the configured DVC remote; no checkout was executed.",
        }


TOOL_SCHEMAS = [
    {"name": "dvc_list_datasets", "description": "List tracked datasets", "inputs": {}},
    {"name": "dvc_current_version", "description": "Get current Git and DVC data revisions", "inputs": {}},
    {"name": "dvc_compare_versions", "description": "Compare two Git revisions of data",
     "inputs": {"old_ref": "string", "new_ref": "string"}},
    {"name": "dvc_detect_changes", "description": "Check uncommitted data changes", "inputs": {}},
    {"name": "dvc_reproduction_plan", "description": "Explain isolated dataset reproduction",
     "inputs": {"ref": "string"}},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["list", "version", "status", "compare", "reproduce"])
    parser.add_argument("references", nargs="*")
    args = parser.parse_args()
    tools = DvcTools(Path.cwd())
    try:
        if args.operation == "list" and not args.references:
            result = tools.list_datasets()
        elif args.operation == "version" and not args.references:
            result = tools.current_version()
        elif args.operation == "status" and not args.references:
            result = tools.detect_changes()
        elif args.operation == "compare" and len(args.references) == 2:
            result = tools.compare_versions(*args.references)
        elif args.operation == "reproduce" and len(args.references) == 1:
            result = tools.reproduction_plan(args.references[0])
        else:
            parser.error("wrong number of references for this operation")
        print(json.dumps(result, indent=2))
    except (ToolError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
