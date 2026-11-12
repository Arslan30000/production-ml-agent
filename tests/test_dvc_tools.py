"""DVC tool tests with controlled subprocess responses."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent.tools.dvc_tools import DvcTools, ToolError

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


class DvcToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_list_uses_approved_dvc_command(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, '[{"path":"data.csv"}]', ""))
        result = DvcTools(self.root, runner=runner).list_datasets()
        self.assertEqual(result["datasets"][0]["path"], "data.csv")
        self.assertEqual(runner.call_args.args[0],
                         ["dvc", "list", "--dvc-only", "--json", "-R", ".", "data"])

    def test_version_contains_git_commit_hash_and_workspace_state(self):
        runner = Mock(side_effect=[
            subprocess.CompletedProcess([], 0, COMMIT_A + "\n", ""),
            subprocess.CompletedProcess([], 0, '[{"path":"data.csv","md5":"123"}]', ""),
            subprocess.CompletedProcess([], 0, "{}", ""),
        ])
        result = DvcTools(self.root, runner=runner).current_version()
        self.assertEqual(result["git_commit"], COMMIT_A)
        self.assertEqual(result["tracked"][0]["md5"], "123")
        self.assertEqual(result["workspace_changes"], {})

    def test_compare_resolves_both_refs_before_dvc_diff(self):
        runner = Mock(side_effect=[
            subprocess.CompletedProcess([], 0, COMMIT_A, ""),
            subprocess.CompletedProcess([], 0, COMMIT_B, ""),
            subprocess.CompletedProcess([], 0, '{"modified":["data.csv"]}', ""),
        ])
        result = DvcTools(self.root, runner=runner).compare_versions("v1", "HEAD")
        self.assertEqual(result["changes"]["modified"], ["data.csv"])
        self.assertEqual(runner.call_args.args[0][-2:], [COMMIT_A, COMMIT_B])

    def test_rejects_invalid_ref_before_executing_any_command(self):
        runner = Mock()
        with self.assertRaises(ValueError):
            DvcTools(self.root, runner=runner).reproduction_plan("--help")
        runner.assert_not_called()

    def test_reproduction_plan_does_not_modify_worktree(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, COMMIT_A, ""))
        result = DvcTools(self.root, runner=runner).reproduction_plan("v1")
        self.assertIn(COMMIT_A, result["commands"][0])
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[0][0], "git")

    def test_timeout_and_command_failure_do_not_invent_results(self):
        timeout = Mock(side_effect=subprocess.TimeoutExpired("dvc", 20))
        with self.assertRaisesRegex(ToolError, "timed out"):
            DvcTools(self.root, runner=timeout).detect_changes()
        failure = Mock(return_value=subprocess.CompletedProcess([], 1, "", "secret path"))
        with self.assertRaisesRegex(ToolError, "command failed") as context:
            DvcTools(self.root, runner=failure).detect_changes()
        self.assertNotIn("secret path", str(context.exception))

    def test_invalid_json_is_reported_as_error(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, "not json", ""))
        with self.assertRaisesRegex(ToolError, "invalid JSON"):
            DvcTools(self.root, runner=runner).list_datasets()


if __name__ == "__main__":
    unittest.main()

