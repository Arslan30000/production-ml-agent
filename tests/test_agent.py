import json
import logging
import unittest
from io import StringIO
from unittest.mock import Mock

from agent.audit import production_audit
from agent.orchestrator import Agent, Tool, build_tools


def call(name, args=None, call_id="1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {})}}


class StubProvider:
    def __init__(self, *messages):
        self.messages = list(messages)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(messages[:])
        return self.messages.pop(0)


class AgentTests(unittest.TestCase):
    def test_selects_and_executes_all_four_service_tools(self):
        provider = StubProvider(
            {"tool_calls": [call("mlflow_best_run", {"experiment_id": "1"}, "a"),
                            call("dvc_current_version", call_id="b"),
                            call("k8s_deployed_version", call_id="c"),
                            call("airflow_recent_runs", call_id="d")]},
            {"content": "The selected run is a metric candidate; deployment linkage is unknown."})
        dvc, mlflow, airflow, kube = Mock(), Mock(), Mock(), Mock()
        dvc.current_version.return_value = {"git_commit": "abc"}
        mlflow.best_run.return_value = {"candidate": {"run_id": "r1"}}
        airflow.recent_runs.return_value = {"dag_runs": []}
        kube.deployed_version.return_value = {"deployments": []}
        result = Agent(provider, build_tools(dvc, mlflow, airflow, kube)).run(
            "Find the best model, its dataset, deployment, and workflow")
        self.assertEqual([e["tool"] for e in result["evidence"]], [
            "mlflow_best_run", "dvc_current_version", "k8s_deployed_version", "airflow_recent_runs"])
        self.assertTrue(all(e["result"]["ok"] for e in result["evidence"]))
        self.assertFalse(airflow.trigger_training.called)

    def test_unavailable_tool_cannot_produce_fake_operational_result(self):
        provider = StubProvider({"tool_calls": [call("k8s_deployed_version")]},
                                {"content": "Deployment is healthy with version 99."})
        tools = [Tool("k8s_deployed_version", "inspect", Mock(side_effect=TimeoutError()), {})]
        result = Agent(provider, tools).run("What is deployed?")
        self.assertFalse(result["complete"])
        self.assertIn("unknown", result["answer"])
        self.assertNotIn("version 99", result["answer"])
        self.assertEqual(result["evidence"][0]["result"]["error"], "service_unavailable")

    def test_model_prose_cannot_invent_version_after_a_successful_tool(self):
        provider = StubProvider({"tool_calls": [call("dvc_current_version")]},
                                {"content": "The deployed version is 99."})
        result = Agent(provider, [Tool("dvc_current_version", "inspect",
                                       lambda: {"git_commit": "abc", "workspace_changes": {}}, {})]
                       ).run("What is current?")
        self.assertIn("Git abc", result["answer"])
        self.assertNotIn("version is 99", result["answer"])

    def test_write_requires_explicit_approval_even_when_model_requests_it(self):
        trigger = Mock(return_value={"dag_run_id": "r1"})
        tool = Tool("airflow_trigger_training", "trigger", trigger, {}, write_action="training")
        provider = StubProvider({"tool_calls": [call("airflow_trigger_training")]},
                                {"content": "Training started."})
        result = Agent(provider, [tool]).run("Should we retrain?")
        self.assertFalse(result["complete"])
        self.assertEqual(result["evidence"][0]["result"]["error"], "approval_required")
        trigger.assert_not_called()

    def test_invalid_tool_arguments_never_execute(self):
        function = Mock()
        tool = Tool("inspect", "inspect", function, {"name": {"type": "string"}}, ("name",))
        result = Agent(StubProvider({"tool_calls": [call("inspect", {"name": 3})]},
                                    {"content": "done"}), [tool]).run("Inspect")
        self.assertEqual(result["evidence"][0]["result"]["error"], "invalid_request")
        function.assert_not_called()

    def test_structured_log_has_no_secret_or_tool_output(self):
        stream = StringIO()
        logger = logging.getLogger("production_ml_agent")
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            result = Agent(StubProvider({"tool_calls": [call("inspect")]},
                                        {"content": "done"}),
                           [Tool("inspect", "inspect", lambda: {"log": "Bearer secret123"}, {})]
                           ).run("Inspect", request_id="request-1")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        entry = json.loads(stream.getvalue().strip())
        self.assertEqual(entry["request_id"], "request-1")
        self.assertEqual(entry["tool_name"], "inspect")
        self.assertEqual(entry["status"], "ok")
        self.assertIn("duration_ms", entry)
        self.assertNotIn("secret123", stream.getvalue())
        self.assertNotIn("secret123", str(result))

    def test_audit_correlates_recorded_provenance_without_automatic_retraining(self):
        dvc, mlflow, airflow, kube = Mock(), Mock(), Mock(), Mock()
        mlflow.best_run.return_value = {"candidate": {
            "run_id": "r1", "metrics": {"f1": 0.8},
            "tags": {"dataset_git_commit": "abc", "dataset_dvc_md5": "def"}}}
        dvc.current_version.return_value = {"git_commit": "abc"}
        kube.deployed_version.return_value = {"deployments": [{
            "name": "equipment-api", "available": 2, "desired": 2, "images": ["v1"]}]}
        kube.diagnose.return_value = {"unhealthy": []}
        airflow.recent_runs.return_value = {"dag_runs": [{
            "dag_run_id": "run1", "state": "success", "conf": {}}]}
        result = production_audit(dvc, mlflow, airflow, kube, experiment_id="1")
        self.assertEqual(len(result["evidence"]), 5)
        self.assertIn("DVC MD5 def", " ".join(result["facts"]))
        self.assertIn("2/2", " ".join(result["facts"]))
        self.assertFalse(result["retraining_triggered"])
        self.assertIn("not linked", " ".join(result["unknowns"]))


if __name__ == "__main__":
    unittest.main()
