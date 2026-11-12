"""MLflow tool behavior with a client double; no server required in CI."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from agent.tools.mlflow_tools import MlflowToolError, MlflowTools


def run(run_id, f1=None, status="FINISHED", start=1):
    metrics = {} if f1 is None else {"f1": f1}
    return NS(info=NS(run_id=run_id, experiment_id="1", status=status, start_time=start),
              data=NS(metrics=metrics, params={"seed": "42"},
                      tags={"dataset_git_commit": "abc", "dataset_dvc_md5": "def"}))


class MlflowToolTests(unittest.TestCase):
    def test_list_experiments_uses_client(self):
        client = Mock()
        client.search_experiments.return_value = [NS(experiment_id="1", name="baseline",
                                                    lifecycle_stage="active")]
        self.assertEqual(MlflowTools(client).list_experiments()["experiments"][0]["name"],
                         "baseline")

    def test_list_runs_returns_recorded_metrics_and_provenance(self):
        client = Mock()
        client.search_runs.return_value = [run("one", 0.8)]
        result = MlflowTools(client).list_runs("1", limit=5)
        self.assertEqual(result["runs"][0]["tags"]["dataset_git_commit"], "abc")
        client.search_runs.assert_called_once_with(["1"], max_results=5,
                                                  order_by=["attributes.start_time DESC"])

    def test_best_run_excludes_failed_and_missing_metrics(self):
        client = Mock()
        client.search_runs.return_value = [run("failed", 0.99, "FAILED"), run("missing"),
                                           run("best", 0.84), run("other", 0.81)]
        result = MlflowTools(client).best_run("1")
        self.assertEqual(result["candidate"]["run_id"], "best")
        self.assertIsNone(result["production_ready"])

    def test_no_metric_does_not_invent_a_candidate(self):
        client = Mock()
        client.search_runs.return_value = [run("one")]
        self.assertIsNone(MlflowTools(client).best_run("1")["candidate"])

    def test_run_details_includes_artifacts(self):
        client = Mock()
        client.get_run.return_value = run("one", 0.8)
        client.list_artifacts.side_effect = [
            [NS(path="model", is_dir=True, file_size=None)],
            [NS(path="model/model.json", is_dir=False, file_size=500)],
        ]
        result = MlflowTools(client).run_details("one")
        self.assertEqual(result["artifacts"][1]["path"], "model/model.json")

    def test_compare_rejects_duplicate_runs_and_bad_metric(self):
        client = Mock()
        with self.assertRaises(ValueError):
            MlflowTools(client).compare_runs(["one", "one"])
        with self.assertRaises(ValueError):
            MlflowTools(client).best_run("1", metric="f1 DESC; DROP")
        client.assert_not_called()

    def test_service_failure_does_not_produce_fake_data(self):
        client = Mock()
        client.search_runs.side_effect = TimeoutError()
        with self.assertRaises(MlflowToolError):
            MlflowTools(client).list_runs("1")

    def test_model_registry_versions_when_available(self):
        client = Mock()
        client.search_model_versions.return_value = [NS(version="1", run_id="one",
                                                       status="READY", source="runs:/one/model")]
        self.assertEqual(MlflowTools(client).model_versions("baseline")["versions"][0]["run_id"],
                         "one")

    def test_compare_experiments_keeps_selections_separate(self):
        client = Mock()
        client.search_runs.side_effect = [[run("one", 0.8)], [run("two", 0.9)]]
        result = MlflowTools(client).compare_experiments(["1", "2"])
        self.assertEqual([entry["selection"]["candidate"]["run_id"]
                          for entry in result["experiments"]], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
