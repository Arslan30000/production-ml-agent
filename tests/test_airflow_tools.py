import unittest
from unittest.mock import Mock

from agent.tools.airflow_tools import AirflowTools


class AirflowToolTests(unittest.TestCase):
    def setUp(self):
        self.transport = Mock(return_value={"dag_runs": []})
        self.tools = AirflowTools("http://127.0.0.1:8080", "test-token",
                                  transport=self.transport)

    def test_trigger_requires_confirmation(self):
        with self.assertRaises(PermissionError):
            self.tools.trigger_training(confirmed=False)
        self.transport.assert_not_called()

    def test_trigger_uses_only_approved_dag_and_conf(self):
        self.tools.trigger_training(confirmed=True, conf={"seed": 42})
        self.transport.assert_called_once_with("POST", "/dags/equipment_training/dagRuns",
                                               {"logical_date": None, "conf": {"seed": 42}})
        with self.assertRaises(ValueError):
            self.tools.trigger_training(confirmed=True, conf={"shell": "rm"})

    def test_trigger_rejects_invalid_configuration_before_request(self):
        for conf in ({"seed": "42"}, {"minimum_f1": 1.5},
                     {"inject_failure": "shell"}):
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                self.tools.trigger_training(confirmed=True, conf=conf)
        self.transport.assert_not_called()

    def test_failed_tasks_are_based_on_api_result(self):
        self.transport.return_value = {"task_instances": [
            {"task_id": "validation", "state": "failed"},
            {"task_id": "training", "state": "success"}]}
        self.assertEqual(self.tools.failed_tasks("manual__1")["failed"][0]["task_id"],
                         "validation")

    def test_rejects_path_injection(self):
        with self.assertRaises(ValueError):
            self.tools.task_log("../other", "validation")
        self.transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
