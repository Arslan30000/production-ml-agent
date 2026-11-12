import json
import subprocess
import unittest
from unittest.mock import Mock

from agent.tools.kubernetes_tools import KubernetesTools, KubernetesToolError


class KubernetesToolTests(unittest.TestCase):
    def test_rejects_unauthorized_modification_without_running_kubectl(self):
        runner = Mock()
        tools = KubernetesTools(runner=runner)
        with self.assertRaises(PermissionError):
            tools.restart_deployment("equipment-api", confirmed=False)
        with self.assertRaises(PermissionError):
            tools.restart_deployment("other", confirmed=True)
        runner.assert_not_called()

    def test_workloads_stay_in_approved_namespace(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, '{"items":[]}', ""))
        KubernetesTools(runner=runner).workloads()
        self.assertEqual(runner.call_count, 2)
        for call in runner.call_args_list:
            self.assertEqual(call.args[0][:3], ["kubectl", "-n", "mlops-agent"])

    def test_diagnosis_reports_unready_pod_and_restart(self):
        pod = {"metadata": {"name": "equipment-api-abc"}, "status": {
            "phase": "Running", "containerStatuses": [{"ready": False, "restartCount": 2}]}}
        runner = Mock(side_effect=[subprocess.CompletedProcess([], 0, json.dumps({"items": [pod]}), ""),
                                   subprocess.CompletedProcess([], 0, '{"items":[]}', ""),
                                   subprocess.CompletedProcess([], 0, '{"items":[]}', "")])
        result = KubernetesTools(runner=runner).diagnose()
        self.assertEqual(result["unhealthy"][0]["name"], "equipment-api-abc")

    def test_service_failure_does_not_invent_health(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 1, "", "private credentials"))
        with self.assertRaises(KubernetesToolError) as caught:
            KubernetesTools(runner=runner).deployed_version()
        self.assertNotIn("private credentials", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
