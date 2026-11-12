"""Exercise the container's inference endpoint with the DVC model artifact."""

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from api import main as inference
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "install the api extra to test FastAPI")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(inference.app)

    def test_prediction_uses_tracked_model_and_reports_its_dataset(self):
        model = Path("artifacts/model.json")
        if not model.exists():
            self.skipTest("run dvc pull to restore the model artifact")
        with patch.object(inference, "MODEL_PATH", model):
            ready = self.client.get("/health/ready")
            prediction = self.client.post("/predict", json={
                "temperature_c": 80, "vibration_mm_s": 4, "load_pct": 65})
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(prediction.status_code, 200)
        result = prediction.json()
        self.assertEqual(result["model_dataset_sha256"], ready.json()["dataset_sha256"])
        self.assertGreaterEqual(result["anomaly_probability"], 0)
        self.assertLessEqual(result["anomaly_probability"], 1)

    def test_missing_model_fails_readiness_and_prediction(self):
        with patch.object(inference, "MODEL_PATH", Path("missing-model.json")):
            self.assertEqual(self.client.get("/health/ready").status_code, 503)
            self.assertEqual(self.client.post("/predict", json={
                "temperature_c": 80, "vibration_mm_s": 4, "load_pct": 65}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
