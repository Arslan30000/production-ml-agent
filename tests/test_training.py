"""Behavior tests for dataset generation and baseline training."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.generate_data import generate
from training.train import load_dataset, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_generation_is_deterministic_and_seed_changes_data(self):
        first = generate(self.root / "a.csv", seed=42)
        second = generate(self.root / "b.csv", seed=42)
        third = generate(self.root / "c.csv", seed=43)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertNotEqual(first.read_bytes(), third.read_bytes())
        features, labels = load_dataset(first)
        self.assertEqual(len(features), 600)
        self.assertEqual(set(labels), {0, 1})

    def test_training_records_real_holdout_metrics_and_provenance(self):
        source = generate(self.root / "data.csv")
        model = self.root / "model.json"
        metrics_path = self.root / "metrics.json"
        metrics = train(source, model, metrics_path)
        self.assertEqual(metrics["train_rows"] + metrics["test_rows"], 600)
        self.assertGreater(metrics["f1"], 0.7)
        self.assertLessEqual(metrics["f1"], 1.0)
        self.assertEqual(sum(metrics["confusion_matrix"].values()), metrics["test_rows"])
        self.assertEqual(json.loads(model.read_text())["dataset_sha256"], metrics["dataset_sha256"])
        self.assertEqual(json.loads(metrics_path.read_text()), metrics)

    def test_training_is_repeatable(self):
        source = generate(self.root / "data.csv")
        first = train(source, self.root / "one.json", self.root / "one_metrics.json")
        second = train(source, self.root / "two.json", self.root / "two_metrics.json")
        self.assertEqual(first, second)
        self.assertEqual((self.root / "one.json").read_bytes(), (self.root / "two.json").read_bytes())

    def test_invalid_labels_fail_before_artifacts_are_written(self):
        source = generate(self.root / "data.csv")
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[1][-1] = "7"
        with source.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaises(ValueError):
            train(source, self.root / "model.json", self.root / "metrics.json")
        self.assertFalse((self.root / "model.json").exists())


if __name__ == "__main__":
    unittest.main()
