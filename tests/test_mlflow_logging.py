"""Provenance checks before an experiment is allowed into MLflow."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from training.log_experiment import dataset_provenance


class ProvenanceTests(unittest.TestCase):
    def test_committed_clean_dataset_provides_both_references(self):
        pointer = "outs:\n- md5: " + "a" * 32 + "\n  path: equipment.csv\n"
        with patch("training.log_experiment._command", side_effect=["b" * 40, pointer, "{}"]), \
             patch("training.log_experiment.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0)):
            result = dataset_provenance(Path("data/raw/equipment.csv"))
        self.assertEqual(result["dataset_git_commit"], "b" * 40)
        self.assertEqual(result["dataset_dvc_md5"], "a" * 32)

    def test_dirty_dvc_pointer_is_rejected_before_logging(self):
        with patch("training.log_experiment._command", return_value="b" * 40), \
             patch("training.log_experiment.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(ValueError, "pointer differs"):
                dataset_provenance(Path("data/raw/equipment.csv"))

    def test_unapproved_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "approved DVC dataset"):
            dataset_provenance(Path("private/data.csv"))


if __name__ == "__main__":
    unittest.main()
