"""Train and evaluate a deterministic logistic baseline with the Python standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

from training.generate_data import FEATURES


def load_dataset(path: Path) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [*FEATURES, "anomaly"]:
            raise ValueError(f"expected columns {[*FEATURES, 'anomaly']}")
        for row in reader:
            values = [float(row[name]) for name in FEATURES]
            label = int(row["anomaly"])
            if not all(math.isfinite(value) for value in values) or label not in (0, 1):
                raise ValueError("dataset contains invalid values")
            features.append(values)
            labels.append(label)
    if len(features) < 20 or set(labels) != {0, 1}:
        raise ValueError("dataset needs at least 20 rows and both classes")
    return features, labels


def _split(labels: list[int], seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * 0.2))
        train.extend(indices[n_test:])
        test.extend(indices[:n_test])
    if set(labels[i] for i in train) != {0, 1}:
        raise ValueError("each class needs at least two rows for a train/test split")
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def train(input_path: Path, model_path: Path, metrics_path: Path, *, seed: int = 42) -> dict:
    features, labels = load_dataset(input_path)
    train_ids, test_ids = _split(labels, seed)
    means = [sum(features[i][j] for i in train_ids) / len(train_ids) for j in range(3)]
    scales = [
        max(1e-12, math.sqrt(sum((features[i][j] - means[j]) ** 2 for i in train_ids) / len(train_ids)))
        for j in range(3)
    ]
    weights = [0.0] * 3
    bias = 0.0
    for _ in range(700):
        gradient = [0.0] * 3
        bias_gradient = 0.0
        for i in train_ids:
            scaled = [(features[i][j] - means[j]) / scales[j] for j in range(3)]
            z = bias + sum(weights[j] * scaled[j] for j in range(3))
            probability = 1 / (1 + math.exp(-max(-35, min(35, z))))
            error = probability - labels[i]
            bias_gradient += error
            for j in range(3):
                gradient[j] += error * scaled[j]
        rate = 0.12 / len(train_ids)
        bias -= rate * bias_gradient
        for j in range(3):
            weights[j] -= rate * (gradient[j] + 0.01 * weights[j])

    tp = fp = tn = fn = 0
    for i in test_ids:
        z = bias + sum(weights[j] * (features[i][j] - means[j]) / scales[j] for j in range(3))
        predicted = int(z >= 0)
        actual = labels[i]
        tp += predicted == 1 and actual == 1
        fp += predicted == 1 and actual == 0
        tn += predicted == 0 and actual == 0
        fn += predicted == 0 and actual == 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "accuracy": (tp + tn) / len(test_ids),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "seed": seed,
        "dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "training_git_head": _git_commit(),
    }
    model = {
        "type": "logistic_regression_baseline",
        "features": FEATURES,
        "means": means,
        "scales": scales,
        "weights": weights,
        "bias": bias,
        "threshold": 0.5,
        "dataset_sha256": metrics["dataset_sha256"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.model, args.metrics, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

