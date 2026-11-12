"""Generate a small synthetic classification dataset without external dependencies."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

FEATURES = ("temperature_c", "vibration_mm_s", "load_pct")


def generate(output: Path, *, seed: int = 42, rows: int = 600) -> Path:
    if rows < 20:
        raise ValueError("rows must be at least 20")
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((*FEATURES, "anomaly"))
        for _ in range(rows):
            load = rng.uniform(10, 100)
            temperature = 31 + 0.42 * load + rng.gauss(0, 7)
            vibration = 0.8 + 0.035 * load + rng.gauss(0, 0.65)
            score = 0.075 * (temperature - 58) + 0.7 * (vibration - 2.7)
            score += rng.gauss(0, 0.9)
            writer.writerow((f"{temperature:.3f}", f"{vibration:.3f}", f"{load:.3f}", int(score > 0)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/equipment.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=600)
    args = parser.parse_args()
    print(generate(args.output, seed=args.seed, rows=args.rows))


if __name__ == "__main__":
    main()

