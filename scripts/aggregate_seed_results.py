#!/usr/bin/env python3
"""Create per-seed and seed-mean CSVs from saved original-plan result JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def _experiment_name(run_name: str) -> str:
    return re.sub(r"-seed\d+(?:-adaptive)?$", "", run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.results_dir.glob("*.json"))]
    rows: list[dict[str, object]] = []
    for payload in payloads:
        run_name = str(payload["run_name"])
        config = payload["config"]
        for split, metrics in payload["splits"].items():
            rows.append({
                "experiment": _experiment_name(run_name),
                "architecture": config["architecture"],
                "seed": int(payload["seed"]),
                "split": split,
                "slot_accuracy": float(metrics["slot_accuracy"]),
                "exact_match": float(metrics["exact_match"]),
            })
    if not rows:
        raise SystemExit("no result JSON files found")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "seed_results.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["experiment"]), int(row["seed"]), str(row["split"])) ))

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["architecture"]), str(row["split"]))].append(row)
    means: list[dict[str, object]] = []
    for (experiment, architecture, split), group in sorted(grouped.items()):
        means.append({
            "experiment": experiment,
            "architecture": architecture,
            "split": split,
            "n_seeds": len(group),
            "slot_accuracy_mean": sum(float(row["slot_accuracy"]) for row in group) / len(group),
            "slot_accuracy_std": (sum((float(row["slot_accuracy"]) - sum(float(item["slot_accuracy"]) for item in group) / len(group)) ** 2 for row in group) / len(group)) ** 0.5,
            "exact_match_mean": sum(float(row["exact_match"]) for row in group) / len(group),
            "exact_match_std": (sum((float(row["exact_match"]) - sum(float(item["exact_match"]) for item in group) / len(group)) ** 2 for row in group) / len(group)) ** 0.5,
        })
    mean_path = output_dir / "seed_mean_summary.csv"
    with mean_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    print(seed_path)
    print(mean_path)


if __name__ == "__main__":
    main()
