#!/usr/bin/env python3
"""Plot per-seed and cross-seed validation learning curves from run metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read_history(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "metrics.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _metric(record: dict[str, object], key: str) -> float:
    validation = record.get("validation")
    if not isinstance(validation, dict) or key not in validation:
        raise ValueError(f"missing validation.{key} in epoch {record.get('epoch')}")
    return float(validation[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="directory containing <run-name>/metrics.jsonl or flat *.metrics.jsonl files")
    parser.add_argument("--prefix", action="append", required=True, help="run-name prefix, e.g. direct-long-lr1e-4")
    parser.add_argument("--metric", choices=("token_accuracy", "loss"), default="token_accuracy")
    parser.add_argument("--output", type=Path, default=Path("runs/original/figures/validation_learning_curves.png"))
    args = parser.parse_args()

    histories: dict[str, list[dict[str, object]]] = {}
    for prefix in args.prefix:
        for run_dir in sorted(args.runs_dir.glob(f"{prefix}-seed*")):
            if (run_dir / "metrics.jsonl").is_file():
                histories[run_dir.name] = _read_history(run_dir)
        for metrics_path in sorted(args.runs_dir.glob(f"{prefix}-seed*.metrics.jsonl")):
            name = metrics_path.name.removesuffix(".metrics.jsonl")
            with metrics_path.open(encoding="utf-8") as stream:
                histories[name] = [json.loads(line) for line in stream if line.strip()]
    if not histories:
        raise SystemExit("no metrics.jsonl files matched the requested prefixes")

    grouped: dict[str, list[tuple[str, list[dict[str, object]]]]] = defaultdict(list)
    for name, history in histories.items():
        group = next((prefix for prefix in args.prefix if name.startswith(prefix)), name)
        grouped[group].append((name, history))

    figure, axes = plt.subplots(len(grouped), 1, figsize=(9, 3.8 * len(grouped)), squeeze=False)
    ylabel = "validation token accuracy" if args.metric == "token_accuracy" else "validation loss"
    for axis, (group, seed_histories) in zip(axes[:, 0], sorted(grouped.items()), strict=True):
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for name, history in seed_histories:
            epochs = [int(record["epoch"]) for record in history]
            values = [_metric(record, args.metric) for record in history]
            axis.plot(epochs, values, alpha=0.45, linewidth=1.3, label=name)
            for epoch, value in zip(epochs, values, strict=True):
                by_epoch[epoch].append(value)
        mean_epochs = sorted(by_epoch)
        means = [sum(by_epoch[epoch]) / len(by_epoch[epoch]) for epoch in mean_epochs]
        axis.plot(mean_epochs, means, color="black", linewidth=2.5, label="seed mean")
        axis.set_title(group)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
