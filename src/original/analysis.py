"""Cross-seed aggregation for original-plan swap-length experiments."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


RAW_FIELDS = (
    "architecture",
    "position_encoding",
    "seed",
    "split",
    "n_swaps",
    "metric",
    "value",
    "n_samples",
)


def flatten_result(result: dict[str, object]) -> list[dict[str, object]]:
    """Convert one run JSON into model×seed×split×swap×metric rows."""

    config = result["config"]
    assert isinstance(config, dict)
    splits = result["splits"]
    assert isinstance(splits, dict)
    shared = {
        "architecture": config["architecture"],
        "position_encoding": config["position_encoding"],
        "seed": result["seed"],
    }
    rows: list[dict[str, object]] = []
    for split, raw_metrics in splits.items():
        assert isinstance(raw_metrics, dict)
        for metric in ("exact_match", "slot_accuracy"):
            rows.append(
                {
                    **shared,
                    "split": split,
                    "n_swaps": "ALL",
                    "metric": metric,
                    "value": float(raw_metrics[metric]),
                    "n_samples": int(raw_metrics["n_samples"]),
                }
            )
        by_swaps = raw_metrics["by_swaps"]
        assert isinstance(by_swaps, dict)
        for n_swaps, raw_swap_metrics in by_swaps.items():
            assert isinstance(raw_swap_metrics, dict)
            rows.append(
                {
                    **shared,
                    "split": split,
                    "n_swaps": int(n_swaps),
                    "metric": "exact_match",
                    "value": float(raw_swap_metrics["exact_match"]),
                    "n_samples": int(raw_swap_metrics["total"]),
                }
            )
    return rows


def summarize_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Compute mean and sample standard deviation across distinct seeds."""

    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    seeds: dict[tuple[object, ...], set[int]] = defaultdict(set)
    sample_counts: dict[tuple[object, ...], int] = defaultdict(int)
    for row in rows:
        key = (
            row["architecture"],
            row["position_encoding"],
            row["split"],
            row["n_swaps"],
            row["metric"],
        )
        grouped[key].append(float(row["value"]))
        seeds[key].add(int(row["seed"]))
        sample_counts[key] += int(row["n_samples"])

    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        architecture, position_encoding, split, n_swaps, metric = key
        summaries.append(
            {
                "architecture": architecture,
                "position_encoding": position_encoding,
                "split": split,
                "n_swaps": n_swaps,
                "metric": metric,
                "n_seeds": len(seeds[key]),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "total_evaluations": sample_counts[key],
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else list(RAW_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    results: Iterable[dict[str, object]],
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows = [row for result in results for row in flatten_result(result)]
    summaries = summarize_rows(raw_rows)
    write_csv(output_dir / "raw_long.csv", raw_rows)
    write_csv(output_dir / "summary.csv", summaries)
    return raw_rows, summaries
