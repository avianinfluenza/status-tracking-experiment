"""Reproducible multi-seed statistics and long-format result conversion."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Iterable, Sequence


def bootstrap_mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return (
        estimates[int(tail * (samples - 1))],
        estimates[int((1.0 - tail) * (samples - 1))],
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length >= 2")
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = mean(rx), mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else 0.0


def paired_effect_size(differences: Sequence[float]) -> float:
    if len(differences) < 2:
        return float("nan")
    spread = stdev(differences)
    return mean(differences) / spread if spread else math.copysign(float("inf"), mean(differences))


def paired_sign_flip_pvalue(
    differences: Sequence[float], *, samples: int = 20_000, seed: int = 0
) -> float:
    """Two-sided paired randomization test, suitable for the planned small seed count."""

    if not differences:
        raise ValueError("differences must be non-empty")
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(samples):
        permuted = mean(value * rng.choice((-1.0, 1.0)) for value in differences)
        extreme += abs(permuted) >= observed
    return extreme / (samples + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def flatten_result(result: dict[str, object]) -> list[dict[str, object]]:
    """Convert a run JSON to model×seed×depth×loop×condition long format."""

    config = result["config"]
    model = config["architecture"]
    seed = result["seed"]
    model_condition = result["protocol"].get("model_condition", "unspecified")
    rows: list[dict[str, object]] = []

    def add_metrics(condition: str, metrics: dict[str, object], **dimensions: object) -> None:
        for metric in (
            "accuracy", "loss", "hidden_norm", "final_update_ratio",
            "latency_per_example_seconds",
        ):
            if metric in metrics:
                rows.append({
                    "model": model,
                    "model_condition": model_condition,
                    "seed": seed,
                    "condition": condition,
                    "target_depth": dimensions.get("target_depth", ""),
                    "num_loops": dimensions.get("num_loops", ""),
                    "num_distractors": dimensions.get("num_distractors", ""),
                    "total_events": dimensions.get("total_events", ""),
                    "template_split": dimensions.get("template_split", "train"),
                    "metric": metric,
                    "value": metrics[metric],
                })

    add_metrics("id", result["E0_id"])
    for depth, metrics in result["E1_depth"].items():
        add_metrics("depth", metrics, target_depth=int(depth))
    for row in result.get("E2_loop_depth_matrix", []):
        add_metrics(
            "loop_depth", row,
            target_depth=row["target_depth"], num_loops=row["num_loops"],
        )
    for depth, metrics in result["E3_matched_length"].items():
        add_metrics(
            "matched_length", metrics, target_depth=int(depth),
            num_distractors=metrics["num_distractors"], total_events=metrics["total_events"],
        )
    for distractors, metrics in result["E4_distractors"].items():
        add_metrics(
            "distractor", metrics,
            target_depth=result["protocol"]["train_max_depth"],
            num_distractors=int(distractors),
        )
    add_metrics("linguistic_ood", result["E5_linguistic_ood"], template_split="ood")
    if "E5_lexical_ood" in result:
        add_metrics("lexical_ood", result["E5_lexical_ood"], template_split="lexical_ood")
    for row in result.get("E6_final_state_probe", []):
        rows.append({
            "model": model, "seed": seed, "condition": "probe",
            "model_condition": model_condition,
            "target_depth": result["protocol"]["train_max_depth"],
            "num_loops": row["loop"], "num_distractors": "", "total_events": "",
            "template_split": "train", "metric": "probe_accuracy",
            "value": row["probe_accuracy"],
        })
    return rows


def summarize_long_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    dimensions = (
        "model", "model_condition", "condition", "target_depth", "num_loops", "num_distractors",
        "total_events", "template_split", "metric",
    )
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(float(row["value"]))
    summary = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        low, high = bootstrap_mean_ci(values, samples=2_000)
        summary.append({
            **dict(zip(dimensions, key)),
            "n_seeds": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        })
    return summary
