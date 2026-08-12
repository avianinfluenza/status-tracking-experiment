#!/usr/bin/env python3
"""Aggregate seed runs into raw-long CSV, summary CSV, and hypothesis statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.systematic.analysis import (
    flatten_result,
    holm_adjust,
    paired_effect_size,
    paired_sign_flip_pvalue,
    spearman_rho,
    summarize_long_rows,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/systematic/aggregate"))
    args = parser.parse_args()

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    rows = [row for run in runs for row in flatten_result(run)]
    summary = summarize_long_rows(rows)
    write_csv(args.out_dir / "raw_long.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)

    slopes = [
        {
            "model": run["config"]["architecture"],
            "model_condition": run["protocol"].get("model_condition", "unspecified"),
            "seed": run["seed"],
            "ood_degradation_slope": run["analysis"]["ood_degradation_slope"],
        }
        for run in runs
    ]
    rhos = []
    for run in runs:
        if run["protocol"].get("model_condition") != "main_compute_matched":
            continue
        best = run.get("E2_best_loop_by_depth")
        if best:
            depths = sorted(int(depth) for depth in best)
            rhos.append({
                "seed": run["seed"],
                "spearman_depth_best_loop": spearman_rho(
                    depths, [int(best[str(depth)]) for depth in depths]
                ),
            })

    main_runs = [run for run in runs
                 if run["protocol"].get("model_condition") == "main_compute_matched"]
    seed_counts = {
        model: len({int(run["seed"]) for run in main_runs
                    if run["config"]["architecture"] == model})
        for model in ("standard", "recurrent")
    }
    accuracy_by_model_seed_depth: dict[tuple[str, int, int], float] = {}
    for run in main_runs:
        model, seed = run["config"]["architecture"], int(run["seed"])
        for depth, metrics in run["E1_depth"].items():
            accuracy_by_model_seed_depth[(model, seed, int(depth))] = float(metrics["accuracy"])
    depths = sorted({depth for _, _, depth in accuracy_by_model_seed_depth})
    paired = []
    p_values = {}
    for depth in depths:
        common_seeds = sorted({seed for model, seed, d in accuracy_by_model_seed_depth
                               if model == "standard" and d == depth}
                              & {seed for model, seed, d in accuracy_by_model_seed_depth
                                 if model == "recurrent" and d == depth})
        if len(common_seeds) < 3:
            continue
        differences = [
            accuracy_by_model_seed_depth[("recurrent", seed, depth)]
            - accuracy_by_model_seed_depth[("standard", seed, depth)]
            for seed in common_seeds
        ]
        key = str(depth)
        p_values[key] = paired_sign_flip_pvalue(differences, seed=depth)
        paired.append({
            "target_depth": depth,
            "n_pairs": len(differences),
            "mean_accuracy_gap": sum(differences) / len(differences),
            "paired_effect_size_dz": paired_effect_size(differences),
            "p_unadjusted": p_values[key],
        })
    adjusted = holm_adjust(p_values)
    for row in paired:
        row["p_holm"] = adjusted[str(row["target_depth"])]

    report = {
        "seed_counts_main_compute_matched": seed_counts,
        "minimum_three_seeds_met": all(count >= 3 for count in seed_counts.values()),
        "slopes": slopes,
        "spearman": rhos,
        "paired_depth_tests": paired,
    }
    (args.out_dir / "hypothesis_statistics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "runs": len(runs), "rows": len(rows)}))


if __name__ == "__main__":
    main()
