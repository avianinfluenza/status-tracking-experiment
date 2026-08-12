#!/usr/bin/env python3
"""Generate the preregistered figures directly from aggregate summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover
    raise SystemExit("Install plotting dependencies with: pip install -e '.[analysis]'") from error


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def line_figure(rows, condition, output, *, train_max_depth, model_condition, ylabel="Accuracy"):
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    models = sorted({row["model"] for row in rows if row["condition"] == condition
                     and row["model_condition"] == model_condition})
    for model in models:
        selected = sorted(
            (row for row in rows if row["condition"] == condition
             and row["model"] == model and row["metric"] == "accuracy"
             and row["model_condition"] == model_condition),
            key=lambda row: numeric(row, "target_depth"),
        )
        if not selected:
            continue
        x = [numeric(row, "target_depth") for row in selected]
        y = [numeric(row, "mean") for row in selected]
        spread = [numeric(row, "std") for row in selected]
        axis.plot(x, y, marker="o", label=model)
        axis.fill_between(x, [a - b for a, b in zip(y, spread)],
                          [a + b for a, b in zip(y, spread)], alpha=0.18)
    axis.axvline(train_max_depth, color="black", linestyle="--", linewidth=1, label="train max")
    axis.set(xlabel="Target transition depth", ylabel=ylabel, ylim=(0, 1.02))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def heatmap_figure(rows, output, model_condition):
    selected = [row for row in rows if row["condition"] == "loop_depth"
                and row["metric"] == "accuracy" and row["model"] == "recurrent"
                and row["model_condition"] == model_condition]
    depths = sorted({int(row["target_depth"]) for row in selected})
    loops = sorted({int(row["num_loops"]) for row in selected})
    values = {(int(row["target_depth"]), int(row["num_loops"])): float(row["mean"])
              for row in selected}
    matrix = [[values.get((depth, loop), float("nan")) for loop in loops] for depth in depths]
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(loops)), loops)
    axis.set_yticks(range(len(depths)), depths)
    axis.set(xlabel="Inference loops K", ylabel="Target depth D")
    fig.colorbar(image, ax=axis, label="Accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def probe_figure(rows, output, model_condition):
    selected = sorted(
        (row for row in rows if row["condition"] == "probe" and row["metric"] == "probe_accuracy"
         and row["model_condition"] == model_condition),
        key=lambda row: int(row["num_loops"]),
    )
    if not selected:
        return
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot([int(row["num_loops"]) for row in selected],
              [float(row["mean"]) for row in selected], marker="o")
    axis.set(xlabel="Loop", ylabel="Linear-probe accuracy", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/systematic/figures"))
    parser.add_argument("--train-max-depth", type=int, default=8)
    parser.add_argument("--model-condition", default="main_compute_matched")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.summary)
    line_figure(rows, "depth", args.out_dir / "figure2_depth_generalization.png",
                train_max_depth=args.train_max_depth, model_condition=args.model_condition)
    heatmap_figure(rows, args.out_dir / "figure3_loop_depth_heatmap.png", args.model_condition)
    line_figure(rows, "matched_length", args.out_dir / "figure4_matched_length.png",
                train_max_depth=args.train_max_depth, model_condition=args.model_condition)
    probe_figure(rows, args.out_dir / "figure5_probe_by_loop.png", args.model_condition)


if __name__ == "__main__":
    main()
