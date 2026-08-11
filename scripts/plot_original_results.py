#!/usr/bin/env python3
"""Plot cross-seed exact match by swap length for the original experiment."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # Pure-SVG fallback keeps the analysis runnable without extras.
    plt = None


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")


def render_svg(
    groups: dict[tuple[str, str, str], list[dict[str, str]]],
    splits: list[str],
    output: Path,
) -> None:
    """Render dependency-free vector charts when matplotlib is unavailable."""

    width, panel_height = 900, 300
    left, right, top, bottom = 80, 30, 45, 55
    height = panel_height * len(splits)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#374151;stroke-width:1}'
        '.grid{stroke:#d1d5db;stroke-width:1}.label{font-size:12px}.title{font-size:16px;font-weight:600}'
        '.legend{font-size:12px}</style>',
    ]
    for panel, split in enumerate(splits):
        y0 = panel * panel_height
        plot_left, plot_right = left, width - right
        plot_top, plot_bottom = y0 + top, y0 + panel_height - bottom
        split_groups = [
            (key, rows) for key, rows in sorted(groups.items()) if key[0] == split
        ]
        all_x = [int(row["n_swaps"]) for _, rows in split_groups for row in rows]
        x_min, x_max = min(all_x), max(all_x)
        x_span = max(x_max - x_min, 1)

        def x_coord(value: int) -> float:
            return plot_left + (value - x_min) / x_span * (plot_right - plot_left)

        def y_coord(value: float) -> float:
            return plot_bottom - value * (plot_bottom - plot_top)

        elements.append(f'<text x="{left}" y="{y0 + 24}" class="title">{html.escape(split)}</text>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_coord(tick)
            elements.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" class="grid"/>')
            elements.append(f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{tick:.2f}</text>')
        elements.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>')
        elements.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>')
        for tick in sorted(set(all_x)):
            x = x_coord(tick)
            elements.append(f'<text x="{x:.1f}" y="{plot_bottom + 20}" text-anchor="middle" class="label">{tick}</text>')
        elements.append(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{y0 + panel_height - 10}" text-anchor="middle" class="label">number of swaps</text>')

        for group_index, ((_, architecture, position), rows) in enumerate(split_groups):
            rows.sort(key=lambda row: int(row["n_swaps"]))
            color = COLORS[group_index % len(COLORS)]
            points = []
            for row in rows:
                x = x_coord(int(row["n_swaps"]))
                mean, std = float(row["mean"]), float(row["std"])
                y = y_coord(mean)
                points.append(f"{x:.1f},{y:.1f}")
                low, high = y_coord(max(mean - std, 0.0)), y_coord(min(mean + std, 1.0))
                elements.append(f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1"/>')
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            legend_x = plot_left + group_index * 180
            elements.append(f'<line x1="{legend_x}" y1="{plot_top + 14}" x2="{legend_x + 22}" y2="{plot_top + 14}" stroke="{color}" stroke-width="3"/>')
            elements.append(f'<text x="{legend_x + 28}" y="{plot_top + 18}" class="legend">{html.escape(architecture + "/" + position)}</text>')
    elements.append("</svg>")
    output.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/original/figures"))
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row for row in rows
        if row["metric"] == "exact_match" and row["n_swaps"] != "ALL"
    ]
    if not selected:
        raise SystemExit("summary contains no per-swap exact_match rows")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[(row["split"], row["architecture"], row["position_encoding"])].append(row)

    splits = sorted({key[0] for key in groups})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        output = args.output_dir / "exact_match_by_swap_length.svg"
        render_svg(groups, splits, output)
        print(output)
        return

    figure, axes = plt.subplots(len(splits), 1, figsize=(8, max(4, 3.2 * len(splits))), squeeze=False)
    for axis, split in zip(axes[:, 0], splits, strict=True):
        for (row_split, architecture, position), group in sorted(groups.items()):
            if row_split != split:
                continue
            group.sort(key=lambda row: int(row["n_swaps"]))
            x = [int(row["n_swaps"]) for row in group]
            mean = [float(row["mean"]) for row in group]
            std = [float(row["std"]) for row in group]
            axis.plot(x, mean, marker="o", label=f"{architecture}/{position}")
            axis.fill_between(x, [m - s for m, s in zip(mean, std)],
                              [m + s for m, s in zip(mean, std)], alpha=0.15)
        axis.set_title(split)
        axis.set_xlabel("number of swaps")
        axis.set_ylabel("five-person exact match")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output = args.output_dir / "exact_match_by_swap_length.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
