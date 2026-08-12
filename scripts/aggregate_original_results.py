#!/usr/bin/env python3
"""Aggregate independently produced original-plan result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original.analysis import aggregate_results


def expand_result_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            direct = path / "result.json"
            if direct.exists():
                paths.append(direct)
            else:
                paths.extend(sorted(path.rglob("result.json")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original" / "aggregate")
    args = parser.parse_args()
    result_paths = expand_result_paths(args.results)
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    payloads = [payload for payload in loaded if payload.get("track") == "original_team_plan"]
    if not payloads:
        raise SystemExit("no original_team_plan run JSON files were provided")
    raw_rows, summaries = aggregate_results(payloads, args.output_dir)
    print(f"wrote {len(raw_rows)} raw rows and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
