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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original" / "aggregate")
    args = parser.parse_args()
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    payloads = [payload for payload in loaded if payload.get("track") == "original_team_plan"]
    if not payloads:
        raise SystemExit("no original_team_plan run JSON files were provided")
    raw_rows, summaries = aggregate_results(payloads, args.output_dir)
    print(f"wrote {len(raw_rows)} raw rows and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
