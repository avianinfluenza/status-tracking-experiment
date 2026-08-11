#!/usr/bin/env python3
"""Run Basic and Looped models across at least three seeds and aggregate them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original.analysis import aggregate_results
from src.original.experiment import parse_args as parse_run_args
from src.original.experiment import run, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--architectures", nargs="+", choices=("direct", "recurrent"),
                        default=["direct", "recurrent"])
    parser.add_argument("--position-encoding", choices=("sinusoidal", "rope"),
                        default="sinusoidal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original")
    parser.add_argument("--adaptive-kl-eval", action="store_true")
    parser.add_argument("--deep-supervision-weight", type=float, default=0.0)
    parser.add_argument("--noop-eval-ratio", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) < 3:
        raise SystemExit("A comparison run requires at least three distinct seeds")
    results = []
    for architecture in args.architectures:
        for seed in args.seeds:
            run_argv = [
                "--architecture", architecture,
                "--position-encoding", args.position_encoding,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--data-dir", str(args.data_dir),
                "--output-dir", str(args.output_dir),
                "--noop-eval-ratio", str(args.noop_eval_ratio),
            ]
            if architecture == "recurrent":
                run_argv.extend(
                    ["--deep-supervision-weight", str(args.deep_supervision_weight)]
                )
                if args.adaptive_kl_eval:
                    run_argv.append("--adaptive-kl-eval")
            if args.smoke:
                run_argv.append("--smoke")
            print(f"[run] architecture={architecture} seed={seed}", flush=True)
            results.append(run(parse_run_args(run_argv)))

    aggregate_dir = args.output_dir / "aggregate"
    raw_rows, summaries = aggregate_results(results, aggregate_dir)
    manifest = {
        "track": "original_team_plan_multiseed",
        "architectures": args.architectures,
        "seeds": sorted(set(args.seeds)),
        "position_encoding": args.position_encoding,
        "smoke": args.smoke,
        "runs": [result["run_name"] for result in results],
        "raw_rows": len(raw_rows),
        "summary_rows": len(summaries),
        "aggregate_dir": str(aggregate_dir),
    }
    write_json(args.output_dir / "multiseed_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
