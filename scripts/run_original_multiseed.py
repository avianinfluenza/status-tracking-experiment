#!/usr/bin/env python3
"""Run original ball-swap models across at least three seeds and aggregate them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original.analysis import aggregate_results
from src.original.experiment import create_unique_run_dir
from src.original.experiment import parse_args as parse_run_args
from src.original.experiment import run, write_json


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--architectures", nargs="+", choices=("direct", "recurrent", "recurrent-r0", "fan-recurrent", "event-recurrent"),
                        default=["direct", "recurrent"])
    parser.add_argument("--position-encoding", choices=("none", "sinusoidal", "rope"),
                        default="sinusoidal")
    parser.add_argument("--fan-input-format", choices=("template", "atomic"), default="template")
    parser.add_argument("--fan-positional-control", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-loops", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original")
    parser.add_argument("--adaptive-kl-eval", action="store_true")
    parser.add_argument("--adaptive-update-threshold", type=float, default=1e9)
    parser.add_argument("--adaptive-min-confidence", type=float, default=0.0)
    parser.add_argument("--adaptive-max-loops", type=int, default=None)
    parser.add_argument("--eval-loop-counts", type=int, nargs="+", default=None)
    parser.add_argument("--deep-supervision-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-probe-eval", action="store_true")
    parser.add_argument("--event-trajectory-probe", action="store_true")
    parser.add_argument("--noop-eval-ratio", type=float, default=0.0)
    parser.add_argument(
        "--slot-first", "--slot_first", dest="slot_first", nargs="?", const=True,
        default=False, type=parse_bool,
    )
    parser.add_argument("--extended-length", "--extended_length", dest="extended_length", action="store_true")
    parser.add_argument("--loop-conditioning", choices=("none", "learned"), default="none")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-blocks", type=int, default=1)
    parser.add_argument("--max-loop-embeddings", type=int, default=64)
    parser.add_argument("--random-loops", action="store_true")
    parser.add_argument("--random-min-loops", type=int, default=None)
    parser.add_argument("--random-max-loops", type=int, default=None)
    parser.add_argument("--swaps-per-loop", type=float, default=None)
    parser.add_argument("--length-matched-eval", action="store_true")
    parser.add_argument("--online-training", action="store_true")
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-min-swaps", type=int, default=2)
    parser.add_argument("--curriculum-max-swaps", type=int, default=10)
    parser.add_argument("--curriculum-steps-per-length", type=int, default=1_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) < 3:
        raise SystemExit("A comparison run requires at least three distinct seeds")
    recurrent_architectures = {"recurrent", "recurrent-r0", "fan-recurrent"}
    if args.event_trajectory_probe and any(
        architecture != "event-recurrent" for architecture in args.architectures
    ):
        raise SystemExit("event trajectory probing requires event-recurrent only")
    if (args.trajectory_probe_eval or args.length_matched_eval) and any(
        architecture not in recurrent_architectures for architecture in args.architectures
    ):
        raise SystemExit("trajectory probing requires recurrent architectures only")
    if args.swaps_per_loop is not None:
        if args.swaps_per_loop <= 0:
            raise SystemExit("--swaps-per-loop must be positive")
        if args.random_loops:
            raise SystemExit("--swaps-per-loop and --random-loops are mutually exclusive")
        if any(architecture not in recurrent_architectures for architecture in args.architectures):
            raise SystemExit("--swaps-per-loop requires recurrent architectures only")
    if args.length_matched_eval and args.swaps_per_loop is None:
        raise SystemExit("--length-matched-eval requires --swaps-per-loop")
    if args.adaptive_max_loops is not None and not args.adaptive_kl_eval:
        raise SystemExit("--adaptive-max-loops requires --adaptive-kl-eval")
    if "fan-recurrent" in args.architectures:
        if args.fan_input_format == "atomic":
            if args.position_encoding != "none" or args.fan_positional_control:
                raise SystemExit("atomic Fan control requires --position-encoding none without --fan-positional-control")
        elif args.fan_positional_control:
            if args.position_encoding != "sinusoidal":
                raise SystemExit("--fan-positional-control requires --position-encoding sinusoidal")
        elif args.position_encoding != "none":
            raise SystemExit("fan-recurrent requires --position-encoding none unless --fan-positional-control is set")
        if not args.online_training:
            raise SystemExit("fan-recurrent requires --online-training")
        if not args.fan_positional_control and args.fan_input_format == "template" and len(set(args.seeds)) < 5:
            raise SystemExit("the Fan-aligned comparison requires at least five distinct seeds")
    suite_name = (
        f"multiseed-{'-'.join(args.architectures)}"
        f"-seeds{'-'.join(map(str, sorted(set(args.seeds))))}"
    )
    suite_dir = create_unique_run_dir(args.output_dir, suite_name)
    write_json(suite_dir / "args.json", vars(args))
    results = []
    for architecture in args.architectures:
        for seed in args.seeds:
            run_argv = [
                "--architecture", architecture,
                "--position-encoding", args.position_encoding,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--d-model", str(args.d_model),
                "--n-heads", str(args.n_heads),
                "--d-ff", str(args.d_ff),
                "--num-layers", str(args.num_layers),
                "--num-loops", str(args.num_loops),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--grad-clip", str(args.grad_clip),
                "--device", args.device,
                "--data-dir", str(args.data_dir),
                "--output-dir", str(suite_dir),
                "--noop-eval-ratio", str(args.noop_eval_ratio),
            ]
            if args.slot_first:
                run_argv.append("--slot-first")
            if args.extended_length:
                run_argv.append("--extended-length")
            if args.no_progress:
                run_argv.append("--no-progress")
            if architecture in recurrent_architectures:
                run_argv.extend(
                    ["--deep-supervision-weight", str(args.deep_supervision_weight)]
                )
                if args.trajectory_probe_eval:
                    run_argv.append("--trajectory-probe-eval")
                if args.swaps_per_loop is not None:
                    run_argv.extend(["--swaps-per-loop", str(args.swaps_per_loop)])
                if args.length_matched_eval:
                    run_argv.append("--length-matched-eval")
                if args.eval_loop_counts:
                    run_argv.append("--eval-loop-counts")
                    run_argv.extend(map(str, args.eval_loop_counts))
                if args.adaptive_kl_eval and architecture in ("recurrent", "recurrent-r0"):
                    run_argv.extend([
                        "--adaptive-kl-eval",
                        "--adaptive-update-threshold", str(args.adaptive_update_threshold),
                        "--adaptive-min-confidence", str(args.adaptive_min_confidence),
                    ])
                    if args.adaptive_max_loops is not None:
                        run_argv.extend(["--adaptive-max-loops", str(args.adaptive_max_loops)])
            if architecture == "recurrent-r0":
                run_argv.extend([
                    "--loop-conditioning", args.loop_conditioning,
                    "--residual-scale", str(args.residual_scale),
                    "--recurrent-blocks", str(args.recurrent_blocks),
                    "--max-loop-embeddings", str(args.max_loop_embeddings),
                ])
                if args.random_loops:
                    run_argv.append("--random-loops")
                if args.random_min_loops is not None:
                    run_argv.extend(["--random-min-loops", str(args.random_min_loops)])
                if args.random_max_loops is not None:
                    run_argv.extend(["--random-max-loops", str(args.random_max_loops)])
            if architecture == "event-recurrent":
                if args.event_trajectory_probe:
                    run_argv.append("--event-trajectory-probe")
            if architecture == "fan-recurrent":
                run_argv.extend(
                    [
                        "--fan-input-format", args.fan_input_format,
                        "--online-training",
                        "--train-steps", str(args.train_steps),
                        "--curriculum-min-swaps", str(args.curriculum_min_swaps),
                        "--curriculum-max-swaps", str(args.curriculum_max_swaps),
                        "--curriculum-steps-per-length",
                        str(args.curriculum_steps_per_length),
                    ]
                )
                if args.fan_positional_control:
                    run_argv.append("--fan-positional-control")
            if args.smoke:
                run_argv.append("--smoke")
            print(f"[run] architecture={architecture} seed={seed}", flush=True)
            results.append(run(parse_run_args(run_argv)))

    aggregate_dir = suite_dir / "aggregate"
    raw_rows, summaries = aggregate_results(results, aggregate_dir)
    manifest = {
        "track": "original_team_plan_multiseed",
        "suite_dir": str(suite_dir),
        "architectures": args.architectures,
        "seeds": sorted(set(args.seeds)),
        "position_encoding": args.position_encoding,
        "smoke": args.smoke,
        "slot_first": args.slot_first,
        "extended_length": args.extended_length,
        "trajectory_probe_eval": args.trajectory_probe_eval,
        "event_trajectory_probe": args.event_trajectory_probe,
        "online_training": args.online_training,
        "train_steps": args.train_steps,
        "curriculum": {
            "min_swaps": args.curriculum_min_swaps,
            "max_swaps": args.curriculum_max_swaps,
            "steps_per_length": args.curriculum_steps_per_length,
        },
        "swaps_per_loop": args.swaps_per_loop,
        "length_matched_eval": args.length_matched_eval,
        "adaptive_max_loops": args.adaptive_max_loops,
        "runs": [
            {
                "run_name": result["run_name"],
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
                "result": result["paths"]["result"],
            }
            for result in results
        ],
        "raw_rows": len(raw_rows),
        "summary_rows": len(summaries),
        "aggregate_dir": str(aggregate_dir),
    }
    write_json(suite_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
