#!/usr/bin/env python3
"""Run diagnostic length-matched oracle evaluation from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original.experiment import (
    EVAL_SPLITS,
    evaluate_classifier,
    evaluate_length_matched_classifier,
    make_loader,
    maybe_compile_model,
    resolve_device,
)
from src.original.model import (
    FanRecurrentTransformer,
    OriginalModelConfig,
    RecurrentR0Transformer,
    RecurrentTransformer,
    build_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--swaps-per-loop", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(EVAL_SPLITS),
        help="JSONL stems to evaluate relative to --data-dir",
    )
    parser.add_argument(
        "--fixed-loop-counts",
        type=int,
        nargs="+",
        default=None,
        help="also evaluate every sample at these fixed recurrence counts",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--slot-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override checkpoint run metadata; otherwise auto-detected from sibling args.json",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.swaps_per_loop <= 0:
        raise SystemExit("--swaps-per-loop must be positive")
    if args.fixed_loop_counts and any(loop_count < 1 for loop_count in args.fixed_loop_counts):
        raise SystemExit("--fixed-loop-counts must all be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(checkpoint_config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain config and model_state")
    config = OriginalModelConfig(**checkpoint_config)
    model = build_model(config)
    if not isinstance(
        model,
        (RecurrentTransformer, RecurrentR0Transformer, FanRecurrentTransformer),
    ):
        raise SystemExit("length-matched evaluation requires a recurrent checkpoint")
    model.load_state_dict(model_state, strict=True)
    device = resolve_device(args.device)
    maybe_compile_model(model.to(device), device)

    run_args_path = args.checkpoint.parent / "args.json"
    saved_run_args = {}
    if run_args_path.is_file():
        loaded_args = json.loads(run_args_path.read_text(encoding="utf-8"))
        if isinstance(loaded_args, dict):
            saved_run_args = loaded_args
    slot_first = (
        bool(saved_run_args.get("slot_first", False))
        if args.slot_first is None
        else args.slot_first
    )

    splits: dict[str, object] = {}
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing dataset split: {path}")
        loader = make_loader(
            path,
            config.architecture,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=0,
        max_samples=args.max_eval_samples,
        slot_first=slot_first,
        swaps_per_loop=args.swaps_per_loop,
        input_format=config.fan_input_format if isinstance(model, FanRecurrentTransformer) else "template",
        )
        metrics = evaluate_length_matched_classifier(
            model,
            loader,
            device,
            swaps_per_loop=args.swaps_per_loop,
        )
        if args.fixed_loop_counts:
            metrics["fixed_loop_sweep"] = {
                str(loop_count): evaluate_classifier(
                    model,
                    loader,
                    device,
                    adaptive_kl=False,
                    num_loops=loop_count,
                )
                for loop_count in sorted(set(args.fixed_loop_counts))
            }
        splits[split] = metrics
    result = {
        "evaluation_mode": (
            "length_matched"
            if isinstance(model, FanRecurrentTransformer)
            else "length_matched_oracle"
        ),
        "checkpoint": str(args.checkpoint),
        "swaps_per_loop": args.swaps_per_loop,
        "fixed_loop_counts": sorted(set(args.fixed_loop_counts or [])),
        "slot_first": slot_first,
        "splits": splits,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
