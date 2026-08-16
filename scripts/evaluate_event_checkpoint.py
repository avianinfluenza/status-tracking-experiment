#!/usr/bin/env python3
"""Evaluate an event-recurrent checkpoint on arbitrary JSONL splits."""

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
    evaluate_event_trajectory_probe,
    make_loader,
    maybe_compile_model,
    resolve_device,
)
from src.original.model import EventWiseRecurrentTransformer, OriginalModelConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--trajectory-probe",
        action="store_true",
        help="also report state accuracy after each real event",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(checkpoint_config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain config and model_state")
    config = OriginalModelConfig(**checkpoint_config)
    model = build_model(config)
    if not isinstance(model, EventWiseRecurrentTransformer):
        raise SystemExit("this evaluator requires an event-recurrent checkpoint")
    model.load_state_dict(model_state, strict=True)
    device = resolve_device(args.device)
    maybe_compile_model(model.to(device), device)

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
        )
        metrics = evaluate_classifier(
            model,
            loader,
            device,
            adaptive_kl=False,
        )
        if args.trajectory_probe:
            metrics["event_trajectory_probe"] = evaluate_event_trajectory_probe(
                model,
                loader,
                device,
            )
        splits[split] = metrics

    result = {
        "evaluation_mode": "native_event_recurrence",
        "checkpoint": str(args.checkpoint),
        "splits": splits,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
