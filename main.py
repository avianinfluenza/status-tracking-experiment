#!/usr/bin/env python3
"""Unified trainer entry point for config-based and direct CLI runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.original.experiment import main as direct_cli_main
from src.trainer import run_from_config


def parse_config_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("config must contain a YAML mapping")
    return loaded


def override_config(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    result = dict(config)
    training = dict(result.get("training") or result.get("train") or {})
    if args.device is not None:
        result["device"] = args.device
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.seed is not None:
        training["seed"] = args.seed
    if args.data_dir is not None:
        result["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        result["output_dir"] = str(args.output_dir)
    if args.smoke:
        result["smoke"] = True
    result["training"] = training
    result.pop("train", None)
    return result


def main() -> None:
    # Preserve the existing direct CLI while adding the team's YAML entrypoint.
    if "--config" not in sys.argv:
        direct_cli_main()
        return
    args = parse_config_args()
    result = run_from_config(override_config(load_yaml(args.config), args))
    compact = {
        split: {
            "exact_match": metrics["exact_match"],
            "slot_accuracy": metrics["slot_accuracy"],
        }
        for split, metrics in result["splits"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
