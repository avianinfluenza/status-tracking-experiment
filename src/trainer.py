"""Adapter from the team's YAML configuration contract to the trainer."""

from __future__ import annotations

import copy
from typing import Any

from .original.experiment import parse_args as parse_run_args
from .original.experiment import run


def _append(arguments: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        arguments.extend((flag, str(value)))


def config_to_argv(config: dict[str, Any]) -> list[str]:
    """Translate a checked YAML mapping into the canonical experiment CLI."""

    model = dict(config.get("model") or {})
    training = dict(config.get("training") or config.get("train") or {})
    evaluation = dict(config.get("evaluation") or {})
    validation = dict(config.get("validation") or {})
    checkpointing = dict(config.get("checkpointing") or {})
    performance = dict(config.get("performance") or {})
    halting = dict(config.get("adaptive_halting") or {})
    ablations = dict(config.get("ablations") or {})
    architecture = config.get("architecture") or model.get("architecture") or model.get("name")
    aliases = {
        "basic": "direct",
        "basic_transformer": "direct",
        "looped": "recurrent",
        "looped_transformer": "recurrent",
    }
    architecture = aliases.get(str(architecture).lower(), architecture)
    if architecture not in ("direct", "cot", "recurrent"):
        raise ValueError("config architecture must be direct, cot, or recurrent")

    arguments = ["--architecture", str(architecture)]
    _append(arguments, "--position-encoding", config.get("position_encoding", model.get("position_encoding")))
    _append(arguments, "--d-model", config.get("d_model", model.get("d_model")))
    _append(arguments, "--n-heads", config.get("n_heads", model.get("n_heads")))
    _append(arguments, "--d-ff", config.get("d_ff", model.get("d_ff")))
    _append(arguments, "--dropout", config.get("dropout", model.get("dropout")))
    _append(arguments, "--num-layers", config.get("num_layers", model.get("num_layers")))
    _append(arguments, "--num-loops", config.get("num_loops", model.get("num_loops")))
    _append(arguments, "--classifier-dim", config.get("classifier_dim", model.get("classifier_dim")))
    _append(arguments, "--epochs", training.get("epochs"))
    _append(arguments, "--batch-size", training.get("batch_size"))
    _append(arguments, "--eval-batch-size", evaluation.get("batch_size"))
    _append(arguments, "--validation-ratio", validation.get("ratio"))
    _append(arguments, "--lr", training.get("learning_rate"))
    _append(arguments, "--weight-decay", training.get("weight_decay"))
    _append(arguments, "--grad-clip", training.get("gradient_clip"))
    optimizer = training.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer = optimizer.get("name")
    _append(arguments, "--optimizer", str(optimizer).lower() if optimizer else None)
    scheduler = training.get("scheduler")
    if isinstance(scheduler, dict):
        _append(arguments, "--warmup-epochs", scheduler.get("warmup_epochs"))
        _append(arguments, "--min-lr", scheduler.get("min_lr"))
        scheduler = scheduler.get("name")
    _append(arguments, "--scheduler", str(scheduler).lower() if scheduler else None)
    seed = training.get("seed")
    if seed is None:
        seeds = training.get("seeds")
        if isinstance(seeds, list) and seeds:
            seed = seeds[0]
    _append(arguments, "--seed", seed)
    _append(arguments, "--device", config.get("device"))
    _append(arguments, "--data-dir", config.get("data_dir"))
    _append(arguments, "--output-dir", config.get("output_dir"))
    _append(arguments, "--run-name", config.get("run_name"))
    _append(arguments, "--num-workers", performance.get("num_workers"))
    _append(arguments, "--checkpoint-every", checkpointing.get("save_every"))
    _append(arguments, "--resume", checkpointing.get("resume"))
    if checkpointing.get("overwrite"):
        arguments.append("--overwrite")
    splits = evaluation.get("splits")
    if splits:
        arguments.append("--eval-splits")
        arguments.extend(map(str, splits))
    metrics = evaluation.get("metrics")
    if metrics:
        arguments.append("--eval-metrics")
        arguments.extend(map(str, metrics))
    _append(arguments, "--kl-threshold", halting.get("threshold"))
    _append(arguments, "--min-loops", halting.get("min_loops"))
    _append(arguments, "--halting-patience", halting.get("patience"))
    _append(arguments, "--deep-supervision-weight", ablations.get("deep_supervision_weight"))
    _append(arguments, "--noop-eval-ratio", ablations.get("noop_eval_ratio"))
    if architecture == "recurrent" and halting.get("enabled_at_evaluation"):
        arguments.append("--adaptive-kl-eval")
    if performance.get("amp", training.get("amp", False)):
        arguments.append("--amp")
    if performance.get("pin_memory") is True:
        arguments.append("--pin-memory")
    elif performance.get("pin_memory") is False:
        arguments.append("--no-pin-memory")
    if performance.get("persistent_workers"):
        arguments.append("--persistent-workers")
    if config.get("smoke"):
        arguments.append("--smoke")
    return arguments


def run_from_config(config: dict[str, Any]) -> dict[str, object]:
    """Run one seed, or every seed listed in ``training.seeds``."""

    training = dict(config.get("training") or config.get("train") or {})
    explicit_seed = training.get("seed")
    seeds = [explicit_seed] if explicit_seed is not None else training.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a non-empty list")
    results: list[dict[str, object]] = []
    for seed in seeds:
        seeded = copy.deepcopy(config)
        seeded_training = dict(seeded.get("training") or seeded.get("train") or {})
        seeded_training["seed"] = int(seed)
        seeded_training.pop("seeds", None)
        seeded["training"] = seeded_training
        seeded.pop("train", None)
        if len(seeds) > 1 and config.get("run_name"):
            seeded["run_name"] = f"{config['run_name']}-seed{int(seed)}"
        results.append(run(parse_run_args(config_to_argv(seeded))))
    if len(results) == 1:
        return results[0]
    return {
        "track": "original_team_plan_multiseed_config",
        "seeds": [int(seed) for seed in seeds],
        "runs": results,
    }


__all__ = ["config_to_argv", "run_from_config"]
