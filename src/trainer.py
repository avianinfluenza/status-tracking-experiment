"""Adapter from the team's YAML configuration contract to the trainer."""

from __future__ import annotations

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
    halting = dict(config.get("adaptive_halting") or {})
    ablations = dict(config.get("ablations") or {})
    architecture = config.get("architecture") or model.get("architecture") or model.get("name")
    aliases = {
        "basic": "direct",
        "basic_transformer": "direct",
        "looped": "recurrent",
        "looped_transformer": "recurrent",
        "r0": "recurrent-r0",
        "recurrent_r0": "recurrent-r0",
        "looped_r0": "recurrent-r0",
    }
    architecture = aliases.get(str(architecture).lower(), architecture)
    if architecture not in ("direct", "cot", "recurrent", "recurrent-r0"):
        raise ValueError("config architecture must be direct, cot, recurrent, or recurrent-r0")

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
    _append(arguments, "--lr", training.get("learning_rate"))
    _append(arguments, "--weight-decay", training.get("weight_decay"))
    _append(arguments, "--grad-clip", training.get("gradient_clip"))
    seed = training.get("seed")
    if seed is None:
        seeds = training.get("seeds")
        if isinstance(seeds, list) and seeds:
            seed = seeds[0]
    _append(arguments, "--seed", seed)
    _append(arguments, "--device", config.get("device"))
    _append(arguments, "--data-dir", config.get("data_dir"))
    _append(arguments, "--output-dir", config.get("output_dir"))
    _append(arguments, "--kl-threshold", halting.get("threshold"))
    _append(arguments, "--adaptive-update-threshold", halting.get("update_threshold"))
    _append(arguments, "--adaptive-min-confidence", halting.get("min_confidence"))
    _append(arguments, "--min-loops", halting.get("min_loops"))
    _append(arguments, "--halting-patience", halting.get("patience"))
    _append(arguments, "--loop-conditioning", config.get("loop_conditioning", model.get("loop_conditioning")))
    _append(arguments, "--residual-scale", config.get("residual_scale", model.get("residual_scale")))
    _append(arguments, "--recurrent-blocks", config.get("recurrent_blocks", model.get("recurrent_blocks")))
    _append(arguments, "--max-loop-embeddings", config.get("max_loop_embeddings", model.get("max_loop_embeddings")))
    if config.get("random_loops") or training.get("random_loops"):
        arguments.append("--random-loops")
    _append(arguments, "--random-min-loops", training.get("random_min_loops"))
    _append(arguments, "--random-max-loops", training.get("random_max_loops"))
    loop_counts = config.get("eval_loop_counts") or dict(config.get("evaluation") or {}).get("loop_counts")
    if loop_counts:
        arguments.append("--eval-loop-counts")
        arguments.extend(map(str, loop_counts))
    _append(arguments, "--deep-supervision-weight", ablations.get("deep_supervision_weight"))
    _append(arguments, "--noop-eval-ratio", ablations.get("noop_eval_ratio"))
    if architecture in ("recurrent", "recurrent-r0") and halting.get("enabled_at_evaluation"):
        arguments.append("--adaptive-kl-eval")
    if config.get("smoke"):
        arguments.append("--smoke")
    return arguments


def run_from_config(config: dict[str, Any]) -> dict[str, object]:
    return run(parse_run_args(config_to_argv(config)))


__all__ = ["config_to_argv", "run_from_config"]
