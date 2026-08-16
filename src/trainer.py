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
        "fan": "fan-recurrent",
        "fan_recurrent": "fan-recurrent",
        "fan_atomic": "fan-recurrent",
        "event_wise": "event-recurrent",
        "event_recurrent": "event-recurrent",
    }
    architecture = aliases.get(str(architecture).lower(), architecture)
    if architecture not in (
        "direct",
        "cot",
        "recurrent",
        "recurrent-r0",
        "fan-recurrent",
        "event-recurrent",
    ):
        raise ValueError(
            "config architecture must be direct, cot, recurrent, recurrent-r0, "
            "fan-recurrent, or event-recurrent"
        )

    arguments = ["--architecture", str(architecture)]
    _append(arguments, "--position-encoding", config.get("position_encoding", model.get("position_encoding")))
    _append(arguments, "--fan-input-format", config.get("fan_input_format", model.get("fan_input_format")))
    _append(arguments, "--direct-input-format", config.get("direct_input_format", model.get("direct_input_format")))
    direct_causal = config.get("direct_causal", model.get("direct_causal"))
    if direct_causal:
        arguments.append("--direct-causal")
    fan_positional_control = config.get(
        "fan_positional_control", model.get("fan_positional_control")
    )
    if fan_positional_control:
        arguments.append("--fan-positional-control")
    _append(arguments, "--d-model", config.get("d_model", model.get("d_model")))
    _append(arguments, "--n-heads", config.get("n_heads", model.get("n_heads")))
    _append(arguments, "--d-ff", config.get("d_ff", model.get("d_ff")))
    _append(arguments, "--dropout", config.get("dropout", model.get("dropout")))
    _append(arguments, "--num-layers", config.get("num_layers", model.get("num_layers")))
    _append(arguments, "--num-loops", config.get("num_loops", model.get("num_loops")))
    _append(arguments, "--classifier-dim", config.get("classifier_dim", model.get("classifier_dim")))
    _append(arguments, "--epochs", training.get("epochs"))
    if training.get("online_training"):
        arguments.append("--online-training")
    _append(arguments, "--train-steps", training.get("train_steps"))
    _append(arguments, "--curriculum-min-swaps", training.get("curriculum_min_swaps"))
    _append(arguments, "--curriculum-max-swaps", training.get("curriculum_max_swaps"))
    _append(
        arguments,
        "--curriculum-steps-per-length",
        training.get("curriculum_steps_per_length"),
    )
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
    _append(arguments, "--swaps-per-loop", training.get("swaps_per_loop"))
    loop_counts = config.get("eval_loop_counts") or dict(config.get("evaluation") or {}).get("loop_counts")
    if loop_counts:
        arguments.append("--eval-loop-counts")
        arguments.extend(map(str, loop_counts))
    _append(arguments, "--deep-supervision-weight", ablations.get("deep_supervision_weight"))
    if config.get("trajectory_probe_eval") or dict(config.get("evaluation") or {}).get("trajectory_probe_eval"):
        arguments.append("--trajectory-probe-eval")
    if dict(config.get("evaluation") or {}).get("event_trajectory_probe"):
        arguments.append("--event-trajectory-probe")
    if dict(config.get("evaluation") or {}).get("length_matched_eval"):
        arguments.append("--length-matched-eval")
    _append(arguments, "--noop-eval-ratio", ablations.get("noop_eval_ratio"))
    if architecture in ("recurrent", "recurrent-r0") and halting.get("enabled_at_evaluation"):
        arguments.append("--adaptive-kl-eval")
        _append(arguments, "--adaptive-max-loops", halting.get("max_loops"))
    if config.get("progress") is False:
        arguments.append("--no-progress")
    if config.get("slot_first", (config.get("model") or {}).get("slot_first", False)):
        arguments.append("--slot-first")
    if config.get("extended_length", False):
        arguments.append("--extended-length")
    if config.get("smoke"):
        arguments.append("--smoke")
    return arguments


def run_from_config(config: dict[str, Any]) -> dict[str, object]:
    return run(parse_run_args(config_to_argv(config)))


__all__ = ["config_to_argv", "run_from_config"]
