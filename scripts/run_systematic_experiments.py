#!/usr/bin/env python3
"""Run the preregistered E0--E7 natural-language state-tracking protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.systematic.data import (
    GENERATOR_VERSION,
    OBJECTS,
    OOD_LOCATIONS,
    OOD_OBJECTS,
    LOCATIONS,
    StateTrackingDataset,
    StateTrackingGenerator,
    TokenVocabulary,
    collate_examples,
)
from src.systematic.experiment import (
    best_loop_by_depth,
    collect_loop_cls_states,
    evaluate,
    fit_loop_probes,
    loop_depth_sweep,
    matched_length_grid,
    ood_degradation_slope,
    trajectory_readout_matrix,
    train_epoch,
)
from src.systematic.model import (
    ModelConfig,
    StateTrackingTransformer,
    closest_parameter_matched_width,
    count_parameters,
    estimate_forward_flops,
)


def loader(examples, vocab, batch_size, shuffle=False):
    return DataLoader(
        StateTrackingDataset(examples, vocab),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_examples, pad_id=vocab.pad_id),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pop_predictions(metrics: dict[str, object], sink: list[dict[str, object]], **tags: object) -> None:
    for row in metrics.pop("predictions", []):
        sink.append({**tags, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("standard", "recurrent", "untied"), default="recurrent")
    parser.add_argument("--loop-conditioning", choices=("none", "learned"), default="none")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-blocks", type=int, default=1)
    parser.add_argument("--random-loops", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-examples", type=int, default=100_000)
    parser.add_argument("--test-examples-per-cell", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-locations", type=int, choices=(8, 12), default=8)
    parser.add_argument("--target-parameters", type=int)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--train-loops", type=int, default=6)
    parser.add_argument("--train-max-depth", type=int, default=8)
    parser.add_argument("--ood-depths", type=int, nargs="+", default=[10, 12, 16, 20, 24, 32])
    parser.add_argument("--loop-counts", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16, 24])
    parser.add_argument("--matched-total-events", type=int, default=24)
    parser.add_argument("--matched-depths", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20])
    parser.add_argument("--distractor-counts", type=int, nargs="+", default=[0, 4, 8, 16, 32, 64])
    parser.add_argument("--id-threshold", type=float, default=0.95)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--condition-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.epochs = 1
        args.train_examples = 64
        args.test_examples_per_cell = 8
        args.batch_size = 16
        args.d_model = 32
        args.num_layers = 2
        args.train_loops = 2
        args.train_max_depth = 2
        args.ood_depths = [4, 8]
        args.loop_counts = [1, 2, 4, 8]
        args.matched_total_events = 8
        args.matched_depths = [2, 4, 8]
        args.distractor_counts = [0, 4, 8]

    if any(depth <= args.train_max_depth for depth in args.ood_depths):
        raise ValueError("all OOD depths must be strictly above train_max_depth")
    if any(depth > args.matched_total_events for depth in args.matched_depths):
        raise ValueError("matched depths cannot exceed matched_total_events")
    condition_name = args.condition_name
    if condition_name is None:
        if args.target_parameters is not None:
            condition_name = "parameter_matched"
        elif args.random_loops:
            condition_name = "random_loop_ablation"
        elif args.loop_conditioning != "none":
            condition_name = "loop_embedding_ablation"
        elif args.residual_scale != 1.0:
            condition_name = "residual_scaling_ablation"
        elif args.recurrent_blocks != 1:
            condition_name = f"sharing_{args.recurrent_blocks}_blocks"
        else:
            condition_name = "main_compute_matched"

    seen: set[str] = set()
    train_generator = StateTrackingGenerator(num_locations=args.num_locations, seed=args.seed)
    train_examples = []
    while len(train_examples) < args.train_examples:
        example = train_generator.generate(
            target_depth=train_generator.rng.randint(1, args.train_max_depth),
            num_distractors=train_generator.rng.randint(0, 16),
            num_entities=train_generator.rng.randint(2, 8),
            linguistic_variation=False,
        )
        if example.example_id not in seen:
            seen.add(example.example_id)
            train_examples.append(example)
    assert max(example.target_depth for example in train_examples) <= args.train_max_depth

    cell_count = args.test_examples_per_cell
    depth_examples = {}
    for depth in [*range(1, args.train_max_depth + 1), *sorted(set(args.ood_depths))]:
        generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 1_000 + depth
        )
        depth_examples[depth] = generator.generate_unique(
            cell_count,
            seen=seen,
            target_depth=depth,
            num_distractors=8,
            num_entities=6,
        )
    id_examples = [example for depth in range(1, args.train_max_depth + 1)
                   for example in depth_examples[depth]]

    vocab = TokenVocabulary.from_schema(num_locations=args.num_locations)
    config = ModelConfig(
        vocab_size=len(vocab),
        num_locations=args.num_locations,
        pad_id=vocab.pad_id,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=4,
        d_ff=4 * args.d_model,
        num_layers=args.num_layers,
        train_loops=args.train_loops,
        recurrent_blocks=args.recurrent_blocks,
        loop_conditioning=args.loop_conditioning,
        residual_scale=args.residual_scale,
    )
    matched_parameter_count = None
    if args.target_parameters is not None:
        matched_width, matched_parameter_count = closest_parameter_matched_width(
            config, args.target_parameters
        )
        config = ModelConfig(**{
            **config.to_dict(), "d_model": matched_width, "d_ff": 4 * matched_width,
        })

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = StateTrackingTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == "cosine" else None
    )
    train_loader = loader(train_examples, vocab, args.batch_size, shuffle=True)
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model, train_loader, optimizer, device,
            random_loop_range=(max(1, args.train_loops // 2), args.train_loops)
            if args.random_loops else None,
        )
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **metrics.__dict__})
        if scheduler is not None:
            scheduler.step()
    training_seconds = time.perf_counter() - training_start

    output = args.output or ROOT / "results" / f"systematic_{args.architecture}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".pt")
    torch.save({
        "model_config": config.to_dict(),
        "model_state": model.state_dict(),
        "vocabulary": vocab.itos,
        "seed": args.seed,
        "generator_version": GENERATOR_VERSION,
    }, checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    predictions: list[dict[str, object]] = []
    id_metrics = evaluate(
        model, loader(id_examples, vocab, args.batch_size), device,
        collect_predictions=True,
    )
    pop_predictions(id_metrics, predictions, condition="id", num_loops=None)
    id_by_depth = {int(depth): float(accuracy) for depth, accuracy in id_metrics["by_depth"].items()}
    id_valid = all(
        id_by_depth.get(depth, 0.0) >= args.id_threshold
        for depth in range(1, args.train_max_depth + 1)
    )

    e1_depth = {}
    for depth, examples in sorted(depth_examples.items()):
        metrics = evaluate(
            model, loader(examples, vocab, args.batch_size), device,
            collect_predictions=True,
        )
        pop_predictions(metrics, predictions, condition="depth", num_loops=None)
        e1_depth[str(depth)] = metrics
    depth_accuracy = {int(depth): float(metrics["accuracy"]) for depth, metrics in e1_depth.items()}

    results: dict[str, object] = {
        "config": config.to_dict(),
        "protocol": {
            "generator_version": GENERATOR_VERSION,
            "dataset_seed": args.seed,
            "train_max_depth": args.train_max_depth,
            "ood_depths": sorted(set(args.ood_depths)),
            "test_examples_per_cell": cell_count,
            "matched_total_events": args.matched_total_events,
            "matched_depths": args.matched_depths,
            "distractor_counts": args.distractor_counts,
            "random_loops": args.random_loops,
            "model_condition": condition_name,
        },
        "seed": args.seed,
        "parameters": count_parameters(model),
        "target_parameters": args.target_parameters,
        "matched_parameter_count": matched_parameter_count,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "training": {
            "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
            "scheduler": args.scheduler, "batch_size": args.batch_size,
            "epochs": args.epochs, "grad_clip": 1.0, "training_seconds": training_seconds,
            "steps_per_epoch": len(train_loader),
            "total_optimizer_steps": len(train_loader) * args.epochs,
        },
        "runtime": {
            "device": str(device),
            "max_sequence_length": max(len(vocab.encode(example.text)) for example in train_examples),
            "estimated_forward_flops_at_max_train_length": estimate_forward_flops(
                config, max(len(vocab.encode(example.text)) for example in train_examples)
            ),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "train_history": history,
        "E0_id": id_metrics,
        "E0_valid": id_valid,
        "E1_depth": e1_depth,
        "analysis": {
            "ood_degradation_slope": ood_degradation_slope(depth_accuracy, args.train_max_depth),
            "ood_conclusions_allowed": id_valid,
            "minimum_id_depth_accuracy": min(id_by_depth.values()),
        },
    }

    e3 = {}
    for cell in matched_length_grid(args.matched_total_events, args.matched_depths):
        generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 3_000 + cell["target_depth"]
        )
        examples = generator.generate_unique(
            cell_count, seen=seen, target_depth=cell["target_depth"],
            num_distractors=cell["num_distractors"], num_entities=6,
        )
        metrics = evaluate(model, loader(examples, vocab, args.batch_size), device)
        e3[str(cell["target_depth"])] = {**cell, **metrics}
    results["E3_matched_length"] = e3

    e4 = {}
    for distractors in args.distractor_counts:
        generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 4_000 + distractors
        )
        examples = generator.generate_unique(
            cell_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=distractors, num_entities=6,
        )
        e4[str(distractors)] = evaluate(model, loader(examples, vocab, args.batch_size), device)
    results["E4_distractors"] = e4

    linguistic_generator = StateTrackingGenerator(
        num_locations=args.num_locations, seed=args.seed + 5_000
    )
    linguistic_examples = linguistic_generator.generate_unique(
        cell_count, seen=seen, target_depth=args.train_max_depth,
        num_distractors=8, num_entities=6,
        linguistic_variation=True, template_split="ood",
    )
    results["E5_linguistic_ood"] = evaluate(
        model, loader(linguistic_examples, vocab, args.batch_size), device
    )
    lexical_generator = StateTrackingGenerator(
        num_locations=args.num_locations, seed=args.seed + 5_500,
        entity_names=OOD_OBJECTS, location_names=OOD_LOCATIONS,
    )
    lexical_examples = lexical_generator.generate_unique(
        cell_count, seen=seen, target_depth=args.train_max_depth,
        num_distractors=8, num_entities=6,
    )
    results["E5_lexical_ood"] = evaluate(
        model, loader(lexical_examples, vocab, args.batch_size), device
    )

    if args.architecture == "recurrent":
        matrix = loop_depth_sweep(
            model,
            {depth: loader(examples, vocab, args.batch_size)
             for depth, examples in depth_examples.items()},
            sorted(set(args.loop_counts)),
            device,
        )
        results["E2_loop_depth_matrix"] = matrix
        results["E2_best_loop_by_depth"] = {
            str(depth): loops for depth, loops in best_loop_by_depth(matrix).items()
        }
        probe_count = 16 if args.smoke else max(500, cell_count)
        probe_train = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 6_000
        ).generate_unique(
            probe_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=8, num_entities=6,
        )
        probe_test = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 7_000
        ).generate_unique(
            probe_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=8, num_entities=6,
        )
        train_features, train_labels = collect_loop_cls_states(
            model, loader(probe_train, vocab, args.batch_size), device,
            num_loops=args.train_loops,
        )
        test_features, test_labels = collect_loop_cls_states(
            model, loader(probe_test, vocab, args.batch_size), device,
            num_loops=args.train_loops,
        )
        results["E6_final_state_probe"] = fit_loop_probes(
            train_features, train_labels, test_features, test_labels,
            num_locations=config.num_locations, epochs=5 if args.smoke else 100,
        )
        results["E6_trajectory_readout_matrix"] = trajectory_readout_matrix(
            model, loader(probe_test, vocab, args.batch_size), device,
            num_loops=args.train_loops,
        )

    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_path = output.with_name(output.stem + "_predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output), "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path), "parameters": results["parameters"],
        "E0_accuracy": id_metrics["accuracy"], "E0_valid": id_valid,
    }, indent=2))


if __name__ == "__main__":
    main()
