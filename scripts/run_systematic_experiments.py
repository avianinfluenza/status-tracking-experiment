#!/usr/bin/env python3
"""Run natural-language baselines or the atomic Fan state-tracking control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    ATOMIC_SERIALIZATION_VERSION,
    AtomicVocabulary,
    DeterministicOnlineBatchStream,
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
    progress_bar,
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
from src.runtime import maybe_compile_model, uncompiled_state_dict


def loader(examples, vocab, batch_size, shuffle=False, events_per_loop=None, seed=0):
    dataset = StateTrackingDataset(examples, vocab)
    if events_per_loop is not None:
        groups = {}
        for index, example in enumerate(examples):
            loops = math.ceil(example.total_events / events_per_loop)
            groups.setdefault(loops, []).append(index)
        rng = random.Random(seed)
        batches = []
        for indices in groups.values():
            if shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[start:start + batch_size]
                for start in range(0, len(indices), batch_size)
            )
        if shuffle:
            rng.shuffle(batches)
        return DataLoader(
            dataset,
            batch_sampler=batches,
            collate_fn=partial(collate_examples, pad_id=vocab.pad_id),
        )
    return DataLoader(
        dataset,
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


def online_max_sequence_length(args: argparse.Namespace, vocab) -> int:
    if args.input_format == "atomic":
        return args.num_entities + args.curriculum_max_events + 1
    target_depth = min(args.train_max_depth, args.curriculum_max_events)
    generator = StateTrackingGenerator(num_locations=args.num_locations, seed=args.seed + 99_000)
    example = generator.generate(
        target_depth=target_depth,
        num_distractors=args.curriculum_max_events - target_depth,
        num_entities=args.num_entities,
        linguistic_variation=False,
    )
    return len(vocab.encode_example(example))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("standard", "recurrent", "untied", "fan-recurrent"),
        default="recurrent",
    )
    parser.add_argument("--input-format", choices=("natural", "atomic"), default="natural")
    parser.add_argument(
        "--position-encoding", choices=("none", "sinusoidal"), default="sinusoidal"
    )
    parser.add_argument("--loop-conditioning", choices=("none", "learned"), default="none")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-blocks", type=int, default=1)
    parser.add_argument("--random-loops", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-examples", type=int, default=100_000)
    parser.add_argument("--online-training", action="store_true")
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-min-events", type=int, default=1)
    parser.add_argument("--curriculum-max-events", type=int, default=24)
    parser.add_argument("--curriculum-steps-per-length", type=int, default=1_000)
    parser.add_argument("--events-per-loop", type=float)
    parser.add_argument("--test-examples-per-cell", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-locations", type=int, choices=(8, 12), default=8)
    parser.add_argument("--num-entities", type=int, choices=range(2, 9))
    parser.add_argument("--target-parameters", type=int)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--train-loops", type=int, default=6)
    parser.add_argument("--train-max-depth", type=int, default=8)
    parser.add_argument("--train-max-distractors", type=int, default=16)
    parser.add_argument("--deep-supervision-weight", type=float, default=0.0)
    parser.add_argument("--ood-depths", type=int, nargs="+", default=[10, 12, 16, 20, 24, 32])
    parser.add_argument("--loop-counts", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16, 24])
    parser.add_argument("--matched-total-events", type=int, default=24)
    parser.add_argument("--matched-depths", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20])
    parser.add_argument("--distractor-counts", type=int, nargs="+", default=[0, 4, 8, 16, 32, 64])
    parser.add_argument("--id-threshold", type=float, default=0.95)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--condition-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.epochs = 1
        args.train_examples = 64
        args.train_steps = 2
        args.test_examples_per_cell = 8
        args.batch_size = 16
        args.d_model = 32
        args.num_layers = 2
        args.train_loops = 2
        args.train_max_depth = 2
        args.curriculum_max_events = min(
            args.curriculum_max_events,
            args.train_max_depth + args.train_max_distractors,
        )
        args.ood_depths = [4, 8]
        args.loop_counts = [1, 2, 4, 8]
        args.matched_total_events = 8
        args.matched_depths = [2, 4, 8]
        args.distractor_counts = [0, 4, 8]
    show_progress = not args.no_progress

    if any(depth <= args.train_max_depth for depth in args.ood_depths):
        raise ValueError("all OOD depths must be strictly above train_max_depth")
    if any(depth > args.matched_total_events for depth in args.matched_depths):
        raise ValueError("matched depths cannot exceed matched_total_events")
    if args.deep_supervision_weight != 0.0:
        raise ValueError("systematic Fan currently supports final-answer-only supervision")
    if args.events_per_loop is not None and args.events_per_loop <= 0:
        raise ValueError("--events-per-loop must be positive")
    if args.events_per_loop is not None and args.architecture not in ("recurrent", "fan-recurrent"):
        raise ValueError("--events-per-loop requires a recurrent architecture")
    if args.events_per_loop is not None and not args.online_training:
        raise ValueError("--events-per-loop is currently available only with --online-training")
    if args.architecture == "fan-recurrent":
        if args.input_format != "atomic":
            raise ValueError("fan-recurrent systematic control requires --input-format atomic")
        if args.position_encoding != "none":
            raise ValueError("fan-recurrent systematic control requires NoPE")
        if not args.online_training:
            raise ValueError("fan-recurrent systematic control requires --online-training")
        if args.events_per_loop is None:
            raise ValueError("fan-recurrent systematic control requires --events-per-loop")
        if args.num_entities is None:
            raise ValueError("atomic online training requires an explicit --num-entities")
    elif args.input_format != "natural" or args.position_encoding != "sinusoidal":
        raise ValueError("existing systematic architectures retain natural input and sinusoidal PE")
    if not 1 <= args.curriculum_min_events <= args.curriculum_max_events:
        raise ValueError("event curriculum must satisfy 1 <= min <= max")
    if args.curriculum_max_events > args.train_max_depth + args.train_max_distractors:
        raise ValueError("curriculum max events exceeds the configured training support")
    if args.curriculum_steps_per_length < 1 or args.train_steps < 1:
        raise ValueError("online step counts must be positive")
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
        elif args.online_training and args.events_per_loop is not None:
            condition_name = f"{args.input_format}_online_k_per_event"
        elif args.online_training:
            condition_name = f"{args.input_format}_online_curriculum"
        else:
            condition_name = "main_compute_matched"

    seen: set[str] = set()
    eval_num_entities = args.num_entities if args.num_entities is not None else 6
    if args.online_training and args.num_entities is None:
        args.num_entities = eval_num_entities
    if args.input_format == "atomic":
        assert args.num_entities is not None
        vocab = AtomicVocabulary(
            num_entities=args.num_entities,
            num_locations=args.num_locations,
        )
    else:
        vocab = TokenVocabulary.from_schema(num_locations=args.num_locations)

    train_examples = []
    if not args.online_training:
        train_generator = StateTrackingGenerator(num_locations=args.num_locations, seed=args.seed)
        while len(train_examples) < args.train_examples:
            example = train_generator.generate(
                target_depth=train_generator.rng.randint(1, args.train_max_depth),
                num_distractors=train_generator.rng.randint(0, args.train_max_distractors),
                num_entities=(
                    args.num_entities
                    if args.num_entities is not None
                    else train_generator.rng.randint(2, 8)
                ),
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
            num_entities=eval_num_entities,
        )
    id_examples = [example for depth in range(1, args.train_max_depth + 1)
                   for example in depth_examples[depth]]

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
        position_encoding=args.position_encoding,
        readout="last" if args.input_format == "atomic" else "cls",
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
    model = maybe_compile_model(StateTrackingTransformer(config).to(device), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.train_steps if args.online_training else args.epochs,
        )
        if args.scheduler == "cosine" else None
    )
    history = []
    training_start = time.perf_counter()
    if args.online_training:
        train_loader = DeterministicOnlineBatchStream(
            num_steps=args.train_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            min_events=args.curriculum_min_events,
            max_events=args.curriculum_max_events,
            steps_per_length=args.curriculum_steps_per_length,
            num_entities=args.num_entities,
            num_locations=args.num_locations,
            max_target_depth=args.train_max_depth,
            max_distractors=args.train_max_distractors,
            vocab=vocab,
        )
        metrics = train_epoch(
            model, train_loader, optimizer, device,
            events_per_loop=args.events_per_loop,
            scheduler=scheduler,
            show_progress=show_progress,
            progress_desc=f"{args.architecture} seed{args.seed} train",
            progress_position=0,
        )
        history.append({
            "optimizer_steps": args.train_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics.__dict__,
        })
    else:
        train_loader = loader(train_examples, vocab, args.batch_size, shuffle=True)
        epoch_progress = None
        epoch_iterable = range(1, args.epochs + 1)
        if show_progress:
            epoch_progress = progress_bar(
                epoch_iterable,
                total=args.epochs,
                desc=f"{args.architecture} seed{args.seed}",
                leave=True,
                position=0,
            )
            epoch_iterable = epoch_progress
        try:
            for epoch in epoch_iterable:
                metrics = train_epoch(
                    model, train_loader, optimizer, device,
                    random_loop_range=(max(1, args.train_loops // 2), args.train_loops)
                    if args.random_loops else None,
                    show_progress=show_progress,
                    progress_desc=f"epoch {epoch}",
                    progress_position=1,
                )
                history.append({
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **metrics.__dict__,
                })
                if scheduler is not None:
                    scheduler.step()
                if epoch_progress is not None:
                    epoch_progress.set_postfix(
                        loss=f"{metrics.loss:.4f}",
                        accuracy=f"{metrics.accuracy:.4f}",
                    )
        finally:
            if epoch_progress is not None:
                epoch_progress.close()
    training_seconds = time.perf_counter() - training_start

    output = args.output or ROOT / "runs" / "systematic" / f"systematic_{args.architecture}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".pt")
    torch.save({
        "model_config": config.to_dict(),
        "model_state": uncompiled_state_dict(model),
        "vocabulary": vocab.itos,
        "seed": args.seed,
        "generator_version": GENERATOR_VERSION,
        "serialization_version": (
            ATOMIC_SERIALIZATION_VERSION if args.input_format == "atomic" else "natural-v1"
        ),
    }, checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    event_schedule = args.events_per_loop
    max_train_sequence_length = (
        online_max_sequence_length(args, vocab)
        if args.online_training
        else max(len(vocab.encode_example(example)) for example in train_examples)
    )

    def evaluation_loader(examples):
        return loader(
            examples,
            vocab,
            args.batch_size,
            events_per_loop=event_schedule,
            seed=args.seed,
        )

    def evaluate_examples(
        examples,
        *,
        desc,
        num_loops=None,
        collect_predictions=False,
    ):
        return evaluate(
            model,
            evaluation_loader(examples),
            device,
            num_loops=num_loops,
            events_per_loop=event_schedule,
            collect_predictions=collect_predictions,
            show_progress=show_progress,
            progress_desc=desc,
            progress_position=0,
        )

    predictions: list[dict[str, object]] = []
    id_metrics = evaluate_examples(id_examples, desc="E0 id", collect_predictions=True)
    pop_predictions(id_metrics, predictions, condition="id", num_loops=None)
    id_by_depth = {int(depth): float(accuracy) for depth, accuracy in id_metrics["by_depth"].items()}
    id_valid = all(
        id_by_depth.get(depth, 0.0) >= args.id_threshold
        for depth in range(1, args.train_max_depth + 1)
    )

    e1_depth = {}
    for depth, examples in sorted(depth_examples.items()):
        metrics = evaluate_examples(
            examples,
            desc=f"E1 depth {depth}",
            collect_predictions=True,
        )
        pop_predictions(metrics, predictions, condition="depth", num_loops=None)
        e1_depth[str(depth)] = metrics
    depth_accuracy = {int(depth): float(metrics["accuracy"]) for depth, metrics in e1_depth.items()}

    results: dict[str, object] = {
        "config": config.to_dict(),
        "protocol": {
            "generator_version": GENERATOR_VERSION,
            "serialization_version": (
                ATOMIC_SERIALIZATION_VERSION if args.input_format == "atomic" else "natural-v1"
            ),
            "dataset_seed": args.seed,
            "num_entities": eval_num_entities,
            "num_locations": args.num_locations,
            "train_max_depth": args.train_max_depth,
            "train_max_distractors": args.train_max_distractors,
            "ood_depths": sorted(set(args.ood_depths)),
            "test_examples_per_cell": cell_count,
            "matched_total_events": args.matched_total_events,
            "matched_depths": args.matched_depths,
            "distractor_counts": args.distractor_counts,
            "random_loops": args.random_loops,
            "input_format": args.input_format,
            "events_per_loop": event_schedule,
            "hidden_update": (
                "h_0=0; h_k=F_theta(h_{k-1}+embed(x))"
                if args.architecture == "fan-recurrent" else None
            ),
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
            "epochs": None if args.online_training else args.epochs,
            "train_steps": args.train_steps if args.online_training else None,
            "online_training": args.online_training,
            "curriculum_min_events": (
                args.curriculum_min_events if args.online_training else None
            ),
            "curriculum_max_events": (
                args.curriculum_max_events if args.online_training else None
            ),
            "curriculum_steps_per_length": (
                args.curriculum_steps_per_length if args.online_training else None
            ),
            "grad_clip": 1.0, "training_seconds": training_seconds,
            "steps_per_epoch": len(train_loader),
            "total_optimizer_steps": (
                args.train_steps if args.online_training else len(train_loader) * args.epochs
            ),
        },
        "runtime": {
            "device": str(device),
            "max_sequence_length": max_train_sequence_length,
            "estimated_forward_flops_at_max_train_length": estimate_forward_flops(
                config,
                max_train_sequence_length,
                num_loops=(
                    math.ceil(args.curriculum_max_events / args.events_per_loop)
                    if event_schedule is not None else None
                ),
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
            num_distractors=cell["num_distractors"], num_entities=eval_num_entities,
        )
        metrics = evaluate_examples(examples, desc=f"E3 depth {cell['target_depth']}")
        e3[str(cell["target_depth"])] = {**cell, **metrics}
    results["E3_matched_length"] = e3

    e4 = {}
    for distractors in args.distractor_counts:
        generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 4_000 + distractors
        )
        examples = generator.generate_unique(
            cell_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=distractors, num_entities=eval_num_entities,
        )
        e4[str(distractors)] = evaluate_examples(
            examples,
            desc=f"E4 distractors {distractors}",
        )
    results["E4_distractors"] = e4

    if args.input_format == "natural":
        linguistic_generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 5_000
        )
        linguistic_examples = linguistic_generator.generate_unique(
            cell_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=8, num_entities=eval_num_entities,
            linguistic_variation=True, template_split="ood",
        )
        results["E5_linguistic_ood"] = evaluate_examples(
            linguistic_examples,
            desc="E5 linguistic",
        )
        lexical_generator = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 5_500,
            entity_names=OOD_OBJECTS, location_names=OOD_LOCATIONS,
        )
        lexical_examples = lexical_generator.generate_unique(
            cell_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=8, num_entities=eval_num_entities,
        )
        results["E5_lexical_ood"] = evaluate_examples(
            lexical_examples,
            desc="E5 lexical",
        )
    else:
        results["E5_not_applicable"] = {
            "reason": "atomic serialization removes rendered linguistic and lexical forms"
        }

    if args.architecture == "recurrent":
        matrix = loop_depth_sweep(
            model,
            {depth: loader(examples, vocab, args.batch_size)
             for depth, examples in depth_examples.items()},
            sorted(set(args.loop_counts)),
            device,
            show_progress=show_progress,
            progress_position=0,
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
            num_distractors=8, num_entities=eval_num_entities,
        )
        probe_test = StateTrackingGenerator(
            num_locations=args.num_locations, seed=args.seed + 7_000
        ).generate_unique(
            probe_count, seen=seen, target_depth=args.train_max_depth,
            num_distractors=8, num_entities=eval_num_entities,
        )
        train_features, train_labels = collect_loop_cls_states(
            model, loader(probe_train, vocab, args.batch_size), device,
            num_loops=args.train_loops,
            show_progress=show_progress,
            progress_desc="E6 probe train features",
            progress_position=0,
        )
        test_features, test_labels = collect_loop_cls_states(
            model, loader(probe_test, vocab, args.batch_size), device,
            num_loops=args.train_loops,
            show_progress=show_progress,
            progress_desc="E6 probe test features",
            progress_position=0,
        )
        results["E6_final_state_probe"] = fit_loop_probes(
            train_features, train_labels, test_features, test_labels,
            num_locations=config.num_locations, epochs=5 if args.smoke else 100,
        )
        results["E6_trajectory_readout_matrix"] = trajectory_readout_matrix(
            model, loader(probe_test, vocab, args.batch_size), device,
            num_loops=args.train_loops,
            show_progress=show_progress,
            progress_desc="E6 trajectory readout",
            progress_position=0,
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
