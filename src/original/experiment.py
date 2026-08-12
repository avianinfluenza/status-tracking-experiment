"""Train and evaluate the three models in the team's original research scope."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.collate import BallSwapDataset, collate_fn
from .data import (
    ExplicitCoTDataset,
    RowsDataset,
    collate_cot,
    group_rows_by_swap_count,
    inject_noop_swaps,
)
from .model import (
    DirectTransformer,
    ExplicitCoTTransformer,
    OriginalModel,
    OriginalModelConfig,
    RecurrentR0Transformer,
    RecurrentTransformer,
    build_model,
    count_parameters,
)


ROOT = Path(__file__).resolve().parents[2]
EVAL_SPLITS = ("id_test", "ood_x4", "ood_x8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _subset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    return Subset(dataset, range(min(maximum, len(dataset))))


def make_loader(
    path: Path,
    architecture: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
) -> DataLoader:
    if architecture == "cot":
        dataset: Dataset = ExplicitCoTDataset(str(path))
        collator = collate_cot
    else:
        dataset = BallSwapDataset(path)
        collator = collate_fn
    return DataLoader(
        _subset(dataset, max_samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )


def move_tensors(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _supports_loop_override(model: OriginalModel) -> bool:
    return isinstance(model, (RecurrentTransformer, RecurrentR0Transformer))


def train_epoch(
    model: OriginalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    deep_supervision_weight: float = 0.0,
    random_loop_range: tuple[int, int] | None = None,
) -> float:
    model.train()
    weighted_loss = 0.0
    target_count = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        sampled_loops = None
        if random_loop_range is not None:
            if not isinstance(model, RecurrentR0Transformer):
                raise ValueError("random-loop training is available only for recurrent-r0")
            sampled_loops = random.randint(*random_loop_range)
        if isinstance(model, ExplicitCoTTransformer):
            logits = model(batch["input_ids"], batch["attention_mask"])
            targets = batch["lm_labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        elif _supports_loop_override(model) and deep_supervision_weight > 0.0:
            loop_logits = model.forward_all_loops(
                batch["input_ids"],
                batch["attn_mask"],
                batch["slot_pos"],
                num_loops=sampled_loops,
            )
            targets = batch["labels"]
            final_logits = loop_logits[:, -1]
            final_loss = F.cross_entropy(
                final_logits.reshape(-1, final_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if loop_logits.shape[1] > 1:
                intermediate = loop_logits[:, :-1]
                repeated_targets = targets[:, None].expand(-1, intermediate.shape[1], -1)
                intermediate_loss = F.cross_entropy(
                    intermediate.reshape(-1, intermediate.shape[-1]),
                    repeated_targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = final_loss + deep_supervision_weight * intermediate_loss
            else:
                loss = final_loss
        else:
            if sampled_loops is not None:
                logits = model(
                    batch["input_ids"],
                    batch["attn_mask"],
                    batch["slot_pos"],
                    num_loops=sampled_loops,
                )
            else:
                logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
            targets = batch["labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = int((targets != -100).sum().item())
        weighted_loss += float(loss.item()) * count
        target_count += count
    return weighted_loss / max(target_count, 1)


def _accumulate_metrics(
    predictions: Tensor,
    labels: Tensor,
    n_swaps: Tensor,
    totals: dict[str, object],
) -> None:
    valid = labels != -100
    correct = (predictions == labels) & valid
    exact = (correct | ~valid).all(dim=1)
    totals["slot_correct"] = int(totals["slot_correct"]) + int(correct.sum().item())
    totals["slot_total"] = int(totals["slot_total"]) + int(valid.sum().item())
    totals["exact_correct"] = int(totals["exact_correct"]) + int(exact.sum().item())
    totals["samples"] = int(totals["samples"]) + labels.shape[0]
    by_swaps = totals["by_swaps"]
    assert isinstance(by_swaps, defaultdict)
    for length, hit in zip(n_swaps.tolist(), exact.tolist(), strict=True):
        by_swaps[int(length)][0] += int(hit)
        by_swaps[int(length)][1] += 1


def _finish_metrics(totals: dict[str, object]) -> dict[str, object]:
    by_swaps = totals["by_swaps"]
    assert isinstance(by_swaps, defaultdict)
    result: dict[str, object] = {
        "slot_accuracy": int(totals["slot_correct"]) / max(int(totals["slot_total"]), 1),
        "exact_match": int(totals["exact_correct"]) / max(int(totals["samples"]), 1),
        "n_samples": int(totals["samples"]),
        "by_swaps": {
            str(length): {"exact_match": hits / count, "correct": hits, "total": count}
            for length, (hits, count) in sorted(by_swaps.items())
        },
    }
    return result


def _empty_totals() -> dict[str, object]:
    return {
        "slot_correct": 0,
        "slot_total": 0,
        "exact_correct": 0,
        "samples": 0,
        "by_swaps": defaultdict(lambda: [0, 0]),
    }


@torch.inference_mode()
def evaluate_classifier(
    model: DirectTransformer | RecurrentTransformer | RecurrentR0Transformer,
    loader: DataLoader,
    device: torch.device,
    *,
    adaptive_kl: bool,
    num_loops: int | None = None,
) -> dict[str, object]:
    model.eval()
    if num_loops is not None and not _supports_loop_override(model):
        raise ValueError("num_loops evaluation is available only for recurrent models")
    totals = _empty_totals()
    step_sum = 0
    halt_sum = 0
    final_kl_sum = 0.0
    final_kl_count = 0
    final_update_sum = 0.0
    final_update_count = 0
    final_confidence_sum = 0.0
    final_confidence_count = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        if adaptive_kl:
            if not _supports_loop_override(model):
                raise ValueError("adaptive halting is available only for recurrent models")
            logits, diagnostics = model.forward_adaptive(
                batch["input_ids"],
                batch["attn_mask"],
                batch["slot_pos"],
                max_loops=num_loops,
            )
            step_sum += int(diagnostics["steps_taken"].sum().item())
            halt_sum += int(diagnostics["halted"].sum().item())
            indices = (diagnostics["steps_taken"] - 1).unsqueeze(1)
            final_kl = diagnostics["symmetric_kl"].gather(1, indices).squeeze(1)
            finite = torch.isfinite(final_kl)
            final_kl_sum += float(final_kl[finite].sum().item())
            final_kl_count += int(finite.sum().item())
            if "update_ratio" in diagnostics:
                final_update = diagnostics["update_ratio"].gather(1, indices).squeeze(1)
                finite_update = torch.isfinite(final_update)
                final_update_sum += float(final_update[finite_update].sum().item())
                final_update_count += int(finite_update.sum().item())
            if "confidence" in diagnostics:
                final_confidence = diagnostics["confidence"].gather(1, indices).squeeze(1)
                finite_confidence = torch.isfinite(final_confidence)
                final_confidence_sum += float(final_confidence[finite_confidence].sum().item())
                final_confidence_count += int(finite_confidence.sum().item())
        else:
            if num_loops is not None:
                logits = model(
                    batch["input_ids"],
                    batch["attn_mask"],
                    batch["slot_pos"],
                    num_loops=num_loops,
                )
            else:
                logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
        _accumulate_metrics(logits.argmax(-1), batch["labels"], batch["n_swaps"], totals)
    metrics = _finish_metrics(totals)
    if num_loops is not None:
        metrics["num_loops"] = num_loops
    if adaptive_kl:
        samples = int(metrics["n_samples"])
        metrics.update(
            {
                "average_loops": step_sum / max(samples, 1),
                "halt_rate": halt_sum / max(samples, 1),
                "final_symmetric_kl": final_kl_sum / max(final_kl_count, 1),
                "halting_signal": (
                    "output_symmetric_kl_confidence_update_ratio"
                    if isinstance(model, RecurrentR0Transformer)
                    else "output_symmetric_kl_only"
                ),
            }
        )
        if final_update_count:
            metrics["final_update_ratio"] = final_update_sum / final_update_count
        if final_confidence_count:
            metrics["final_confidence"] = final_confidence_sum / final_confidence_count
    return metrics


def _rows_from_dataset(dataset: Dataset) -> list[dict[str, object]]:
    if isinstance(dataset, Subset):
        return [dataset.dataset[index] for index in dataset.indices]  # type: ignore[index]
    return [dataset[index] for index in range(len(dataset))]  # type: ignore[index]


@torch.inference_mode()
def evaluate_cot(
    model: ExplicitCoTTransformer,
    dataset: Dataset,
    *,
    generation_batch_size: int,
) -> dict[str, object]:
    """Leak-free evaluation by autoregressively generating all trace states."""

    model.eval()
    totals = _empty_totals()
    groups = group_rows_by_swap_count(_rows_from_dataset(dataset))
    for n_swaps, rows in sorted(groups.items()):
        for start in range(0, len(rows), generation_batch_size):
            chunk = rows[start : start + generation_batch_size]
            predictions = model.generate_states(chunk).cpu()
            labels = torch.tensor([row["labels"] for row in chunk], dtype=torch.long)
            lengths = torch.full((len(chunk),), n_swaps, dtype=torch.long)
            _accumulate_metrics(predictions, labels, lengths, totals)
    metrics = _finish_metrics(totals)
    metrics["evaluation_mode"] = "autoregressive_trace_generation"
    return metrics


def save_checkpoint(path: Path, model: OriginalModel, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("direct", "cot", "recurrent", "recurrent-r0"),
        required=True,
    )
    parser.add_argument("--position-encoding", choices=("sinusoidal", "rope"), default="sinusoidal")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-loops", type=int, default=6)
    parser.add_argument("--classifier-dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adaptive-kl-eval", action="store_true")
    parser.add_argument("--kl-threshold", type=float, default=1e-3)
    parser.add_argument("--adaptive-update-threshold", type=float, default=1e9)
    parser.add_argument("--adaptive-min-confidence", type=float, default=0.0)
    parser.add_argument("--min-loops", type=int, default=2)
    parser.add_argument("--halting-patience", type=int, default=1)
    parser.add_argument("--loop-conditioning", choices=("none", "learned"), default="none")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-blocks", type=int, default=1)
    parser.add_argument("--max-loop-embeddings", type=int, default=64)
    parser.add_argument("--random-loops", action="store_true")
    parser.add_argument("--random-min-loops", type=int, default=None)
    parser.add_argument("--random-max-loops", type=int, default=None)
    parser.add_argument("--eval-loop-counts", type=int, nargs="+", default=None)
    parser.add_argument(
        "--deep-supervision-weight",
        type=float,
        default=0.0,
        help="optional recurrent ablation: final-state CE on every non-final loop",
    )
    parser.add_argument(
        "--noop-eval-ratio",
        type=float,
        default=0.0,
        help="also evaluate ID rows after inserting this fraction of self-swaps",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    recurrent_architectures = {"recurrent", "recurrent-r0"}
    if args.adaptive_kl_eval and args.architecture not in recurrent_architectures:
        raise ValueError("--adaptive-kl-eval requires a recurrent architecture")
    if args.deep_supervision_weight < 0.0:
        raise ValueError("--deep-supervision-weight must be non-negative")
    if args.deep_supervision_weight > 0.0 and args.architecture not in recurrent_architectures:
        raise ValueError("deep supervision is available only for the recurrent model")
    advanced_r0_requested = (
        args.loop_conditioning != "none"
        or args.residual_scale != 1.0
        or args.recurrent_blocks != 1
        or args.random_loops
    )
    if advanced_r0_requested and args.architecture != "recurrent-r0":
        raise ValueError("loop-conditioning, residual-scale, recurrent-blocks, and random-loops require recurrent-r0")
    if args.eval_loop_counts and args.architecture not in recurrent_architectures:
        raise ValueError("--eval-loop-counts requires a recurrent architecture")
    if args.noop_eval_ratio < 0.0:
        raise ValueError("--noop-eval-ratio must be non-negative")
    if args.random_loops:
        random_min = args.random_min_loops if args.random_min_loops is not None else max(1, args.num_loops // 2)
        random_max = args.random_max_loops if args.random_max_loops is not None else args.num_loops
        if not 1 <= random_min <= random_max:
            raise ValueError("random loop range must satisfy 1 <= min <= max")
    else:
        random_min = random_max = None
    if args.smoke:
        args.epochs = 1
        args.d_model = 32
        args.n_heads = 4
        args.d_ff = 64
        args.num_layers = 2
        args.num_loops = 2
        args.min_loops = min(args.min_loops, args.num_loops)
        if args.random_loops:
            random_min = 1
            random_max = args.num_loops
        args.batch_size = 8
        args.eval_batch_size = 4
        args.max_train_samples = 16
        args.max_eval_samples = 4

    seed_everything(args.seed)
    device = resolve_device(args.device)
    config = OriginalModelConfig(
        architecture=args.architecture,
        position_encoding=args.position_encoding,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_loops=args.num_loops,
        classifier_dim=args.classifier_dim,
        kl_threshold=args.kl_threshold,
        min_loops=args.min_loops,
        halting_patience=args.halting_patience,
        loop_conditioning=args.loop_conditioning,
        residual_scale=args.residual_scale,
        recurrent_blocks=args.recurrent_blocks,
        max_loop_embeddings=args.max_loop_embeddings,
        adaptive_update_threshold=args.adaptive_update_threshold,
        adaptive_min_confidence=args.adaptive_min_confidence,
    )
    model = build_model(config).to(device)
    train_loader = make_loader(
        args.data_dir / "train.jsonl",
        args.architecture,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        max_samples=args.max_train_samples,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    started = time.perf_counter()
    losses = []
    for _epoch in range(1, args.epochs + 1):
        losses.append(
            train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.grad_clip,
                deep_supervision_weight=args.deep_supervision_weight,
                random_loop_range=(random_min, random_max) if args.random_loops else None,
            )
        )
    training_seconds = time.perf_counter() - started

    split_metrics: dict[str, object] = {}
    for split in EVAL_SPLITS:
        path = args.data_dir / f"{split}.jsonl"
        if isinstance(model, ExplicitCoTTransformer):
            dataset = _subset(ExplicitCoTDataset(str(path)), args.max_eval_samples)
            metrics = evaluate_cot(model, dataset, generation_batch_size=args.eval_batch_size)
        else:
            loader = make_loader(
                path,
                args.architecture,
                batch_size=args.eval_batch_size,
                shuffle=False,
                seed=args.seed,
                max_samples=args.max_eval_samples,
            )
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                adaptive_kl=args.adaptive_kl_eval,
            )
            if args.eval_loop_counts:
                metrics["loop_sweep"] = {
                    str(loop_count): evaluate_classifier(
                        model,
                        loader,
                        device,
                        adaptive_kl=False,
                        num_loops=loop_count,
                    )
                    for loop_count in args.eval_loop_counts
                }
        split_metrics[split] = metrics

    if args.noop_eval_ratio > 0.0:
        base_dataset = _subset(
            BallSwapDataset(args.data_dir / "id_test.jsonl"),
            args.max_eval_samples,
        )
        noisy_rows = inject_noop_swaps(
            _rows_from_dataset(base_dataset),
            ratio=args.noop_eval_ratio,
            seed=args.seed + 50_000,
        )
        split_name = f"id_test_noop_{args.noop_eval_ratio:g}"
        if isinstance(model, ExplicitCoTTransformer):
            split_metrics[split_name] = evaluate_cot(
                model,
                RowsDataset(noisy_rows),
                generation_batch_size=args.eval_batch_size,
            )
        else:
            noisy_loader = DataLoader(
                RowsDataset(noisy_rows),
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            split_metrics[split_name] = evaluate_classifier(
                model,
                noisy_loader,
                device,
                adaptive_kl=args.adaptive_kl_eval,
            )
            if args.eval_loop_counts:
                split_metrics[split_name]["loop_sweep"] = {
                    str(loop_count): evaluate_classifier(
                        model,
                        noisy_loader,
                        device,
                        adaptive_kl=False,
                        num_loops=loop_count,
                    )
                    for loop_count in args.eval_loop_counts
                }

    if args.run_name:
        run_name = args.run_name
    else:
        name_parts = [args.architecture, args.position_encoding, f"seed{args.seed}"]
        if args.deep_supervision_weight > 0.0:
            name_parts.append(f"ds{args.deep_supervision_weight:g}")
        if args.noop_eval_ratio > 0.0:
            name_parts.append(f"noop{args.noop_eval_ratio:g}")
        if args.architecture == "recurrent-r0":
            if args.loop_conditioning != "none":
                name_parts.append(f"loop{args.loop_conditioning}")
            if args.residual_scale != 1.0:
                name_parts.append(f"rs{args.residual_scale:g}")
            if args.recurrent_blocks != 1:
                name_parts.append(f"blocks{args.recurrent_blocks}")
            if args.random_loops:
                name_parts.append("randomloops")
        if args.adaptive_kl_eval:
            name_parts.append("adaptive")
        run_name = "-".join(name_parts)
    result = {
        "track": "original_team_plan",
        "run_name": run_name,
        "config": config.to_dict(),
        "parameters": count_parameters(model),
        "seed": args.seed,
        "training_loss": losses,
        "training_seconds": training_seconds,
        "ablations": {
            "deep_supervision_weight": args.deep_supervision_weight,
            "noop_eval_ratio": args.noop_eval_ratio,
            "loop_conditioning": args.loop_conditioning,
            "residual_scale": args.residual_scale,
            "recurrent_blocks": args.recurrent_blocks,
            "random_loops": args.random_loops,
            "random_loop_range": [random_min, random_max] if args.random_loops else None,
            "eval_loop_counts": args.eval_loop_counts,
            "adaptive_update_threshold": args.adaptive_update_threshold,
            "adaptive_min_confidence": args.adaptive_min_confidence,
        },
        "splits": split_metrics,
    }
    save_checkpoint(args.output_dir / f"{run_name}.pt", model, args.epochs)
    write_json(args.output_dir / f"{run_name}.json", result)
    return result


def main() -> None:
    result = run(parse_args())
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
