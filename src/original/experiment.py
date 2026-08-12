"""Train and evaluate the three models in the team's original research scope."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Iterable, Sequence

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


def make_dataset(path: Path, architecture: str) -> Dataset:
    if architecture == "cot":
        return ExplicitCoTDataset(str(path))
    return BallSwapDataset(path)


def make_loader(
    source: Path | Dataset,
    architecture: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = make_dataset(source, architecture) if isinstance(source, Path) else source
    collator = collate_cot if architecture == "cot" else collate_fn
    return DataLoader(
        _subset(dataset, max_samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator or torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def split_train_validation(
    dataset: Dataset,
    *,
    validation_ratio: float,
    seed: int,
    max_samples: int | None,
) -> tuple[Dataset, Dataset]:
    """Create a deterministic split stratified by the number of swaps."""

    dataset = _subset(dataset, max_samples)
    size = len(dataset)
    if size < 2:
        raise ValueError("at least two training samples are required for validation")
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        row = dataset[index]  # type: ignore[index]
        if not isinstance(row, dict) or "n_swaps" not in row:
            raise ValueError("training rows must contain n_swaps for stratification")
        groups[int(row["n_swaps"])].append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for n_swaps, indices in sorted(groups.items()):
        shuffled = list(indices)
        random.Random(seed * 1_000_003 + n_swaps).shuffle(shuffled)
        if len(shuffled) == 1:
            validation_size = 0
        else:
            validation_size = max(1, round(len(shuffled) * validation_ratio))
            validation_size = min(validation_size, len(shuffled) - 1)
        validation_indices.extend(shuffled[:validation_size])
        train_indices.extend(shuffled[validation_size:])

    if not validation_indices:
        validation_indices.append(train_indices.pop())
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def move_tensors(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward_loss(
    model: OriginalModel,
    batch: dict[str, Tensor],
    deep_supervision_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(model, ExplicitCoTTransformer):
        logits = model(batch["input_ids"], batch["attention_mask"])
        targets = batch["lm_labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits, targets
    if isinstance(model, RecurrentTransformer) and deep_supervision_weight > 0.0:
        loop_logits = model.forward_all_loops(
            batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
        )
        targets = batch["labels"]
        logits = loop_logits[:, -1]
        final_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
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
            return final_loss + deep_supervision_weight * intermediate_loss, logits, targets
        return final_loss, logits, targets
    logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    targets = batch["labels"]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss, logits, targets


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train_epoch(
    model: OriginalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    deep_supervision_weight: float = 0.0,
    *,
    scaler: Any | None = None,
    amp_enabled: bool = False,
) -> float:
    model.train()
    weighted_loss = 0.0
    target_count = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            loss, _logits, targets = _forward_loss(
                model, batch, deep_supervision_weight
            )
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        count = int((targets != -100).sum().item())
        weighted_loss += float(loss.item()) * count
        target_count += count
    return weighted_loss / max(target_count, 1)


@torch.inference_mode()
def validate_epoch(
    model: OriginalModel,
    loader: DataLoader,
    device: torch.device,
    deep_supervision_weight: float = 0.0,
    *,
    amp_enabled: bool = False,
) -> dict[str, float | int]:
    """Compute teacher-forced validation metrics for checkpoint selection."""

    model.eval()
    weighted_loss = 0.0
    target_count = 0
    correct_count = 0
    exact_correct = 0
    sample_count = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        with _autocast_context(device, amp_enabled):
            loss, logits, targets = _forward_loss(model, batch, deep_supervision_weight)
        valid = targets != -100
        predictions = logits.argmax(-1)
        count = int(valid.sum().item())
        weighted_loss += float(loss.item()) * count
        target_count += count
        correct_count += int(((predictions == targets) & valid).sum().item())
        if targets.ndim == 2 and targets.shape[1] == 5:
            exact_correct += int((((predictions == targets) | ~valid).all(dim=1)).sum().item())
            sample_count += targets.shape[0]
    metrics: dict[str, float | int] = {
        "loss": weighted_loss / max(target_count, 1),
        "token_accuracy": correct_count / max(target_count, 1),
        "target_count": target_count,
    }
    if sample_count:
        metrics["exact_match"] = exact_correct / sample_count
        metrics["n_samples"] = sample_count
    return metrics


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
    model: DirectTransformer | RecurrentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    adaptive_kl: bool,
) -> dict[str, object]:
    model.eval()
    totals = _empty_totals()
    step_sum = 0
    halt_sum = 0
    final_kl_sum = 0.0
    final_kl_count = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        if adaptive_kl:
            if not isinstance(model, RecurrentTransformer):
                raise ValueError("KL halting is available only for the recurrent model")
            logits, diagnostics = model.forward_adaptive(
                batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
            )
            step_sum += int(diagnostics["steps_taken"].sum().item())
            halt_sum += int(diagnostics["halted"].sum().item())
            indices = (diagnostics["steps_taken"] - 1).unsqueeze(1)
            final_kl = diagnostics["symmetric_kl"].gather(1, indices).squeeze(1)
            finite = torch.isfinite(final_kl)
            final_kl_sum += float(final_kl[finite].sum().item())
            final_kl_count += int(finite.sum().item())
        else:
            logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
        _accumulate_metrics(logits.argmax(-1), batch["labels"], batch["n_swaps"], totals)
    metrics = _finish_metrics(totals)
    if adaptive_kl:
        samples = int(metrics["n_samples"])
        metrics.update(
            {
                "average_loops": step_sum / max(samples, 1),
                "halt_rate": halt_sum / max(samples, 1),
                "final_symmetric_kl": final_kl_sum / max(final_kl_count, 1),
                "halting_signal": "output_symmetric_kl_only",
            }
        )
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


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def save_training_checkpoint(
    path: Path,
    *,
    model: OriginalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    history: list[dict[str, object]],
    best_validation_loss: float,
    best_epoch: int,
    training_seconds: float,
    run_config: dict[str, object],
) -> dict[str, object]:
    state: dict[str, object] = {
        "format_version": 2,
        "config": model.config.to_dict(),
        "run_config": run_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "rng_state": _rng_state(),
    }
    _atomic_torch_save(path, state)
    return state


def load_training_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.2 compatibility
        return torch.load(path, map_location=device)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rewrite_metrics_log(path: Path, history: Sequence[dict[str, object]]) -> None:
    """Atomically make the durable epoch log match checkpoint history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in history:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("direct", "cot", "recurrent"), required=True)
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
    parser.add_argument("--min-loops", type=int, default=2)
    parser.add_argument("--halting-patience", type=int, default=1)
    parser.add_argument(
        "--deep-supervision-weight",
        type=float,
        default=0.0,
        help="optional recurrent ablation: CE on every non-final loop",
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
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--scheduler", choices=("none", "cosine", "linear"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint path, or 'auto' to use this run's checkpoints/last.pt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing run with the same name before starting",
    )
    parser.add_argument("--eval-splits", nargs="+", choices=EVAL_SPLITS, default=list(EVAL_SPLITS))
    parser.add_argument(
        "--eval-metrics",
        nargs="+",
        choices=("slot_accuracy", "exact_match", "exact_match_by_swaps"),
        default=["slot_accuracy", "exact_match", "exact_match_by_swaps"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _build_optimizer(args: argparse.Namespace, model: OriginalModel) -> torch.optim.Optimizer:
    common = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **common)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **common)


def _build_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> Any | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
        )

    def linear_multiplier(completed_epochs: int) -> float:
        if args.warmup_epochs > 0 and completed_epochs < args.warmup_epochs:
            return (completed_epochs + 1) / args.warmup_epochs
        decay_epochs = max(args.epochs - args.warmup_epochs, 1)
        progress = (completed_epochs - args.warmup_epochs + 1) / decay_epochs
        minimum = args.min_lr / args.lr if args.lr > 0 else 0.0
        return max(minimum, 1.0 - progress * (1.0 - minimum))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_multiplier)


def _make_grad_scaler(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # PyTorch 2.2 fallback
        return torch.cuda.amp.GradScaler(enabled=True)


def _default_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    name_parts = [args.architecture, args.position_encoding, f"seed{args.seed}"]
    if args.deep_supervision_weight > 0.0:
        name_parts.append(f"ds{args.deep_supervision_weight:g}")
    if args.noop_eval_ratio > 0.0:
        name_parts.append(f"noop{args.noop_eval_ratio:g}")
    if args.adaptive_kl_eval:
        name_parts.append("adaptive")
    return "-".join(name_parts)


def _validate_run_name(run_name: str) -> None:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run name must be a single non-empty path component")


def _prepare_run_outputs(
    args: argparse.Namespace, run_name: str
) -> tuple[Path, Path, Path, Path]:
    """Reject accidental collisions or explicitly clear one exact run."""

    _validate_run_name(run_name)
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / run_name
    result_json = output_dir / f"{run_name}.json"
    result_checkpoint = output_dir / f"{run_name}.pt"
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    existing = [path for path in (run_dir, result_json, result_checkpoint) if path.exists()]
    if args.resume is None and existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"run '{run_name}' already exists ({paths}); use --resume or --overwrite"
        )
    if args.overwrite:
        output_dir.mkdir(parents=True, exist_ok=True)
        if run_dir.parent.resolve() != output_dir or run_dir == output_dir:
            raise ValueError("refusing to overwrite a run outside output-dir")
        if run_dir.exists():
            shutil.rmtree(run_dir)
        for path in (result_json, result_checkpoint):
            if path.exists():
                path.unlink()
    return run_dir, run_dir / "checkpoints", run_dir / "metrics.jsonl", result_json


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.adaptive_kl_eval and args.architecture != "recurrent":
        raise ValueError("--adaptive-kl-eval requires --architecture recurrent")
    if args.deep_supervision_weight < 0.0:
        raise ValueError("--deep-supervision-weight must be non-negative")
    if args.deep_supervision_weight > 0.0 and args.architecture != "recurrent":
        raise ValueError("deep supervision is available only for the recurrent model")
    if args.noop_eval_ratio < 0.0:
        raise ValueError("--noop-eval-ratio must be non-negative")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between zero and one")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("--persistent-workers requires --num-workers greater than zero")
    if args.smoke:
        args.epochs = 1
        args.d_model = 32
        args.n_heads = 4
        args.d_ff = 64
        args.num_layers = 2
        args.num_loops = 2
        args.min_loops = min(args.min_loops, args.num_loops)
        args.batch_size = 8
        args.eval_batch_size = 4
        args.max_train_samples = 16
        args.max_eval_samples = 4

    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    run_name = _default_run_name(args)
    run_dir, checkpoint_dir, metrics_path, result_json_path = _prepare_run_outputs(
        args, run_name
    )
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
    )
    model = build_model(config).to(device)
    full_train_dataset = make_dataset(args.data_dir / "train.jsonl", args.architecture)
    train_dataset, validation_dataset = split_train_validation(
        full_train_dataset,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_samples=args.max_train_samples,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.architecture,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        max_samples=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
    )
    train_generator = torch.Generator()
    train_loader = make_loader(
        train_dataset,
        args.architecture,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        max_samples=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        generator=train_generator,
    )
    optimizer = _build_optimizer(args, model)
    scheduler = _build_scheduler(args, optimizer)
    scaler = _make_grad_scaler(amp_enabled)
    run_config: dict[str, object] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    start_epoch = 1
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    previous_training_seconds = 0.0
    resume_path: Path | None = None
    if args.resume == "auto":
        resume_path = checkpoint_dir / "last.pt"
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = load_training_checkpoint(resume_path, device)
        checkpoint_config = state.get("config")
        if checkpoint_config != config.to_dict():
            raise ValueError("resume checkpoint model config does not match this run")
        model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])  # type: ignore[arg-type]
        if scaler is not None and state.get("scaler_state") is not None:
            scaler.load_state_dict(state["scaler_state"])  # type: ignore[arg-type]
        history = list(state.get("history", []))  # type: ignore[arg-type]
        best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        best_epoch = int(state.get("best_epoch", 0))
        previous_training_seconds = float(state.get("training_seconds", 0.0))
        start_epoch = int(state["epoch"]) + 1
        if "rng_state" in state:
            _restore_rng_state(state["rng_state"])  # type: ignore[arg-type]
        rewrite_metrics_log(metrics_path, history)
    else:
        rewrite_metrics_log(metrics_path, [])

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_generator.manual_seed(args.seed + epoch)
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            deep_supervision_weight=args.deep_supervision_weight,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation = validate_epoch(
            model,
            validation_loader,
            device,
            deep_supervision_weight=args.deep_supervision_weight,
            amp_enabled=amp_enabled,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        improved = float(validation["loss"]) < best_validation_loss
        if improved:
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch
        elapsed = previous_training_seconds + time.perf_counter() - started
        state = save_training_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            training_seconds=elapsed,
            run_config=run_config,
        )
        if improved:
            _atomic_torch_save(checkpoint_dir / "best.pt", state)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            _atomic_torch_save(checkpoint_dir / f"epoch_{epoch}.pt", state)
        append_jsonl(metrics_path, epoch_record)

    training_seconds = previous_training_seconds + time.perf_counter() - started
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("no best checkpoint is available for final evaluation")
    best_state = load_training_checkpoint(best_path, device)
    model.load_state_dict(best_state["model_state"])  # type: ignore[arg-type]

    split_metrics: dict[str, object] = {}
    for split in args.eval_splits:
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
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=args.persistent_workers,
            )
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                adaptive_kl=args.adaptive_kl_eval,
            )
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

    result = {
        "track": "original_team_plan",
        "run_name": run_name,
        "config": config.to_dict(),
        "parameters": count_parameters(model),
        "seed": args.seed,
        "training_loss": [record["train_loss"] for record in history],
        "training_history": history,
        "training_seconds": training_seconds,
        "validation": {
            "ratio": args.validation_ratio,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "best_epoch": best_epoch,
            "best_loss": best_validation_loss,
        },
        "training": {
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "amp_requested": args.amp,
            "amp_enabled": amp_enabled,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        },
        "evaluation": {
            "splits": list(args.eval_splits),
            "metrics": list(args.eval_metrics),
            "checkpoint": "best.pt",
        },
        "ablations": {
            "deep_supervision_weight": args.deep_supervision_weight,
            "noop_eval_ratio": args.noop_eval_ratio,
        },
        "splits": split_metrics,
    }
    _atomic_torch_save(args.output_dir / f"{run_name}.pt", best_state)
    write_json(result_json_path, result)
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
