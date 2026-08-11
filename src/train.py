"""Train and evaluate vanilla or recurrent state-tracking Transformers."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

try:  # Support module-style and script-style execution.
    from .collate import BallSwapDataset, collate_fn
    from .model import ModelConfig, StateTrackingModel, count_parameters
except ImportError:  # pragma: no cover
    from collate import BallSwapDataset, collate_fn
    from model import ModelConfig, StateTrackingModel, count_parameters


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("id_test", "ood_x4", "ood_x8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_experiment_loader(
    path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    max_samples: int | None,
) -> DataLoader:
    dataset = BallSwapDataset(path)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_one_epoch(
    model: StateTrackingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    randomize_recurrent_steps: bool = False,
) -> float:
    model.train()
    loss_sum = 0.0
    label_count = 0
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        recurrent_steps = None
        if randomize_recurrent_steps and model.config.model_type == "recurrent":
            recurrent_steps = random.randint(
                model.config.min_recurrent_steps,
                model.config.recurrent_steps,
            )
        logits = model(
            batch["input_ids"],
            batch["attn_mask"],
            batch["slot_pos"],
            recurrent_steps=recurrent_steps,
        )
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        valid_count = int((labels != -100).sum().item())
        loss_sum += float(loss.item()) * valid_count
        label_count += valid_count
    return loss_sum / max(label_count, 1)


@torch.inference_mode()
def evaluate(
    model: StateTrackingModel,
    loader: DataLoader,
    device: torch.device,
    *,
    adaptive: bool = False,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    label_count = 0
    slot_correct = 0
    exact_correct = 0
    sample_count = 0
    recurrent_step_sum = 0
    halted_count = 0
    final_kl_sum = 0.0
    final_kl_count = 0
    final_update_ratio_sum = 0.0
    final_confidence_sum = 0.0
    by_swaps: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        if adaptive and model.config.model_type == "recurrent":
            logits, diagnostics = model.forward_adaptive(
                batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
            )
            recurrent_step_sum += int(diagnostics["steps_taken"].sum().item())
            halted_count += int(diagnostics["halted"].sum().item())
            final_indices = (diagnostics["steps_taken"] - 1).unsqueeze(1)
            final_kl = diagnostics["symmetric_kl"].gather(1, final_indices).squeeze(1)
            finite_kl = torch.isfinite(final_kl)
            final_kl_sum += float(final_kl[finite_kl].sum().item())
            final_kl_count += int(finite_kl.sum().item())
            final_update_ratio_sum += float(
                diagnostics["update_ratio"].gather(1, final_indices).sum().item()
            )
            final_confidence_sum += float(
                diagnostics["confidence"].gather(1, final_indices).sum().item()
            )
        else:
            logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        predictions = logits.argmax(dim=-1)
        valid = labels != -100
        correct = (predictions == labels) & valid
        exact = (correct | ~valid).all(dim=1)

        valid_count = int(valid.sum().item())
        loss_sum += float(loss.item()) * valid_count
        label_count += valid_count
        slot_correct += int(correct.sum().item())
        exact_correct += int(exact.sum().item())
        sample_count += labels.shape[0]

        for n_swaps, hit in zip(batch_cpu["n_swaps"].tolist(), exact.cpu().tolist()):
            by_swaps[int(n_swaps)][0] += int(hit)
            by_swaps[int(n_swaps)][1] += 1

    swap_metrics = {
        str(n_swaps): {
            "exact_match": hits / total,
            "correct": hits,
            "total": total,
        }
        for n_swaps, (hits, total) in sorted(by_swaps.items())
    }
    metrics: dict[str, object] = {
        "loss": loss_sum / max(label_count, 1),
        "slot_accuracy": slot_correct / max(label_count, 1),
        "exact_match": exact_correct / max(sample_count, 1),
        "n_samples": sample_count,
        "by_swaps": swap_metrics,
    }
    if adaptive and model.config.model_type == "recurrent":
        metrics["average_recurrent_steps"] = recurrent_step_sum / max(sample_count, 1)
        metrics["halt_rate"] = halted_count / max(sample_count, 1)
        metrics["final_symmetric_kl"] = final_kl_sum / max(final_kl_count, 1)
        metrics["final_update_ratio"] = final_update_ratio_sum / max(sample_count, 1)
        metrics["final_confidence"] = final_confidence_sum / max(sample_count, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: StateTrackingModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_exact_match: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_exact_match": best_exact_match,
        },
        path,
    )


def load_model_checkpoint(path: Path, device: torch.device) -> StateTrackingModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = StateTrackingModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def write_results(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def result_row(
    *,
    split: str,
    metrics: dict[str, object],
    model: StateTrackingModel,
    seed: int,
    best_epoch: int,
    training_seconds: float,
) -> dict[str, object]:
    config = model.config
    row: dict[str, object] = {
        "model": config.model_type,
        "seed": seed,
        "split": split,
        "exact_match": metrics["exact_match"],
        "slot_accuracy": metrics["slot_accuracy"],
        "loss": metrics["loss"],
        "n_samples": metrics["n_samples"],
        "by_swaps": metrics["by_swaps"],
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "dim_feedforward": config.dim_feedforward,
        "classifier_dim": config.classifier_dim,
    }
    if config.model_type == "vanilla":
        row["L"] = config.num_layers
    else:
        row["T"] = config.recurrent_steps
        row["loop_conditioning"] = config.loop_conditioning
        row["residual_scale"] = config.residual_scale
        if "average_recurrent_steps" in metrics:
            row["average_recurrent_steps"] = metrics["average_recurrent_steps"]
            row["halt_rate"] = metrics["halt_rate"]
            row["final_symmetric_kl"] = metrics["final_symmetric_kl"]
            row["final_update_ratio"] = metrics["final_update_ratio"]
            row["final_confidence"] = metrics["final_confidence"]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vanilla", "recurrent"), default="vanilla")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--classifier-dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--loop-conditioning",
        choices=("sinusoidal", "none"),
        default="none",
    )
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--min-recurrent-steps", type=int, default=1)
    parser.add_argument("--halting-threshold", type=float, default=1e-3)
    parser.add_argument("--halting-patience", type=int, default=2)
    parser.add_argument("--halting-min-confidence", type=float, default=0.5)
    parser.add_argument("--halting-update-threshold", type=float, default=0.25)
    parser.add_argument(
        "--randomize-recurrent-steps",
        action="store_true",
        help="sample a loop count per training batch from [min_recurrent_steps, T]",
    )
    parser.add_argument(
        "--adaptive-eval",
        action="store_true",
        help="halt each recurrent sample from consecutive slot-distribution KL",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--eval-only", type=Path, default=None, metavar="CHECKPOINT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")

    if args.eval_only is not None:
        model = load_model_checkpoint(args.eval_only, device)
        best_epoch = -1
        training_seconds = 0.0
    else:
        config = ModelConfig(
            model_type=args.model,
            d_model=args.d_model,
            n_heads=args.n_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_layers=args.num_layers,
            recurrent_steps=args.recurrent_steps,
            classifier_dim=args.classifier_dim,
            loop_conditioning=args.loop_conditioning,
            residual_scale=args.residual_scale,
            min_recurrent_steps=args.min_recurrent_steps,
            halting_threshold=args.halting_threshold,
            halting_patience=args.halting_patience,
            halting_min_confidence=args.halting_min_confidence,
            halting_update_threshold=args.halting_update_threshold,
        )
        model = StateTrackingModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        train_loader = make_experiment_loader(
            args.data_dir / "train.jsonl",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed,
            max_samples=args.max_train_samples,
        )
        id_loader = make_experiment_loader(
            args.data_dir / "id_test.jsonl",
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            max_samples=args.max_eval_samples,
        )

        run_name = args.run_name or (
            f"{args.model}_"
            f"{'L' + str(args.num_layers) if args.model == 'vanilla' else 'T' + str(args.recurrent_steps)}_"
            f"seed{args.seed}"
        )
        checkpoint_path = args.checkpoint_dir / f"{run_name}.pt"
        best_exact_match = -1.0
        best_id_loss = float("inf")
        best_epoch = 0
        started = time.perf_counter()
        print(f"parameters={count_parameters(model):,}")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.grad_clip,
                randomize_recurrent_steps=args.randomize_recurrent_steps,
            )
            id_metrics = evaluate(model, id_loader, device)
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"id_exact={id_metrics['exact_match']:.4f} "
                f"id_slot={id_metrics['slot_accuracy']:.4f}"
            )
            current_exact = float(id_metrics["exact_match"])
            current_id_loss = float(id_metrics["loss"])
            if current_exact > best_exact_match or (
                current_exact == best_exact_match and current_id_loss < best_id_loss
            ):
                best_exact_match = current_exact
                best_id_loss = current_id_loss
                best_epoch = epoch
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_exact_match)

        training_seconds = time.perf_counter() - started
        model = load_model_checkpoint(checkpoint_path, device)

    run_name = args.run_name or (
        f"{model.config.model_type}_"
        f"{'L' + str(model.config.num_layers) if model.config.model_type == 'vanilla' else 'T' + str(model.config.recurrent_steps)}_"
        f"seed{args.seed}"
    )
    rows = []
    for split in DEFAULT_SPLITS:
        loader = make_experiment_loader(
            args.data_dir / f"{split}.jsonl",
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            max_samples=args.max_eval_samples,
        )
        metrics = evaluate(model, loader, device, adaptive=args.adaptive_eval)
        rows.append(
            result_row(
                split=split,
                metrics=metrics,
                model=model,
                seed=args.seed,
                best_epoch=best_epoch,
                training_seconds=training_seconds,
            )
        )
        adaptive_summary = ""
        if "average_recurrent_steps" in metrics:
            adaptive_summary = (
                f" avg_steps={metrics['average_recurrent_steps']:.2f}"
                f" halt_rate={metrics['halt_rate']:.3f}"
                f" final_kl={metrics['final_symmetric_kl']:.2e}"
                f" update_ratio={metrics['final_update_ratio']:.3f}"
                f" confidence={metrics['final_confidence']:.3f}"
            )
        print(
            f"{split:8s} exact={metrics['exact_match']:.4f} "
            f"slot={metrics['slot_accuracy']:.4f} loss={metrics['loss']:.6f}"
            f"{adaptive_summary}"
        )

    result_path = args.runs_dir / f"{run_name}.json"
    write_results(result_path, rows)
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
