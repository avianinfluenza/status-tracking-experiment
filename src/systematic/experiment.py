"""Training and the E1--E7 evaluation controls for systematic state updating."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .model import StateTrackingTransformer


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@dataclass(frozen=True)
class TrainMetrics:
    loss: float
    accuracy: float
    grad_norm: float
    hidden_norm: float
    average_loops: float


def train_epoch(
    model: StateTrackingTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    random_loop_range: tuple[int, int] | None = None,
) -> TrainMetrics:
    """Final-answer supervision only; no intermediate-state loss is applied."""

    model.train()
    totals = defaultdict(float)
    n_samples = 0
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        loops = None
        if random_loop_range is not None:
            if model.config.architecture != "recurrent":
                raise ValueError("random loops are only valid for the recurrent model")
            loops = random.randint(*random_loop_range)
        logits, hidden_states = model(
            batch["input_ids"],
            batch["attention_mask"],
            num_loops=loops,
            return_hidden_states=True,
        )
        loss = F.cross_entropy(logits, batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        size = int(batch["labels"].shape[0])
        totals["loss"] += float(loss.item()) * size
        totals["correct"] += int((logits.argmax(-1) == batch["labels"]).sum().item())
        totals["grad_norm"] += float(grad_norm) * size
        totals["hidden_norm"] += float(hidden_states[-1].norm(dim=-1).mean().item()) * size
        totals["loops"] += float(loops or model.config.train_loops) * size
        n_samples += size
    return TrainMetrics(
        loss=totals["loss"] / n_samples,
        accuracy=totals["correct"] / n_samples,
        grad_norm=totals["grad_norm"] / n_samples,
        hidden_norm=totals["hidden_norm"] / n_samples,
        average_loops=totals["loops"] / n_samples,
    )


@torch.inference_mode()
def evaluate(
    model: StateTrackingTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    num_loops: int | None = None,
    collect_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    n_samples = 0
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    hidden_norm_sum = 0.0
    update_ratio_sum = 0.0
    prediction_rows: list[dict[str, object]] = []
    _synchronize(device)
    start_time = time.perf_counter()
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        logits, states = model(
            batch["input_ids"],
            batch["attention_mask"],
            num_loops=num_loops,
            return_hidden_states=True,
        )
        labels = batch["labels"]
        predictions = logits.argmax(-1)
        probabilities = logits.softmax(-1)
        confidences = probabilities.amax(-1)
        per_example_nll = F.cross_entropy(logits, labels, reduction="none")
        size = int(labels.shape[0])
        loss_sum += float(F.cross_entropy(logits, labels).item()) * size
        correct += int((predictions == labels).sum().item())
        n_samples += size
        hidden_norm_sum += float(states[-1].norm(dim=-1).mean().item()) * size
        if len(states) > 1:
            delta = (states[-1] - states[-2]).norm(dim=-1).mean()
            base = states[-2].norm(dim=-1).mean().clamp_min(1e-8)
            update_ratio_sum += float((delta / base).item()) * size
        for depth, prediction, label in zip(
            batch_cpu["target_depth"].tolist(), predictions.cpu().tolist(), labels.cpu().tolist()
        ):
            by_depth[int(depth)][0] += int(prediction == label)
            by_depth[int(depth)][1] += 1
        if collect_predictions:
            for index in range(size):
                prediction_rows.append({
                    "example_id": batch_cpu["example_id"][index],
                    "target_depth": int(batch_cpu["target_depth"][index]),
                    "num_distractors": int(batch_cpu["num_distractors"][index]),
                    "total_events": int(batch_cpu["total_events"][index]),
                    "template_split": batch_cpu["template_split"][index],
                    "label": int(labels[index].item()),
                    "prediction": int(predictions[index].item()),
                    "correct": bool((predictions[index] == labels[index]).item()),
                    "confidence": float(confidences[index].item()),
                    "nll": float(per_example_nll[index].item()),
                })
    _synchronize(device)
    elapsed = time.perf_counter() - start_time
    result: dict[str, object] = {
        "loss": loss_sum / max(n_samples, 1),
        "accuracy": correct / max(n_samples, 1),
        "n_samples": n_samples,
        "num_loops": num_loops,
        "hidden_norm": hidden_norm_sum / max(n_samples, 1),
        "final_update_ratio": update_ratio_sum / max(n_samples, 1),
        "latency_seconds": elapsed,
        "latency_per_example_seconds": elapsed / max(n_samples, 1),
        "by_depth": {
            str(depth): hits / total for depth, (hits, total) in sorted(by_depth.items())
        },
    }
    if collect_predictions:
        result["predictions"] = prediction_rows
    return result


def ood_degradation_slope(depth_accuracy: dict[int, float], train_max_depth: int) -> float:
    """Least-squares Accuracy/Depth slope for D > D_train."""

    points = sorted((depth, accuracy) for depth, accuracy in depth_accuracy.items()
                    if depth > train_max_depth)
    if len(points) < 2:
        raise ValueError("at least two OOD depths are required for a slope")
    mean_depth = sum(depth for depth, _ in points) / len(points)
    mean_accuracy = sum(accuracy for _, accuracy in points) / len(points)
    denominator = sum((depth - mean_depth) ** 2 for depth, _ in points)
    return sum(
        (depth - mean_depth) * (accuracy - mean_accuracy)
        for depth, accuracy in points
    ) / denominator


def loop_depth_sweep(
    model: StateTrackingTransformer,
    loaders_by_depth: dict[int, DataLoader],
    loop_counts: Sequence[int],
    device: torch.device,
) -> list[dict[str, object]]:
    """E3: full reasoning-depth × inference-loop matrix."""

    if model.config.architecture != "recurrent":
        raise ValueError("loop sweep requires a recurrent model")
    rows = []
    for depth, loader in sorted(loaders_by_depth.items()):
        for loops in loop_counts:
            metrics = evaluate(model, loader, device, num_loops=loops)
            rows.append({"target_depth": depth, "num_loops": loops, **metrics})
    return rows


def best_loop_by_depth(rows: Iterable[dict[str, object]]) -> dict[int, int]:
    """Return K*(D), resolving ties toward the smaller computation budget."""

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["target_depth"])].append(row)
    return {
        depth: int(max(candidates, key=lambda row: (float(row["accuracy"]), -int(row["num_loops"])))
                   ["num_loops"])
        for depth, candidates in sorted(grouped.items())
    }


def matched_length_grid(total_events: int, depths: Sequence[int]) -> list[dict[str, int]]:
    """E4 design cells: hold context event count fixed and vary only depth."""

    if any(depth > total_events or depth < 0 for depth in depths):
        raise ValueError("each depth must be within the total event budget")
    return [
        {
            "target_depth": depth,
            "num_distractors": total_events - depth,
            "total_events": total_events,
        }
        for depth in depths
    ]


class LoopStateProbe(nn.Module):
    """Linear probe used after freezing the recurrent encoder."""

    def __init__(self, d_model: int, num_locations: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_locations)

    def forward(self, cls_states: Tensor) -> Tensor:
        return self.linear(cls_states)


@torch.inference_mode()
def collect_loop_cls_states(
    model: StateTrackingTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    num_loops: int,
) -> tuple[Tensor, Tensor]:
    """E7 feature extraction. Probe fitting is intentionally separate."""

    if model.config.architecture != "recurrent":
        raise ValueError("loop probes require a recurrent model")
    model.eval()
    features, labels = [], []
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        _, states = model(
            batch["input_ids"], batch["attention_mask"],
            num_loops=num_loops, return_hidden_states=True,
        )
        features.append(torch.stack([state[:, 0] for state in states], dim=1).cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(features), torch.cat(labels)


def fit_loop_probes(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    num_locations: int,
    epochs: int = 100,
    learning_rate: float = 1e-2,
) -> list[dict[str, float | int]]:
    """Fit an independent linear probe at every loop on frozen features.

    Probe data must be disjoint from the model training data, and probe-test
    examples must be disjoint from probe-train examples. This function makes
    no update to the recurrent encoder.
    """

    if train_features.ndim != 3 or test_features.ndim != 3:
        raise ValueError("probe features must have shape [sample, loop, d_model]")
    if train_features.shape[1:] != test_features.shape[1:]:
        raise ValueError("train and test probe feature shapes do not match")
    # Feature extraction uses inference_mode; clone outside that context so
    # autograd can save probe inputs without ever touching the frozen encoder.
    train_features = train_features.clone()
    test_features = test_features.clone()
    train_labels = train_labels.clone()
    test_labels = test_labels.clone()
    results: list[dict[str, float | int]] = []
    for loop_index in range(train_features.shape[1]):
        probe = LoopStateProbe(train_features.shape[-1], num_locations)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=0.0)
        x_train = train_features[:, loop_index]
        for _ in range(epochs):
            logits = probe(x_train)
            loss = F.cross_entropy(logits, train_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            test_logits = probe(test_features[:, loop_index])
            accuracy = float((test_logits.argmax(-1) == test_labels).float().mean().item())
        results.append({
            "loop": loop_index + 1,
            "probe_accuracy": accuracy,
            "probe_train_loss": float(loss.item()),
        })
    return results


@torch.inference_mode()
def trajectory_readout_matrix(
    model: StateTrackingTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    num_loops: int,
) -> list[dict[str, float | int]]:
    """Compare each loop readout with every symbolic intermediate state.

    The model is trained only on final answers.  Reusing its final classifier
    here introduces no intermediate supervision and does not assume that loop
    index equals event index; the returned matrix tests whether such an
    alignment emerges.
    """

    if model.config.architecture != "recurrent":
        raise ValueError("trajectory readout requires a recurrent model")
    hits: Tensor | None = None
    total = 0
    model.eval()
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        _, states = model(
            batch["input_ids"], batch["attention_mask"],
            num_loops=num_loops, return_hidden_states=True,
        )
        trajectories = batch_cpu["trajectory"]
        trajectory_lengths = {len(trajectory) for trajectory in trajectories}
        if len(trajectory_lengths) != 1:
            raise ValueError("trajectory matrix requires a fixed target depth")
        gold = torch.tensor(trajectories, device=device, dtype=torch.long)
        predictions = torch.stack(
            [model.classifier(state[:, 0]).argmax(-1) for state in states], dim=1
        )
        batch_hits = (predictions.unsqueeze(-1) == gold.unsqueeze(1)).sum(0).cpu()
        hits = batch_hits if hits is None else hits + batch_hits
        total += predictions.shape[0]
    if hits is None:
        return []
    return [
        {
            "loop": loop + 1,
            "trajectory_step": trajectory_step,
            "accuracy": float(hits[loop, trajectory_step].item() / total),
        }
        for loop in range(hits.shape[0])
        for trajectory_step in range(hits.shape[1])
    ]
