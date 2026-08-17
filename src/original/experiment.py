"""Train and evaluate the three models in the team's original research scope."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
import types
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm.auto import tqdm

from ..data.collate import BallSwapDataset, collate_fn, encode_event
from ..data.vocab import N_ENTITIES
from .data import (
    DeterministicOnlineBatchStream,
    ExplicitCoTDataset,
    RowsDataset,
    collate_cot,
    group_rows_by_swap_count,
    inject_noop_swaps,
)
from .model import (
    DirectTransformer,
    EventWiseRecurrentTransformer,
    ExplicitCoTTransformer,
    FanRecurrentTransformer,
    OriginalModel,
    OriginalModelConfig,
    RecurrentR0Transformer,
    RecurrentTransformer,
    SelfAttention,
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


def maybe_compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Compile the model forward path on CUDA while preserving its concrete type."""

    if device.type != "cuda":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("CUDA execution requires a PyTorch build with torch.compile")
    model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
    return model


def _subset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    return Subset(dataset, range(min(maximum, len(dataset))))


def length_proportional_loop_counts(n_swaps: Tensor, swaps_per_loop: float) -> Tensor:
    """Return ``ceil(n_swaps / swaps_per_loop)`` for every sample."""

    if swaps_per_loop <= 0:
        raise ValueError("swaps_per_loop must be positive")
    if n_swaps.ndim != 1 or bool((n_swaps < 1).any()):
        raise ValueError("n_swaps must be a positive [batch] tensor")
    return torch.ceil(n_swaps.to(torch.float32) / swaps_per_loop).to(dtype=torch.long)


def fan_learning_rate_multiplier(
    optimizer_step: int,
    *,
    train_steps: int,
    curriculum_steps: int,
) -> float:
    """Hold LR through curriculum, then cosine-decay it to zero."""

    if train_steps < 1 or curriculum_steps < 0:
        raise ValueError("train_steps must be positive and curriculum_steps non-negative")
    if optimizer_step <= curriculum_steps or curriculum_steps >= train_steps:
        return 1.0
    progress = min(
        1.0,
        (optimizer_step - curriculum_steps) / max(train_steps - curriculum_steps, 1),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class SwapCountBatchSampler(Sampler[list[int]]):
    """Shuffle batches within equal length-proportional recurrence budgets."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        swaps_per_loop: float,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if swaps_per_loop <= 0:
            raise ValueError("swaps_per_loop must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            row = dataset[index]
            if not isinstance(row, dict) or "n_swaps" not in row:
                raise ValueError("length-proportional batching requires rows with n_swaps")
            n_swaps = int(row["n_swaps"])
            loops = math.ceil(n_swaps / swaps_per_loop)
            if loops < 1:
                raise ValueError("n_swaps must be positive")
            self.buckets[loops].append(index)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for loops in sorted(self.buckets):
            indices = list(self.buckets[loops])
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.buckets.values())


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
    slot_first: bool = False,
    swaps_per_loop: float | None = None,
    input_format: str = "template",
) -> DataLoader:
    if architecture == "cot":
        if slot_first:
            raise ValueError("--slot-first is supported for direct/recurrent classifiers, not cot")
        dataset = ExplicitCoTDataset(str(source)) if isinstance(source, Path) else source
        collator = collate_cot
    else:
        dataset = BallSwapDataset(source) if isinstance(source, Path) else source
        collator = partial(collate_fn, slot_first=slot_first, input_format=input_format)
    selected_dataset = _subset(dataset, max_samples)
    if swaps_per_loop is not None:
        if architecture == "cot":
            raise ValueError("length-proportional loops are supported only for recurrent classifiers")
        return DataLoader(
            selected_dataset,
            batch_sampler=SwapCountBatchSampler(
                selected_dataset,
                batch_size=batch_size,
                swaps_per_loop=swaps_per_loop,
                shuffle=shuffle,
                seed=seed,
            ),
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
        )
    return DataLoader(
        selected_dataset,
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
    if isinstance(model, EventWiseRecurrentTransformer):
        targets = batch["labels"]
        logits = model(
            batch["initial_colors"],
            batch["register_mask"],
            batch["event_input_ids"],
            batch["event_mask"],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits, targets
    if _supports_loop_override(model) and deep_supervision_weight > 0.0:
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


class InferenceComputeMeter:
    """Runtime FLOP and wall-time meter for this repository's forward passes.

    Counts multiply-add work in ``nn.Linear`` modules and the two attention
    matrix multiplications (QK^T and AV) from their runtime tensor shapes.
    It deliberately excludes embedding lookups, normalisation, masking,
    softmax, activations, and elementwise residual operations.
    """

    method = "runtime linear projections + attention QK^T/AV; excludes embedding, norm, softmax, activation, masking, and elementwise ops"

    def __init__(self, model: OriginalModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.linear_flops = 0
        self.attention_flops = 0
        self.inference_seconds = 0.0
        self.forward_calls = 0
        self._hooks: list[Any] = []
        self._patched_attention: list[SelfAttention] = []

    def __enter__(self) -> "InferenceComputeMeter":
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._linear_hook))
            if isinstance(module, SelfAttention):
                self._hooks.append(module.register_forward_hook(self._attention_hook))
                self._patch_incremental_attention(module)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for attention in self._patched_attention:
            delattr(attention, "incremental")
        self._patched_attention.clear()

    def _linear_hook(self, module: nn.Module, inputs: tuple[object, ...], _output: object) -> None:
        assert isinstance(module, nn.Linear)
        input_tensor = inputs[0]
        if not isinstance(input_tensor, Tensor):
            raise TypeError("linear meter expected a Tensor input")
        vectors = input_tensor.numel() // module.in_features
        self.linear_flops += 2 * vectors * module.in_features * module.out_features

    def _attention_hook(self, _module: nn.Module, inputs: tuple[object, ...], _output: object) -> None:
        input_tensor = inputs[0]
        if not isinstance(input_tensor, Tensor):
            raise TypeError("attention meter expected a Tensor input")
        batch, length, width = input_tensor.shape
        # QK^T and AV each cost 2 * B * L * L * d_model FLOPs.
        self.attention_flops += 4 * batch * length * length * width

    def _patch_incremental_attention(self, attention: SelfAttention) -> None:
        original = attention.incremental

        def counted_incremental(
            _module: SelfAttention,
            x: Tensor,
            position_ids: Tensor,
            cache: tuple[Tensor, Tensor] | None,
        ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
            batch, new_length, width = x.shape
            key_length = new_length if cache is None else new_length + cache[0].shape[-2]
            self.attention_flops += 4 * batch * new_length * key_length * width
            return original(x, position_ids, cache)

        attention.incremental = types.MethodType(counted_incremental, attention)
        self._patched_attention.append(attention)

    def measure(self, forward: Any) -> Any:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = forward()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.inference_seconds += time.perf_counter() - started
        self.forward_calls += 1
        return result

    def summary(self, n_samples: int) -> dict[str, object]:
        flops = self.linear_flops + self.attention_flops
        return {
            "flops": flops,
            "tflops": flops / 1e12,
            "flops_per_sample": flops / max(n_samples, 1),
            "inference_seconds": self.inference_seconds,
            "milliseconds_per_sample": 1e3 * self.inference_seconds / max(n_samples, 1),
            "samples_per_second": n_samples / max(self.inference_seconds, 1e-12),
            "forward_calls": self.forward_calls,
            "method": self.method,
        }


def _supports_loop_override(model: OriginalModel) -> bool:
    return isinstance(
        model,
        (RecurrentTransformer, RecurrentR0Transformer, FanRecurrentTransformer),
    )


def proportional_target_indices(num_loops: int, n_swaps: Tensor) -> Tensor:
    """Map each 1-indexed recurrent loop to a zero-indexed swap state.

    The result has shape ``[batch, num_loops]`` and implements
    ``ceil(k * N / K) - 1`` exactly with integer arithmetic for every sample.
    """

    if num_loops < 1:
        raise ValueError("num_loops must be positive")
    if n_swaps.ndim != 1 or bool((n_swaps < 1).any()):
        raise ValueError("n_swaps must be a positive [batch] tensor")
    loop_numbers = torch.arange(1, num_loops + 1, device=n_swaps.device).unsqueeze(0)
    return torch.div(loop_numbers * n_swaps.unsqueeze(1) + num_loops - 1, num_loops,
                     rounding_mode="floor") - 1


def swap_chunk_target_indices(
    num_loops: int,
    n_swaps: Tensor,
    swaps_per_loop: float,
) -> Tensor:
    """Map loop ``k`` to the state after ``min(ceil(k * r), N)`` swaps."""

    if num_loops < 1 or swaps_per_loop <= 0:
        raise ValueError("num_loops and swaps_per_loop must be positive")
    if n_swaps.ndim != 1 or bool((n_swaps < 1).any()):
        raise ValueError("n_swaps must be a positive [batch] tensor")
    completed = torch.minimum(
        torch.ceil(
            torch.arange(1, num_loops + 1, device=n_swaps.device, dtype=torch.float32).unsqueeze(0)
            * swaps_per_loop
        ).to(dtype=torch.long),
        n_swaps.unsqueeze(1),
    )
    return completed - 1


def trajectory_targets_for_loops(
    trajectory_labels: Tensor,
    n_swaps: Tensor,
    num_loops: int,
    swaps_per_loop: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Gather proportional or fixed-swap-chunk targets for every loop."""

    if trajectory_labels.ndim != 3:
        raise ValueError("trajectory_labels must have shape [batch, swaps, slots]")
    if trajectory_labels.shape[0] != n_swaps.shape[0]:
        raise ValueError("trajectory_labels and n_swaps batch sizes must match")
    indices = (
        proportional_target_indices(num_loops, n_swaps)
        if swaps_per_loop is None
        else swap_chunk_target_indices(num_loops, n_swaps, swaps_per_loop)
    )
    if bool((indices >= trajectory_labels.shape[1]).any()):
        raise ValueError("trajectory_labels does not cover all declared swaps")
    targets = trajectory_labels.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, trajectory_labels.shape[-1]),
    )
    return targets, indices


def _require_trajectory_labels(batch: dict[str, Tensor]) -> Tensor:
    labels = batch.get("trajectory_labels")
    if labels is None:
        raise ValueError(
            "trajectory probing requires intermediate_states in every dataset row. "
            "Regenerate the dataset with `python -m src.data.data --out data`."
        )
    return labels


def noop_event_input_ids(person_ids: Tensor) -> Tensor:
    """Encode active self-swaps for the requested person in each sample."""

    if person_ids.ndim != 1 or person_ids.dtype != torch.long:
        raise ValueError("person_ids must be a one-dimensional long tensor")
    if bool(((person_ids < 0) | (person_ids >= N_ENTITIES)).any()):
        raise ValueError("person_ids contains an invalid entity")
    table = person_ids.new_tensor([encode_event(index, index) for index in range(N_ENTITIES)])
    return table[person_ids]


def relative_state_update_norm(updated: Tensor, reference: Tensor, register_mask: Tensor) -> Tensor:
    """Return one masked relative L2 update magnitude per sample."""

    if updated.shape != reference.shape or updated.ndim != 3:
        raise ValueError("updated and reference must have shape [batch, registers, hidden]")
    if register_mask.shape != updated.shape[:2]:
        raise ValueError("register_mask must match the state register dimensions")
    mask = register_mask.unsqueeze(-1).to(dtype=updated.dtype)
    difference = ((updated - reference).square() * mask).sum(dim=(1, 2)).sqrt()
    magnitude = (reference.square() * mask).sum(dim=(1, 2)).sqrt().clamp_min(1e-8)
    return difference / magnitude




def timestamp_label(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%Y%m%d-%H%M%S")


def _safe_path_name(name: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._=-]+", "-", name).strip("-")
    return collapsed or "run"


def create_unique_run_dir(
    output_dir: Path,
    run_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    base = f"{_safe_path_name(run_name)}__{timestamp or timestamp_label()}"
    for index in range(1000):
        suffix = "" if index == 0 else f"-{index + 1:02d}"
        candidate = output_dir / f"{base}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not create a unique run directory for {run_name}")


def _progress_bar(
    iterable: Iterable,
    *,
    total: int,
    desc: str,
    leave: bool,
    position: int,
):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        position=position,
        dynamic_ncols=True,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return _json_ready(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    name_parts = [args.architecture, args.position_encoding, f"seed{args.seed}"]
    if args.architecture == "fan-recurrent":
        if args.fan_input_format != "template":
            name_parts.append(f"input{args.fan_input_format}")
        if args.fan_positional_control:
            name_parts.append("poscontrol")
    if args.architecture == "direct":
        if args.direct_input_format != "template":
            name_parts.append(f"input{args.direct_input_format}")
        if args.direct_causal:
            name_parts.append("causal")
    if args.extended_length:
        name_parts.append("extended-length")
    if args.slot_first:
        name_parts.append("slotfirst")
    if args.deep_supervision_weight > 0.0:
        name_parts.append(f"ds{args.deep_supervision_weight:g}")
    if args.swaps_per_loop is not None:
        name_parts.append(f"spl{args.swaps_per_loop:g}")
    if args.trajectory_probe_eval:
        name_parts.append("trajprobe")
    if args.event_trajectory_probe:
        name_parts.append("eventprobe")
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
    if args.online_training:
        name_parts.extend(
            (
                "online",
                f"curr{args.curriculum_min_swaps}-{args.curriculum_max_swaps}",
                f"steps{args.train_steps}",
            )
        )
    return "-".join(name_parts)


def train_epoch(
    model: OriginalModel,
    loader: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    deep_supervision_weight: float = 0.0,
    *,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    random_loop_range: tuple[int, int] | None = None,
    swaps_per_loop: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int | None = None,
    show_progress: bool = False,
    metrics: dict[str, float] | None = None,
) -> float:
    if swaps_per_loop is not None and random_loop_range is not None:
        raise ValueError("length-proportional loops and random-loop training are mutually exclusive")
    if swaps_per_loop is not None and not _supports_loop_override(model):
        raise ValueError("length-proportional loops require a recurrent model")
    model.train()
    weighted_loss = 0.0
    target_count = 0
    slot_correct = 0
    exact_correct = 0
    sample_count = 0
    progress = None
    batches: Iterable = loader
    if show_progress:
        desc = f"epoch {epoch}" if epoch is not None else "optimizer steps"
        progress = _progress_bar(
            loader,
            total=len(loader),
            desc=desc,
            leave=False,
            position=1,
        )
        batches = progress
    try:
        for batch_cpu in batches:
            batch = move_tensors(batch_cpu, device)
            sampled_loops = None
            if random_loop_range is not None:
                if not isinstance(model, RecurrentR0Transformer):
                    raise ValueError("random-loop training is available only for recurrent-r0")
                sampled_loops = random.randint(*random_loop_range)
            elif swaps_per_loop is not None:
                loop_counts = length_proportional_loop_counts(batch["n_swaps"], swaps_per_loop)
                if not bool((loop_counts == loop_counts[0]).all()):
                    raise ValueError("length-proportional batches must have a shared loop count")
                sampled_loops = int(loop_counts[0].item())
            if isinstance(model, ExplicitCoTTransformer):
                logits = model(batch["input_ids"], batch["attention_mask"])
                targets = batch["lm_labels"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
            elif isinstance(model, EventWiseRecurrentTransformer):
                targets = batch["labels"]
                logits = model(
                    batch["initial_colors"],
                    batch["register_mask"],
                    batch["event_input_ids"],
                    batch["event_mask"],
                )
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
                logits = final_logits
                final_loss = F.cross_entropy(
                    final_logits.reshape(-1, final_logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                if loop_logits.shape[1] > 1:
                    loss = final_loss
                    if deep_supervision_weight > 0.0:
                        intermediate = loop_logits[:, :-1]
                        repeated_targets = targets[:, None].expand(-1, intermediate.shape[1], -1)
                        intermediate_loss = F.cross_entropy(
                            intermediate.reshape(-1, intermediate.shape[-1]),
                            repeated_targets.reshape(-1),
                            ignore_index=-100,
                        )
                        loss = loss + deep_supervision_weight * intermediate_loss
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
            if scheduler is not None:
                scheduler.step()
            count = int((targets != -100).sum().item())
            weighted_loss += float(loss.item()) * count
            target_count += count
            valid = targets != -100
            predictions = logits.detach().argmax(dim=-1)
            correct = (predictions == targets) & valid
            slot_correct += int(correct.sum().item())
            exact_correct += int((correct | ~valid).all(dim=1).sum().item())
            sample_count += targets.shape[0]
            if progress is not None:
                running_loss = weighted_loss / max(target_count, 1)
                progress.set_postfix(
                    loss=f"{running_loss:.4f}",
                    slot_accuracy=f"{slot_correct / max(target_count, 1):.4f}",
                    exact_match=f"{exact_correct / max(sample_count, 1):.4f}",
                )
    finally:
        if progress is not None:
            progress.close()
    if metrics is not None:
        metrics.update(
            loss=weighted_loss / max(target_count, 1),
            slot_accuracy=slot_correct / max(target_count, 1),
            exact_match=exact_correct / max(sample_count, 1),
        )
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
    model: DirectTransformer | RecurrentTransformer | RecurrentR0Transformer | FanRecurrentTransformer | EventWiseRecurrentTransformer,
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
    with InferenceComputeMeter(model, device) as meter:
        for batch_cpu in loader:
            batch = move_tensors(batch_cpu, device)
            if adaptive_kl:
                if not _supports_loop_override(model):
                    raise ValueError("adaptive halting is available only for recurrent models")
                logits, diagnostics = meter.measure(
                    lambda: model.forward_adaptive(
                        batch["input_ids"],
                        batch["attn_mask"],
                        batch["slot_pos"],
                        max_loops=num_loops,
                    )
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
                if isinstance(model, EventWiseRecurrentTransformer):
                    logits = meter.measure(
                        lambda: model(
                            batch["initial_colors"],
                            batch["register_mask"],
                            batch["event_input_ids"],
                            batch["event_mask"],
                        )
                    )
                elif num_loops is not None:
                    logits = meter.measure(
                        lambda: model(
                            batch["input_ids"],
                            batch["attn_mask"],
                            batch["slot_pos"],
                            num_loops=num_loops,
                        )
                    )
                else:
                    logits = meter.measure(
                        lambda: model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
                    )
            _accumulate_metrics(logits.argmax(-1), batch["labels"], batch["n_swaps"], totals)
    metrics = _finish_metrics(totals)
    metrics["inference_compute"] = meter.summary(int(metrics["n_samples"]))
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


@torch.inference_mode()
def evaluate_length_matched_classifier(
    model: RecurrentTransformer | RecurrentR0Transformer | FanRecurrentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    swaps_per_loop: float,
) -> dict[str, object]:
    """Evaluate with an externally supplied, per-sample recurrence budget.

    This is a diagnostic oracle: it selects ``ceil(n_swaps / swaps_per_loop)``
    from dataset metadata and never exposes that value to the model.
    """

    if swaps_per_loop <= 0:
        raise ValueError("swaps_per_loop must be positive")
    model.eval()
    totals = _empty_totals()
    loop_histogram: dict[int, int] = defaultdict(int)
    loop_sum = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        loop_counts = length_proportional_loop_counts(batch["n_swaps"], swaps_per_loop)
        for loop_count in loop_counts.unique(sorted=True).tolist():
            mask = loop_counts == loop_count
            sample_count = int(mask.sum().item())
            logits = model(
                batch["input_ids"][mask],
                batch["attn_mask"][mask],
                batch["slot_pos"][mask],
                num_loops=int(loop_count),
            )
            _accumulate_metrics(
                logits.argmax(-1),
                batch["labels"][mask],
                batch["n_swaps"][mask],
                totals,
            )
            loop_histogram[int(loop_count)] += sample_count
            loop_sum += int(loop_count) * sample_count
    metrics = _finish_metrics(totals)
    metrics.update(
        {
            "evaluation_mode": (
                "length_matched"
                if isinstance(model, FanRecurrentTransformer)
                else "length_matched_oracle"
            ),
            "swaps_per_loop": swaps_per_loop,
            "average_loops": loop_sum / max(int(metrics["n_samples"]), 1),
            "loop_count_histogram": {
                str(loop_count): count for loop_count, count in sorted(loop_histogram.items())
            },
        }
    )
    return metrics


@torch.inference_mode()
def evaluate_trajectory_probe(
    model: RecurrentTransformer | RecurrentR0Transformer | FanRecurrentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    num_loops: int,
    swaps_per_loop: float | None = None,
) -> dict[str, object]:
    """Measure shared-head readout accuracy against proportional state targets."""

    model.eval()
    slot_correct = [0] * num_loops
    slot_total = [0] * num_loops
    exact_correct = [0] * num_loops
    samples = [0] * num_loops
    target_sum = [0] * num_loops
    target_min: list[int | None] = [None] * num_loops
    target_max: list[int | None] = [None] * num_loops
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        trajectory_labels = _require_trajectory_labels(batch)
        loop_logits = model.forward_all_loops(
            batch["input_ids"], batch["attn_mask"], batch["slot_pos"], num_loops=num_loops
        )
        assert isinstance(loop_logits, Tensor)
        targets, target_indices = trajectory_targets_for_loops(
            trajectory_labels, batch["n_swaps"], num_loops, swaps_per_loop=swaps_per_loop
        )
        predictions = loop_logits.argmax(dim=-1)
        for loop_index in range(num_loops):
            target = targets[:, loop_index]
            valid = target != -100
            correct = (predictions[:, loop_index] == target) & valid
            slot_correct[loop_index] += int(correct.sum().item())
            slot_total[loop_index] += int(valid.sum().item())
            exact_correct[loop_index] += int((correct | ~valid).all(dim=1).sum().item())
            samples[loop_index] += target.shape[0]
            target_values = (target_indices[:, loop_index] + 1).tolist()
            target_sum[loop_index] += sum(target_values)
            current_min, current_max = min(target_values), max(target_values)
            target_min[loop_index] = current_min if target_min[loop_index] is None else min(target_min[loop_index], current_min)
            target_max[loop_index] = current_max if target_max[loop_index] is None else max(target_max[loop_index], current_max)

    loops = [
        {
            "loop": loop_index + 1,
            "slot_accuracy": slot_correct[loop_index] / max(slot_total[loop_index], 1),
            "exact_match": exact_correct[loop_index] / max(samples[loop_index], 1),
            "target_swap_index": {
                "min": target_min[loop_index],
                "max": target_max[loop_index],
                "mean": target_sum[loop_index] / max(samples[loop_index], 1),
            },
        }
        for loop_index in range(num_loops)
    ]
    slot_values = [float(item["slot_accuracy"]) for item in loops]
    exact_values = [float(item["exact_match"]) for item in loops]
    return {
        "num_loops": num_loops,
        "target_schedule": "swap_chunks" if swaps_per_loop is not None else "proportional",
        "swaps_per_loop": swaps_per_loop,
        "loops": loops,
        "monotonicity": {
            "slot_accuracy_decreases": sum(next_value < value for value, next_value in zip(slot_values, slot_values[1:])),
            "exact_match_decreases": sum(next_value < value for value, next_value in zip(exact_values, exact_values[1:])),
            "slot_accuracy_non_decreasing": all(next_value >= value for value, next_value in zip(slot_values, slot_values[1:])),
            "exact_match_non_decreasing": all(next_value >= value for value, next_value in zip(exact_values, exact_values[1:])),
        },
    }


@torch.inference_mode()
def evaluate_event_trajectory_probe(
    model: EventWiseRecurrentTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """Measure state readout after each real event against its aligned target."""

    model.eval()
    slot_correct: list[int] = []
    slot_total: list[int] = []
    exact_correct: list[int] = []
    sample_total: list[int] = []
    initial_slot_correct = 0
    initial_slot_total = 0
    initial_exact_correct = 0
    initial_sample_total = 0
    real_update_sum = 0.0
    noop_update_sum = 0.0
    update_sample_total = 0
    for batch_cpu in loader:
        batch = move_tensors(batch_cpu, device)
        trajectory_labels = _require_trajectory_labels(batch)
        initial_state = model.initialize_state(
            batch["initial_colors"],
            batch["register_mask"],
        )
        initial_predictions = model._classify(
            initial_state,
            batch["register_mask"],
        ).argmax(dim=-1)
        initial_valid = batch["register_mask"].bool()
        initial_correct = (initial_predictions == batch["initial_colors"]) & initial_valid
        initial_slot_correct += int(initial_correct.sum().item())
        initial_slot_total += int(initial_valid.sum().item())
        initial_exact_correct += int((initial_correct | ~initial_valid).all(dim=1).sum().item())
        initial_sample_total += batch["initial_colors"].shape[0]
        event_logits, hidden_states = model.forward_all_events(
            batch["initial_colors"],
            batch["register_mask"],
            batch["event_input_ids"],
            batch["event_mask"],
            return_hidden_states=True,
        )
        predictions = event_logits.argmax(dim=-1)
        num_events = event_logits.shape[1]
        missing = num_events - len(slot_correct)
        if missing > 0:
            slot_correct.extend([0] * missing)
            slot_total.extend([0] * missing)
            exact_correct.extend([0] * missing)
            sample_total.extend([0] * missing)
        for event_index in range(num_events):
            active = batch["event_mask"][:, event_index].bool()
            if not bool(active.any()):
                continue
            previous_state = initial_state if event_index == 0 else hidden_states[event_index - 1]
            current_state = hidden_states[event_index]
            real_norm = relative_state_update_norm(
                current_state,
                previous_state,
                batch["register_mask"],
            )
            person_ids = torch.full(
                (previous_state.shape[0],),
                event_index % N_ENTITIES,
                dtype=torch.long,
                device=previous_state.device,
            )
            noop_state = model.recurrent_step(
                previous_state,
                noop_event_input_ids(person_ids),
                batch["register_mask"],
                active.long(),
            )
            noop_norm = relative_state_update_norm(
                noop_state,
                previous_state,
                batch["register_mask"],
            )
            real_update_sum += float(real_norm[active].sum().item())
            noop_update_sum += float(noop_norm[active].sum().item())
            update_sample_total += int(active.sum().item())
            target = trajectory_labels[:, event_index]
            valid = (target != -100) & active.unsqueeze(1)
            correct = (predictions[:, event_index] == target) & valid
            slot_correct[event_index] += int(correct.sum().item())
            slot_total[event_index] += int(valid.sum().item())
            exact = (correct | ~valid).all(dim=1) & active
            exact_correct[event_index] += int(exact.sum().item())
            sample_total[event_index] += int(active.sum().item())

    events = [
        {
            "event": event_index + 1,
            "n_samples": sample_total[event_index],
            "slot_accuracy": slot_correct[event_index] / max(slot_total[event_index], 1),
            "exact_match": exact_correct[event_index] / max(sample_total[event_index], 1),
        }
        for event_index in range(len(slot_correct))
        if sample_total[event_index] > 0
    ]
    return {
        "evaluation_mode": "event_aligned_trajectory",
        "initial_state": {
            "n_samples": initial_sample_total,
            "slot_accuracy": initial_slot_correct / max(initial_slot_total, 1),
            "exact_match": initial_exact_correct / max(initial_sample_total, 1),
        },
        "overall_slot_accuracy": sum(slot_correct) / max(sum(slot_total), 1),
        "overall_exact_match": sum(exact_correct) / max(sum(sample_total), 1),
        "update_norms": {
            "n_event_states": update_sample_total,
            "real_relative_l2": real_update_sum / max(update_sample_total, 1),
            "noop_relative_l2": noop_update_sum / max(update_sample_total, 1),
            "real_to_noop_ratio": real_update_sum / max(noop_update_sum, 1e-12),
        },
        "events": events,
    }


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
    device = model.embedding.weight.device
    with InferenceComputeMeter(model, device) as meter:
        for n_swaps, rows in sorted(groups.items()):
            for start in range(0, len(rows), generation_batch_size):
                chunk = rows[start : start + generation_batch_size]
                predictions = meter.measure(lambda: model.generate_states(chunk)).cpu()
                labels = torch.tensor([row["labels"] for row in chunk], dtype=torch.long)
                lengths = torch.full((len(chunk),), n_swaps, dtype=torch.long)
                _accumulate_metrics(predictions, labels, lengths, totals)
    metrics = _finish_metrics(totals)
    metrics["evaluation_mode"] = "autoregressive_trace_generation"
    metrics["inference_compute"] = meter.summary(int(metrics["n_samples"]))
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
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
    def parse_bool(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=(
            "direct",
            "cot",
            "recurrent",
            "recurrent-r0",
            "fan-recurrent",
            "event-recurrent",
        ),
        required=True,
    )
    parser.add_argument(
        "--position-encoding",
        choices=("none", "sinusoidal", "rope"),
        default="sinusoidal",
    )
    parser.add_argument(
        "--fan-input-format",
        choices=("template", "atomic"),
        default="template",
        help="Fan control input representation; atomic compresses each init assignment and swap to one token",
    )
    parser.add_argument(
        "--fan-positional-control",
        action="store_true",
        help="explicit template+sinusoidal control; never part of the NoPE Fan main condition",
    )
    parser.add_argument(
        "--direct-input-format",
        choices=("template", "atomic"),
        default="template",
        help="input representation for the fixed-depth direct Transformer baseline",
    )
    parser.add_argument(
        "--direct-causal",
        action="store_true",
        help="use causal self-attention in the fixed-depth direct Transformer baseline",
    )
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
    parser.add_argument(
        "--swaps-per-loop",
        type=float,
        default=None,
        help="use ceil(n_swaps / r) recurrent loops; targets remain final-state only",
    )
    parser.add_argument(
        "--length-matched-eval",
        action="store_true",
        help="diagnostic oracle evaluation with ceil(n_swaps / r) loops per sample",
    )
    parser.add_argument("--eval-loop-counts", type=int, nargs="+", default=None)
    parser.add_argument(
        "--deep-supervision-weight",
        type=float,
        default=0.0,
        help="optional recurrent ablation: final-state CE on every non-final loop",
    )
    parser.add_argument(
        "--trajectory-probe-eval",
        action="store_true",
        help="record intermediate-state accuracy for every recurrent loop at evaluation",
    )
    parser.add_argument(
        "--event-trajectory-probe",
        action="store_true",
        help="record state accuracy after every real event for event-recurrent",
    )
    parser.add_argument(
        "--adaptive-max-loops",
        type=int,
        default=None,
        help="maximum recurrent steps for adaptive evaluation (never derived from n_swaps)",
    )
    parser.add_argument(
        "--noop-eval-ratio",
        type=float,
        default=0.0,
        help="also evaluate ID rows after inserting this fraction of self-swaps",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--online-training",
        action="store_true",
        help="generate deterministic fresh training batches keyed by seed and optimizer step",
    )
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-min-swaps", type=int, default=2)
    parser.add_argument("--curriculum-max-swaps", type=int, default=10)
    parser.add_argument(
        "--curriculum-steps-per-length",
        type=int,
        default=1_000,
        help="optimizer steps before increasing the online curriculum's maximum swap count",
    )
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
    parser.add_argument(
        "--extended-length",
        "--extended_length",
        dest="extended_length",
        action="store_true",
        help="use data/extended_length (train/ID 2~32, OOD x4 40~80, OOD x8 80~160)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "original")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--slot-first",
        "--slot_first",
        dest="slot_first",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="place fixed output SLOT registers before the body (accepts optional True/False)",
    )
    parser.add_argument("--no-progress", action="store_true")
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
    return build_run_name(args)


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
    run_started_at = datetime.now()
    run_started = time.perf_counter()
    recurrent_architectures = {"recurrent", "recurrent-r0", "fan-recurrent"}
    adaptive_architectures = {"recurrent", "recurrent-r0"}
    if args.adaptive_kl_eval and args.architecture not in adaptive_architectures:
        raise ValueError("--adaptive-kl-eval requires a recurrent architecture")
    if args.deep_supervision_weight < 0.0:
        raise ValueError("--deep-supervision-weight must be non-negative")
    if args.deep_supervision_weight > 0.0 and args.architecture not in recurrent_architectures:
        raise ValueError("deep supervision is available only for the recurrent model")
    if args.event_trajectory_probe and args.architecture != "event-recurrent":
        raise ValueError("--event-trajectory-probe requires --architecture event-recurrent")
    if (args.trajectory_probe_eval or args.length_matched_eval) and args.architecture not in recurrent_architectures:
        raise ValueError("trajectory probing and length-matched evaluation require a recurrent model")
    if args.swaps_per_loop is not None:
        if args.swaps_per_loop <= 0:
            raise ValueError("--swaps-per-loop must be positive")
        if args.random_loops:
            raise ValueError("--swaps-per-loop and --random-loops are mutually exclusive")
        if args.architecture not in recurrent_architectures:
            raise ValueError("--swaps-per-loop requires a recurrent architecture")
    if args.length_matched_eval and args.swaps_per_loop is None:
        raise ValueError("--length-matched-eval requires --swaps-per-loop")
    if args.adaptive_max_loops is not None:
        if args.adaptive_max_loops < 1:
            raise ValueError("--adaptive-max-loops must be positive")
        if not args.adaptive_kl_eval:
            raise ValueError("--adaptive-max-loops requires --adaptive-kl-eval")
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
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between zero and one")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("--persistent-workers requires --num-workers greater than zero")
    if args.online_training and args.architecture not in {"fan-recurrent", "direct"}:
        raise ValueError("--online-training requires fan-recurrent or direct")
    if args.architecture != "fan-recurrent" and (
        args.fan_input_format != "template" or args.fan_positional_control
    ):
        raise ValueError("Fan input and positional controls require --architecture fan-recurrent")
    if args.architecture != "direct" and (
        args.direct_input_format != "template" or args.direct_causal
    ):
        raise ValueError("direct input and causal controls require --architecture direct")
    if args.architecture == "direct" and args.online_training:
        if (
            args.direct_input_format != "atomic"
            or not args.direct_causal
            or args.position_encoding != "none"
        ):
            raise ValueError(
                "online direct baseline requires atomic input, causal attention, and NoPE"
            )
    if args.architecture == "fan-recurrent":
        if args.fan_positional_control:
            if args.position_encoding != "sinusoidal":
                raise ValueError("--fan-positional-control requires --position-encoding sinusoidal")
        elif args.position_encoding != "none":
            raise ValueError(
                "fan-recurrent requires --position-encoding none unless "
                "--fan-positional-control is set"
            )
        if not args.online_training:
            raise ValueError("fan-recurrent requires --online-training")
        if args.swaps_per_loop is None:
            args.swaps_per_loop = 1.0
        args.length_matched_eval = True
    if args.train_steps < 1:
        raise ValueError("--train-steps must be positive")
    if not 1 <= args.curriculum_min_swaps <= args.curriculum_max_swaps:
        raise ValueError("curriculum range must satisfy 1 <= min <= max")
    if args.curriculum_steps_per_length < 1:
        raise ValueError("--curriculum-steps-per-length must be positive")
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
        if args.online_training:
            args.train_steps = 2
            args.curriculum_min_swaps = 2
            args.curriculum_max_swaps = 2
            args.curriculum_steps_per_length = 1
    if args.slot_first and args.architecture == "cot":
        raise ValueError("--slot-first is supported for direct/recurrent classifiers, not cot")
    if args.slot_first and args.architecture == "event-recurrent":
        raise ValueError("--slot-first does not apply to separate event-recurrent state registers")
    if args.slot_first and args.architecture == "fan-recurrent":
        raise ValueError("fan-recurrent is causal and requires output SLOT tokens at the end")

    seed_everything(args.seed)
    if args.extended_length and args.data_dir == ROOT / "data":
        args.data_dir = ROOT / "data" / "extended_length"
    required_names = EVAL_SPLITS if args.online_training else ("train", *EVAL_SPLITS)
    required_splits = [args.data_dir / f"{split}.jsonl" for split in required_names]
    missing_splits = [str(path) for path in required_splits if not path.is_file()]
    if missing_splits:
        raise FileNotFoundError(
            "missing dataset split(s): " + ", ".join(missing_splits)
            + ". Generate them with `python -m src.data.data --extended-length --out data/extended_length`."
        )
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
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
        loop_conditioning=args.loop_conditioning,
        residual_scale=args.residual_scale,
        recurrent_blocks=args.recurrent_blocks,
        max_loop_embeddings=args.max_loop_embeddings,
        adaptive_update_threshold=args.adaptive_update_threshold,
        adaptive_min_confidence=args.adaptive_min_confidence,
        fan_input_format=args.fan_input_format,
        fan_positional_control=args.fan_positional_control,
        direct_input_format=args.direct_input_format,
        direct_causal=args.direct_causal,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    args_path = run_dir / "args.json"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    write_json(config_path, config.to_dict())
    write_json(args_path, vars(args))

    model = maybe_compile_model(build_model(config).to(device), device)
    optimizer = _build_optimizer(args, model)
    scheduler: Any | None = None
    scaler = _make_grad_scaler(amp_enabled)
    run_config: dict[str, object] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    history: list[dict[str, object]] = []
    losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    previous_training_seconds = 0.0
    train_dataset: Dataset | None = None
    validation_dataset: Dataset | None = None
    train_loader: Iterable[dict[str, Tensor]] | None = None
    show_progress = not args.no_progress
    started = time.perf_counter()
    if args.online_training:
        if args.resume is not None:
            raise ValueError("online training resume is not supported in this merged path")
        train_stream = DeterministicOnlineBatchStream(
            num_steps=args.train_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            min_swaps=args.curriculum_min_swaps,
            max_swaps=args.curriculum_max_swaps,
            steps_per_length=args.curriculum_steps_per_length,
            slot_first=args.slot_first,
            input_format=(
                args.fan_input_format
                if args.architecture == "fan-recurrent"
                else args.direct_input_format
            ),
        )
        train_loader = train_stream
        curriculum_steps = (
            args.curriculum_max_swaps - args.curriculum_min_swaps
        ) * args.curriculum_steps_per_length
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: fan_learning_rate_multiplier(
                step,
                train_steps=args.train_steps,
                curriculum_steps=curriculum_steps,
            ),
        )
        train_metrics: dict[str, float] = {}
        losses.append(
            train_epoch(
                model,
                train_stream,
                optimizer,
                device,
                args.grad_clip,
                deep_supervision_weight=args.deep_supervision_weight,
                swaps_per_loop=args.swaps_per_loop,
                scheduler=scheduler,
                show_progress=show_progress,
                metrics=train_metrics,
            )
        )
        training_seconds = time.perf_counter() - started
        history.append({
            "epoch": args.train_steps,
            "train_loss": losses[-1],
            "training": train_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": training_seconds,
        })
        best_epoch = args.train_steps
        best_validation_loss = float("nan")
        best_state = save_training_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=args.train_steps,
            history=history,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            training_seconds=training_seconds,
            run_config=run_config,
        )
        _atomic_torch_save(checkpoint_dir / "best.pt", best_state)
        append_jsonl(metrics_path, history[-1])
    else:
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
            slot_first=args.slot_first,
            input_format=args.direct_input_format if args.architecture == "direct" else "template",
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
            slot_first=args.slot_first,
            swaps_per_loop=args.swaps_per_loop,
            input_format=(
                args.direct_input_format if args.architecture == "direct" else "template"
            ),
        )
        scheduler = _build_scheduler(args, optimizer)
        start_epoch = 1
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
            losses = [float(record["train_loss"]) for record in history if "train_loss" in record]
            best_validation_loss = float(state.get("best_validation_loss", float("inf")))
            best_epoch = int(state.get("best_epoch", 0))
            previous_training_seconds = float(state.get("training_seconds", 0.0))
            start_epoch = int(state["epoch"]) + 1
            if "rng_state" in state:
                _restore_rng_state(state["rng_state"])  # type: ignore[arg-type]
            rewrite_metrics_log(metrics_path, history)
        else:
            rewrite_metrics_log(metrics_path, [])

        epoch_iterable: Iterable[int] = range(start_epoch, args.epochs + 1)
        epoch_progress = None
        if show_progress:
            epoch_progress = _progress_bar(
                epoch_iterable,
                total=max(args.epochs - start_epoch + 1, 0),
                desc=f"{args.architecture} seed{args.seed}",
                leave=True,
                position=0,
            )
            epoch_iterable = epoch_progress
        try:
            for epoch in epoch_iterable:
                train_generator.manual_seed(args.seed + epoch)
                if isinstance(getattr(train_loader, "batch_sampler", None), SwapCountBatchSampler):
                    train_loader.batch_sampler.epoch = epoch  # type: ignore[union-attr]
                epoch_started = time.perf_counter()
                train_metrics = {}
                loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    args.grad_clip,
                    deep_supervision_weight=args.deep_supervision_weight,
                    random_loop_range=(random_min, random_max) if args.random_loops else None,
                    swaps_per_loop=args.swaps_per_loop,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    epoch=epoch,
                    show_progress=show_progress,
                    metrics=train_metrics,
                )
                losses.append(loss)
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
                    "train_loss": loss,
                    "train_metrics": train_metrics,
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
                if epoch_progress is not None:
                    epoch_progress.set_postfix(
                        loss=f"{loss:.4f}",
                        slot_accuracy=f"{train_metrics['slot_accuracy']:.4f}",
                        exact_match=f"{train_metrics['exact_match']:.4f}",
                    )
        finally:
            if epoch_progress is not None:
                epoch_progress.close()
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
                slot_first=args.slot_first,
                input_format=(
                    args.fan_input_format
                    if isinstance(model, FanRecurrentTransformer)
                    else args.direct_input_format
                    if isinstance(model, DirectTransformer)
                    else "template"
                ),
            )
            if isinstance(model, FanRecurrentTransformer):
                metrics = evaluate_length_matched_classifier(
                    model,
                    loader,
                    device,
                    swaps_per_loop=args.swaps_per_loop,
                )
            else:
                metrics = evaluate_classifier(
                    model,
                    loader,
                    device,
                    adaptive_kl=args.adaptive_kl_eval,
                    num_loops=args.adaptive_max_loops if args.adaptive_kl_eval else None,
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
            if args.length_matched_eval and not isinstance(model, FanRecurrentTransformer):
                length_matched_loader = make_loader(
                    path,
                    args.architecture,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    seed=args.seed,
                    max_samples=args.max_eval_samples,
                    slot_first=args.slot_first,
                    swaps_per_loop=args.swaps_per_loop,
                    input_format=(
                        args.fan_input_format
                        if isinstance(model, FanRecurrentTransformer)
                        else args.direct_input_format
                        if isinstance(model, DirectTransformer)
                        else "template"
                    ),
                )
                metrics["length_matched_oracle"] = evaluate_length_matched_classifier(
                    model,
                    length_matched_loader,
                    device,
                    swaps_per_loop=args.swaps_per_loop,
                )
            if args.trajectory_probe_eval:
                probe_loop_counts = sorted(set([args.num_loops, *(args.eval_loop_counts or [])]))
                metrics["trajectory_probe"] = {
                    str(loop_count): evaluate_trajectory_probe(
                        model,
                        loader,
                        device,
                        num_loops=loop_count,
                        swaps_per_loop=args.swaps_per_loop,
                    )
                    for loop_count in probe_loop_counts
                }
            if args.event_trajectory_probe:
                assert isinstance(model, EventWiseRecurrentTransformer)
                metrics["event_trajectory_probe"] = evaluate_event_trajectory_probe(
                    model,
                    loader,
                    device,
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
                collate_fn=partial(
                    collate_fn,
                    slot_first=args.slot_first,
                    input_format=(
                        args.fan_input_format
                        if isinstance(model, FanRecurrentTransformer)
                        else args.direct_input_format
                        if isinstance(model, DirectTransformer)
                        else "template"
                    ),
                ),
            )
            if isinstance(model, FanRecurrentTransformer):
                split_metrics[split_name] = evaluate_length_matched_classifier(
                    model,
                    noisy_loader,
                    device,
                    swaps_per_loop=args.swaps_per_loop,
                )
            else:
                split_metrics[split_name] = evaluate_classifier(
                    model,
                    noisy_loader,
                    device,
                    adaptive_kl=args.adaptive_kl_eval,
                    num_loops=args.adaptive_max_loops if args.adaptive_kl_eval else None,
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
            if args.length_matched_eval and not isinstance(model, FanRecurrentTransformer):
                split_metrics[split_name]["length_matched_oracle"] = evaluate_length_matched_classifier(
                    model,
                    noisy_loader,
                    device,
                    swaps_per_loop=args.swaps_per_loop,
                )
            if args.trajectory_probe_eval:
                probe_loop_counts = sorted(set([args.num_loops, *(args.eval_loop_counts or [])]))
                split_metrics[split_name]["trajectory_probe"] = {
                    str(loop_count): evaluate_trajectory_probe(
                        model,
                        noisy_loader,
                        device,
                        num_loops=loop_count,
                        swaps_per_loop=args.swaps_per_loop,
                    )
                    for loop_count in probe_loop_counts
                }
            if args.event_trajectory_probe:
                assert isinstance(model, EventWiseRecurrentTransformer)
                split_metrics[split_name]["event_trajectory_probe"] = evaluate_event_trajectory_probe(
                    model,
                    noisy_loader,
                    device,
                )

    run_finished_at = datetime.now()
    total_seconds = time.perf_counter() - run_started
    fan_track = (
        "fan_atomic_sinusoidal_control"
        if args.architecture == "fan-recurrent"
        and args.fan_input_format == "atomic"
        and args.fan_positional_control
        else "fan_template_sinusoidal_control"
        if args.architecture == "fan-recurrent" and args.fan_positional_control
        else "fan_atomic_nope_control"
        if args.architecture == "fan-recurrent" and args.fan_input_format == "atomic"
        else "fan_aligned"
        if args.architecture == "fan-recurrent"
        else "basic_atomic_nope_causal"
        if args.architecture == "direct"
        and args.direct_input_format == "atomic"
        and args.direct_causal
        and args.position_encoding == "none"
        else "original_team_plan"
    )
    train_sample_count = len(train_dataset) if train_dataset is not None else None
    validation_sample_count = len(validation_dataset) if validation_dataset is not None else None
    optimizer_steps = (
        args.train_steps
        if args.online_training
        else len(history) * len(train_loader) if train_loader is not None else 0
    )
    result = {
        "track": fan_track,
        "run_name": run_name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "total_seconds": total_seconds,
        "config": config.to_dict(),
        "parameters": count_parameters(model),
        "seed": args.seed,
        "training_loss": [record["train_loss"] for record in history],
        "training_history": history,
        "training_seconds": training_seconds,
        "validation": {
            "ratio": args.validation_ratio,
            "train_samples": train_sample_count,
            "validation_samples": validation_sample_count,
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
            "trajectory_probe_eval": args.trajectory_probe_eval,
            "event_trajectory_probe": args.event_trajectory_probe,
            "noop_eval_ratio": args.noop_eval_ratio,
            "loop_conditioning": args.loop_conditioning,
            "residual_scale": args.residual_scale,
            "recurrent_blocks": args.recurrent_blocks,
            "random_loops": args.random_loops,
            "random_loop_range": [random_min, random_max] if args.random_loops else None,
            "swaps_per_loop": args.swaps_per_loop,
            "length_matched_eval": args.length_matched_eval,
            "eval_loop_counts": args.eval_loop_counts,
            "adaptive_max_loops": args.adaptive_max_loops,
            "adaptive_update_threshold": args.adaptive_update_threshold,
            "adaptive_min_confidence": args.adaptive_min_confidence,
            "slot_first": args.slot_first,
            "extended_length": args.extended_length,
            "fan_input_format": args.fan_input_format if args.architecture == "fan-recurrent" else None,
            "fan_positional_control": args.fan_positional_control if args.architecture == "fan-recurrent" else False,
            "direct_input_format": args.direct_input_format if args.architecture == "direct" else None,
            "direct_causal": args.direct_causal if args.architecture == "direct" else False,
        },
        "training_regime": {
            "objective": (
                "final_ce_plus_deep_supervision_ablation"
                if args.deep_supervision_weight > 0.0
                else "final_ce_only"
            ),
            "online_training": args.online_training,
            "optimizer_steps": optimizer_steps,
            "curriculum_min_swaps": args.curriculum_min_swaps if args.online_training else None,
            "curriculum_max_swaps": args.curriculum_max_swaps if args.online_training else None,
            "curriculum_steps_per_length": (
                args.curriculum_steps_per_length if args.online_training else None
            ),
            "online_seed_scheme": (
                "splitmix64(seed, optimizer_step, sample_index)"
                if args.online_training
                else None
            ),
        },
        "paths": {
            "config": str(config_path),
            "args": str(args_path),
            "result": str(result_path),
            "legacy_result": str(result_json_path),
            "checkpoint": str(checkpoint_path),
            "legacy_checkpoint": str(args.output_dir / f"{run_name}.pt"),
        },
        "splits": split_metrics,
    }
    _atomic_torch_save(args.output_dir / f"{run_name}.pt", best_state)
    _atomic_torch_save(checkpoint_path, best_state)
    write_json(result_path, result)
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
