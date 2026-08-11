"""Minimal standard and recurrent Transformers for the controlled experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


Architecture = Literal["standard", "recurrent", "untied"]
LoopConditioning = Literal["none", "learned"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_locations: int = 8
    pad_id: int = 0
    architecture: Architecture = "recurrent"
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    num_layers: int = 6
    train_loops: int = 6
    recurrent_blocks: int = 1
    max_loop_embeddings: int = 64
    loop_conditioning: LoopConditioning = "none"
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.architecture not in ("standard", "recurrent", "untied"):
            raise ValueError(f"unknown architecture: {self.architecture}")
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if min(self.num_layers, self.train_loops, self.num_locations) < 1:
            raise ValueError("layers, loops, and locations must be positive")
        if self.loop_conditioning not in ("none", "learned"):
            raise ValueError(f"unknown loop conditioning: {self.loop_conditioning}")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if self.loop_conditioning == "learned" and self.max_loop_embeddings < self.train_loops:
            raise ValueError("max_loop_embeddings must cover training loops")
        if not 1 <= self.recurrent_blocks <= self.train_loops:
            raise ValueError("recurrent_blocks must be in [1, train_loops]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        table = torch.zeros(length, self.d_model, device=device)
        angles = positions * frequencies
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])
        return table.to(dtype=dtype)


def make_block(config: ModelConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.d_ff,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class StateTrackingTransformer(nn.Module):
    """One interface for standard, tied-recurrent, and untied controls.

    R0 is ``architecture='recurrent', loop_conditioning='none',
    residual_scale=1``.  Its recurrence is exactly ``h <- shared_block(h)``.
    Stabilizers and loop identity are explicit ablations, never hidden defaults.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.positions = SinusoidalPositions(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        if config.architecture == "standard":
            self.blocks = nn.ModuleList(make_block(config) for _ in range(config.num_layers))
        elif config.architecture == "untied":
            self.blocks = nn.ModuleList(make_block(config) for _ in range(config.train_loops))
        else:
            if config.recurrent_blocks == 1:
                self.shared_block = make_block(config)
            else:
                self.recurrent_blocks = nn.ModuleList(
                    make_block(config) for _ in range(config.recurrent_blocks)
                )
        self.loop_embedding = (
            nn.Embedding(config.max_loop_embeddings, config.d_model)
            if config.loop_conditioning == "learned"
            else None
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.GELU(),
            nn.Linear(2 * config.d_model, config.num_locations),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[config.pad_id].zero_()

    def _embed(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be [batch, sequence]")
        valid = attention_mask.bool()
        x = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        x = x + self.positions(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
        x = self.embedding_dropout(x) * valid.unsqueeze(-1)
        return x, ~valid

    def _run_recurrent_step(self, x: Tensor, padding: Tensor, step: int) -> Tensor:
        conditioned = x
        if self.loop_embedding is not None:
            if step >= self.config.max_loop_embeddings:
                raise ValueError("loop count exceeds max_loop_embeddings for learned conditioning")
            conditioned = conditioned + self.loop_embedding.weight[step].view(1, 1, -1)
        block = (
            self.shared_block
            if self.config.recurrent_blocks == 1
            else self.recurrent_blocks[step % self.config.recurrent_blocks]
        )
        candidate = block(conditioned, src_key_padding_mask=padding)
        if self.config.residual_scale == 1.0:
            return candidate
        return x + self.config.residual_scale * (candidate - x)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        num_loops: int | None = None,
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        x, padding = self._embed(input_ids, attention_mask)
        hidden_states: list[Tensor] = []
        if self.config.architecture == "standard":
            if num_loops is not None:
                raise ValueError("num_loops is only defined for iterative architectures")
            for block in self.blocks:
                x = block(x, src_key_padding_mask=padding)
                hidden_states.append(self.final_norm(x))
        elif self.config.architecture == "untied":
            steps = self.config.train_loops if num_loops is None else num_loops
            if not 1 <= steps <= len(self.blocks):
                raise ValueError("untied steps cannot exceed its unique block count")
            for block in self.blocks[:steps]:
                x = block(x, src_key_padding_mask=padding)
                hidden_states.append(self.final_norm(x))
        else:
            steps = self.config.train_loops if num_loops is None else num_loops
            if steps < 1:
                raise ValueError("num_loops must be positive")
            # Unlike the legacy implementation, R0 deliberately permits
            # inference recurrence beyond the training recurrence.
            for step in range(steps):
                x = self._run_recurrent_step(x, padding, step)
                hidden_states.append(self.final_norm(x))
        final = self.final_norm(x)
        return (final, tuple(hidden_states)) if return_hidden_states else final

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        num_loops: int | None = None,
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        encoded = self.encode(
            input_ids,
            attention_mask,
            num_loops=num_loops,
            return_hidden_states=return_hidden_states,
        )
        if return_hidden_states:
            final, hidden_states = encoded
            return self.classifier(final[:, 0]), hidden_states
        return self.classifier(encoded[:, 0])

    @torch.inference_mode()
    def forward_adaptive(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        max_loops: int,
        min_loops: int = 2,
        kl_threshold: float = 1e-3,
        update_threshold: float = 0.05,
        min_confidence: float = 0.5,
        patience: int = 2,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Optional per-sample halting ablation; not part of the R0 main run."""

        if self.config.architecture != "recurrent":
            raise ValueError("adaptive halting requires the recurrent architecture")
        if not 1 <= min_loops <= max_loops or patience < 1:
            raise ValueError("invalid adaptive-loop bounds")
        x, padding = self._embed(input_ids, attention_mask)
        batch_size = x.shape[0]
        halted = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        stable = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        steps_taken = torch.full((batch_size,), max_loops, dtype=torch.long, device=x.device)
        previous_probs: Tensor | None = None
        kl_history, update_history, confidence_history = [], [], []
        logits: Tensor | None = None
        for step in range(max_loops):
            active = ~halted
            candidate = self._run_recurrent_step(x, padding, step)
            updated = torch.where(active.view(-1, 1, 1), candidate, x)
            delta = (updated[:, 0] - x[:, 0]).norm(dim=-1)
            base = x[:, 0].norm(dim=-1).clamp_min(1e-8)
            update_ratio = delta / base
            x = updated
            logits = self.classifier(self.final_norm(x)[:, 0])
            probs = logits.softmax(-1)
            confidence = probs.amax(-1)
            if previous_probs is None:
                symmetric_kl = torch.full_like(confidence, float("nan"))
            else:
                epsilon = torch.finfo(probs.dtype).eps
                p, q = probs.clamp_min(epsilon), previous_probs.clamp_min(epsilon)
                symmetric_kl = 0.5 * (
                    (p * (p.log() - q.log())).sum(-1)
                    + (q * (q.log() - p.log())).sum(-1)
                )
                stable_now = (
                    active
                    & (step + 1 >= min_loops)
                    & (symmetric_kl <= kl_threshold)
                    & (update_ratio <= update_threshold)
                    & (confidence >= min_confidence)
                )
                stable = torch.where(stable_now, stable + 1, torch.where(active, 0, stable))
                new_halts = active & (stable >= patience)
                steps_taken = torch.where(new_halts, step + 1, steps_taken)
                halted |= new_halts
            kl_history.append(symmetric_kl)
            update_history.append(update_ratio)
            confidence_history.append(confidence)
            previous_probs = probs
            if bool(halted.all()):
                break
        assert logits is not None
        return logits, {
            "steps_taken": steps_taken,
            "halted": halted,
            "symmetric_kl": torch.stack(kl_history, 1),
            "update_ratio": torch.stack(update_history, 1),
            "confidence": torch.stack(confidence_history, 1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def closest_parameter_matched_width(
    config: ModelConfig,
    target_parameters: int,
    *,
    max_d_model: int = 1024,
) -> tuple[int, int]:
    """Find the head-compatible width closest to a requested parameter budget."""

    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    best_width, best_count, best_gap = config.n_heads, 0, float("inf")
    for width in range(config.n_heads, max_d_model + 1, config.n_heads):
        candidate = ModelConfig(**{
            **config.to_dict(),
            "d_model": width,
            "d_ff": 4 * width,
        })
        count = count_parameters(StateTrackingTransformer(candidate))
        gap = abs(count - target_parameters)
        if gap < best_gap:
            best_width, best_count, best_gap = width, count, gap
    return best_width, best_count


def estimate_forward_flops(config: ModelConfig, sequence_length: int, *, num_loops: int | None = None) -> int:
    """Approximate one-sample encoder FLOPs for comparison, not hardware billing."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    depth = (
        config.num_layers
        if config.architecture == "standard"
        else (config.train_loops if num_loops is None else num_loops)
    )
    d, ff, length = config.d_model, config.d_ff, sequence_length
    # Q/K/V/output projections + attention score/value products + two FFN projections.
    per_block = 4 * length * d * d + 2 * length * length * d + 2 * length * d * ff
    embedding_and_head = length * d + 3 * d * d + d * config.num_locations
    return int(depth * per_block + embedding_and_head)
