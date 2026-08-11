"""Transformer models for the ball-swap state-tracking experiment.

Both encoders intentionally expose the same interface:

    hidden = encoder(input_ids, attn_mask)  # [batch, sequence, d_model]

The :class:`StateTrackingModel` wrapper gathers the five slot positions and
applies one shared color classifier.  This keeps the comparison focused on the
encoder architecture rather than on a different output head.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

try:  # Support both ``python -m src.train`` and ``python src/train.py``.
    from .classifier import Classifier
    from .vocab import N_LABELS, PAD_ID, VOCAB_SIZE
except ImportError:  # pragma: no cover - exercised by script-style execution.
    from classifier import Classifier
    from vocab import N_LABELS, PAD_ID, VOCAB_SIZE


ModelType = Literal["vanilla", "recurrent"]
LoopConditioning = Literal["sinusoidal", "none"]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by training and checkpoints."""

    model_type: ModelType
    vocab_size: int = VOCAB_SIZE
    n_labels: int = N_LABELS
    pad_id: int = PAD_ID
    d_model: int = 128
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.0
    num_layers: int = 4
    recurrent_steps: int = 4
    norm_first: bool = True
    classifier_dim: int = 2
    loop_conditioning: LoopConditioning = "none"
    residual_scale: float = 1.0
    min_recurrent_steps: int = 1
    halting_threshold: float = 1e-3
    halting_patience: int = 2
    halting_min_confidence: float = 0.5
    halting_update_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.model_type not in ("vanilla", "recurrent"):
            raise ValueError(f"unknown model_type: {self.model_type}")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.recurrent_steps < 1:
            raise ValueError("recurrent_steps must be at least 1")
        if self.classifier_dim < 1:
            raise ValueError("classifier_dim must be at least 1")
        if self.loop_conditioning not in ("sinusoidal", "none"):
            raise ValueError(f"unknown loop_conditioning: {self.loop_conditioning}")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if not 1 <= self.min_recurrent_steps <= self.recurrent_steps:
            raise ValueError("min_recurrent_steps must be in [1, recurrent_steps]")
        if self.halting_threshold < 0.0:
            raise ValueError("halting_threshold must be non-negative")
        if self.halting_patience < 1:
            raise ValueError("halting_patience must be at least 1")
        if not 0.0 <= self.halting_min_confidence <= 1.0:
            raise ValueError("halting_min_confidence must be in [0, 1]")
        if self.halting_update_threshold < 0.0:
            raise ValueError("halting_update_threshold must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SinusoidalEncoding(nn.Module):
    """Length-unbounded sinusoidal encodings with a lazily grown cache."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("_table", torch.empty(0, d_model), persistent=False)

    def _build(self, length: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        table = torch.zeros(length, self.d_model, device=device, dtype=torch.float32)
        angles = positions * frequencies
        table[:, 0::2] = torch.sin(angles)
        if self.d_model > 1:
            table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])
        return table

    def _ensure_length(self, length: int, device: torch.device) -> None:
        if self._table.device != device or self._table.shape[0] < length:
            self._table = self._build(length, device)

    def lookup(self, position_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
        if position_ids.numel() == 0:
            return torch.empty(*position_ids.shape, self.d_model, device=position_ids.device, dtype=dtype)
        max_position = int(position_ids.max().item()) + 1
        self._ensure_length(max_position, position_ids.device)
        return self._table[position_ids].to(dtype=dtype)


def _make_encoder_layer(config: ModelConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=config.norm_first,
    )


class BaseStateEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.position_encoding = SinusoidalEncoding(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_id].zero_()

    def _embed(self, input_ids: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if input_ids.ndim != 2 or attn_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attn_mask must both have shape [B, L]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long dtype")

        valid = attn_mask.to(dtype=torch.bool)
        key_padding_mask = ~valid

        # PAD appears before SLOT tokens.  Physical indices would therefore make
        # one sample's SLOT positions depend on the longest peer in its batch.
        # Cumulative valid-token indices keep positions batch-composition invariant.
        logical_positions = attn_mask.to(dtype=torch.long).cumsum(dim=1) - 1
        logical_positions.clamp_(min=0)

        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        positions = self.position_encoding.lookup(logical_positions, dtype=x.dtype)
        x = x + positions * valid.unsqueeze(-1)
        return self.embedding_dropout(x), key_padding_mask, valid.unsqueeze(-1)


class VanillaTransformerEncoder(BaseStateEncoder):
    """A stack of independent Transformer blocks."""

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type != "vanilla":
            raise ValueError("VanillaTransformerEncoder requires model_type='vanilla'")
        super().__init__(config)
        # Construct each layer separately.  This avoids identical initial values
        # while keeping all parameters independent.
        self.layers = nn.ModuleList(_make_encoder_layer(config) for _ in range(config.num_layers))

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        x, key_padding_mask, valid = self._embed(input_ids, attn_mask)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return self.final_norm(x) * valid


class RecurrentTransformerEncoder(BaseStateEncoder):
    """One conditioned, damped Transformer block applied for ``T`` steps.

    Weight sharing provides the parameter efficiency being tested, while the
    explicit step signal prevents every recurrence from being forced to perform
    an indistinguishable computation. The outer residual scale damps the
    repeated nonlinear map and limits fixed-point collapse or unstable growth.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type != "recurrent":
            raise ValueError("RecurrentTransformerEncoder requires model_type='recurrent'")
        super().__init__(config)
        self.shared_layer = _make_encoder_layer(config)
        # Non-learned conditioning preserves exact trainable-parameter matching
        # with a one-layer vanilla encoder.
        self.timestep_encoding = SinusoidalEncoding(config.d_model)

    def prepare(self, input_ids: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Embed a batch and expose masks for fixed or adaptive recurrence."""

        return self._embed(input_ids, attn_mask)

    def recurrent_step(
        self,
        x: Tensor,
        key_padding_mask: Tensor,
        valid: Tensor,
        step: int,
    ) -> tuple[Tensor, Tensor]:
        """Run one loop and return the damped state plus the raw update."""

        if step < 1:
            raise ValueError("step must be one-indexed and positive")
        conditioned = x
        if self.config.loop_conditioning == "sinusoidal":
            step_id = torch.tensor([step], device=x.device, dtype=torch.long)
            step_vector = self.timestep_encoding.lookup(step_id, dtype=x.dtype)[0]
            conditioned = conditioned + step_vector.view(1, 1, -1) * valid

        candidate = self.shared_layer(conditioned, src_key_padding_mask=key_padding_mask)
        raw_update = (candidate - x) * valid
        x = (x + self.config.residual_scale * raw_update) * valid
        return x, raw_update

    def normalize(self, x: Tensor, valid: Tensor) -> Tensor:
        return self.final_norm(x) * valid

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        *,
        recurrent_steps: int | None = None,
    ) -> Tensor:
        x, key_padding_mask, valid = self.prepare(input_ids, attn_mask)
        steps = self.config.recurrent_steps if recurrent_steps is None else recurrent_steps
        if steps < 1:
            raise ValueError("recurrent_steps must be positive")
        for step in range(1, steps + 1):
            x, _ = self.recurrent_step(x, key_padding_mask, valid, step)
        return self.normalize(x, valid)


def gather_slot_states(hidden: Tensor, slot_pos: Tensor) -> Tensor:
    """Gather ``[B, n_slots, D]`` states from full sequence outputs."""

    if hidden.ndim != 3 or slot_pos.ndim != 2 or hidden.shape[0] != slot_pos.shape[0]:
        raise ValueError("hidden must be [B, L, D] and slot_pos must be [B, n_slots]")
    indices = slot_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(dim=1, index=indices)


class StateTrackingModel(nn.Module):
    """Encoder plus the shared five-color classification head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.model_type == "vanilla":
            self.encoder: BaseStateEncoder = VanillaTransformerEncoder(config)
        else:
            self.encoder = RecurrentTransformerEncoder(config)
        self.classifier = Classifier(
            d_model=config.d_model,
            classifier_dim=config.classifier_dim,
            n_class=config.n_labels,
        )

    def encode(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        *,
        recurrent_steps: int | None = None,
    ) -> Tensor:
        if isinstance(self.encoder, RecurrentTransformerEncoder):
            return self.encoder(input_ids, attn_mask, recurrent_steps=recurrent_steps)
        if recurrent_steps is not None:
            raise ValueError("recurrent_steps override is only valid for the recurrent model")
        return self.encoder(input_ids, attn_mask)

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        slot_pos: Tensor,
        *,
        recurrent_steps: int | None = None,
    ) -> Tensor:
        hidden = self.encode(input_ids, attn_mask, recurrent_steps=recurrent_steps)
        return self.classifier(gather_slot_states(hidden, slot_pos))

    @torch.inference_mode()
    def forward_adaptive(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        slot_pos: Tensor,
        *,
        max_steps: int | None = None,
        min_steps: int | None = None,
        threshold: float | None = None,
        patience: int | None = None,
        min_confidence: float | None = None,
        update_threshold: float | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run per-sample adaptive halting from consecutive slot distributions.

        A sample halts only after its mean symmetric KL divergence remains below
        ``threshold`` for ``patience`` consecutive comparisons. Halted states
        are frozen while unfinished samples keep iterating.
        """

        if not isinstance(self.encoder, RecurrentTransformerEncoder):
            raise ValueError("adaptive halting is only valid for the recurrent model")

        encoder = self.encoder
        max_steps = self.config.recurrent_steps if max_steps is None else max_steps
        min_steps = self.config.min_recurrent_steps if min_steps is None else min_steps
        threshold = self.config.halting_threshold if threshold is None else threshold
        patience = self.config.halting_patience if patience is None else patience
        min_confidence = (
            self.config.halting_min_confidence if min_confidence is None else min_confidence
        )
        update_threshold = (
            self.config.halting_update_threshold
            if update_threshold is None
            else update_threshold
        )
        if not 1 <= min_steps <= max_steps <= self.config.recurrent_steps:
            raise ValueError(
                "adaptive steps must satisfy 1 <= min_steps <= max_steps <= configured steps"
            )
        if threshold < 0.0 or patience < 1:
            raise ValueError("threshold must be non-negative and patience positive")
        if not 0.0 <= min_confidence <= 1.0 or update_threshold < 0.0:
            raise ValueError("confidence must be in [0, 1] and update threshold non-negative")

        x, key_padding_mask, valid = encoder.prepare(input_ids, attn_mask)
        batch_size = input_ids.shape[0]
        halted = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        stable_counts = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        steps_taken = torch.full(
            (batch_size,), max_steps, dtype=torch.long, device=input_ids.device
        )
        previous_probs: Tensor | None = None
        kl_history: list[Tensor] = []
        update_ratio_history: list[Tensor] = []
        confidence_history: list[Tensor] = []
        logits: Tensor | None = None

        for step in range(1, max_steps + 1):
            active = ~halted
            previous_x = x
            candidate, _ = encoder.recurrent_step(x, key_padding_mask, valid, step)
            x = torch.where(active.view(-1, 1, 1), candidate, x)

            update_norm = (x - previous_x).flatten(1).norm(dim=1)
            state_norm = previous_x.flatten(1).norm(dim=1).clamp_min(1e-8)
            update_ratio = update_norm / state_norm
            update_ratio_history.append(update_ratio)

            hidden = encoder.normalize(x, valid)
            logits = self.classifier(gather_slot_states(hidden, slot_pos))
            probs = logits.softmax(dim=-1)
            confidence = probs.amax(dim=-1).mean(dim=1)
            confidence_history.append(confidence)
            if previous_probs is None:
                symmetric_kl = torch.full(
                    (batch_size,), float("nan"), device=input_ids.device, dtype=probs.dtype
                )
            else:
                epsilon = torch.finfo(probs.dtype).eps
                p = probs.clamp_min(epsilon)
                q = previous_probs.clamp_min(epsilon)
                kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
                kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
                symmetric_kl = (0.5 * (kl_pq + kl_qp)).mean(dim=1)

                stable_now = (
                    active
                    & (step >= min_steps)
                    & (symmetric_kl <= threshold)
                    & (confidence >= min_confidence)
                    & (update_ratio <= update_threshold)
                )
                stable_counts = torch.where(
                    stable_now,
                    stable_counts + 1,
                    torch.where(active, torch.zeros_like(stable_counts), stable_counts),
                )
                newly_halted = active & (stable_counts >= patience)
                steps_taken = torch.where(
                    newly_halted,
                    torch.full_like(steps_taken, step),
                    steps_taken,
                )
                halted = halted | newly_halted

            kl_history.append(symmetric_kl)
            previous_probs = probs
            if bool(halted.all()):
                break

        assert logits is not None
        diagnostics = {
            "steps_taken": steps_taken,
            "halted": halted,
            "symmetric_kl": torch.stack(kl_history, dim=1),
            "update_ratio": torch.stack(update_ratio_history, dim=1),
            "confidence": torch.stack(confidence_history, dim=1),
        }
        return logits, diagnostics


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
