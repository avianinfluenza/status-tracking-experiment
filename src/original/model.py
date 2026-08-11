"""Direct, explicit-CoT, and recurrent models for the original experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from ..data.vocab import N_ENTITIES, N_LABELS, PAD_ID, VOCAB_SIZE
from ..model.classifier import Classifier
from .data import (
    BOS_ID,
    COLOR_IDS,
    COT_VOCAB_SIZE,
    END_STATE_ID,
    SLOT_TOKEN_IDS,
    STATE_ID,
    encode_initial,
    encode_swap,
)


Architecture = Literal["direct", "cot", "recurrent"]
PositionEncoding = Literal["sinusoidal", "rope"]


@dataclass(frozen=True)
class OriginalModelConfig:
    architecture: Architecture
    position_encoding: PositionEncoding = "sinusoidal"
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    num_layers: int = 6
    num_loops: int = 6
    classifier_dim: int = 2
    kl_threshold: float = 1e-3
    min_loops: int = 2
    halting_patience: int = 1

    def __post_init__(self) -> None:
        if self.architecture not in ("direct", "cot", "recurrent"):
            raise ValueError(f"unknown architecture: {self.architecture}")
        if self.position_encoding not in ("sinusoidal", "rope"):
            raise ValueError(f"unknown position encoding: {self.position_encoding}")
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.d_ff <= 0 or self.num_layers < 1 or self.num_loops < 1:
            raise ValueError("d_ff, num_layers, and num_loops must be positive")
        if self.classifier_dim < 1:
            raise ValueError("classifier_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.kl_threshold < 0.0:
            raise ValueError("kl_threshold must be non-negative")
        if not 1 <= self.min_loops <= self.num_loops:
            raise ValueError("min_loops must be in [1, num_loops]")
        if self.halting_patience < 1:
            raise ValueError("halting_patience must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("cache", torch.empty(0, d_model), persistent=False)

    def _ensure(self, length: int, device: torch.device) -> None:
        if self.cache.device == device and self.cache.shape[0] >= length:
            return
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        angles = position * frequency
        table = torch.zeros(length, self.d_model, device=device)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles[:, : table[:, 1::2].shape[1]].cos()
        self.cache = table

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> Tensor:
        self._ensure(int(position_ids.max().item()) + 1, position_ids.device)
        return self.cache[position_ids].to(dtype=dtype)


def _logical_positions(attention_mask: Tensor) -> Tensor:
    positions = attention_mask.long().cumsum(dim=1) - 1
    return positions.clamp_min(0)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    head_dim = q.shape[-1]
    inv_frequency = 1.0 / (
        10_000
        ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv_frequency
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    cos = angles.cos().to(dtype=q.dtype)
    sin = angles.sin().to(dtype=q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SelfAttention(nn.Module):
    def __init__(self, config: OriginalModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.position_encoding = config.position_encoding
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        causal: bool,
    ) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        if self.position_encoding == "rope":
            q, k = _apply_rope(q, k, position_ids)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask.bool()[:, None, None, :], -torch.inf)
        if causal:
            future = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future[None, None], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        values = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(values)

    def incremental(
        self,
        x: Tensor,
        position_ids: Tensor,
        cache: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend new causal tokens to a cached prefix without recomputation."""

        batch, new_length, width = x.shape
        qkv = self.qkv(x).view(batch, new_length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        if self.position_encoding == "rope":
            q, k = _apply_rope(q, k, position_ids)
        past_length = 0 if cache is None else cache[0].shape[-2]
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = past_length + torch.arange(new_length, device=x.device)
        key_positions = torch.arange(k.shape[-2], device=x.device)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(future[None, None], -torch.inf)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        values = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, new_length, width)
        return self.output(values), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, config: OriginalModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        causal: bool,
    ) -> Tensor:
        x = x + self.dropout(
            self.attention(self.norm1(x), attention_mask, position_ids, causal=causal)
        )
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)

    def incremental(
        self,
        x: Tensor,
        position_ids: Tensor,
        cache: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, new_cache = self.attention.incremental(self.norm1(x), position_ids, cache)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, new_cache


class TokenBackbone(nn.Module):
    def __init__(self, config: OriginalModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=PAD_ID)
        self.sinusoidal = SinusoidalPositionEncoding(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

    def prepare(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have shape [B, L]")
        positions = _logical_positions(attention_mask)
        e = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        h = e
        if self.config.position_encoding == "sinusoidal":
            h = h + self.sinusoidal(positions, h.dtype) * attention_mask.unsqueeze(-1)
        return e, self.embedding_dropout(h), positions


def _gather_slots(hidden: Tensor, slot_positions: Tensor) -> Tensor:
    indices = slot_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(1, indices)


class DirectTransformer(TokenBackbone):
    """Standard Transformer with independent layers and direct five-slot output."""

    def __init__(self, config: OriginalModelConfig) -> None:
        if config.architecture != "direct":
            raise ValueError("DirectTransformer requires architecture='direct'")
        super().__init__(config, VOCAB_SIZE)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.classifier = Classifier(config.d_model, config.classifier_dim, N_LABELS)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        slot_positions: Tensor,
    ) -> Tensor:
        _, h, positions = self.prepare(input_ids, attention_mask)
        for layer in self.layers:
            h = layer(h, attention_mask, positions, causal=False)
        h = self.final_norm(h) * attention_mask.unsqueeze(-1)
        return self.classifier(_gather_slots(h, slot_positions))


def symmetric_output_kl(current: Tensor, previous: Tensor) -> Tensor:
    """Mean symmetric KL over the five output slots, one value per sample."""

    current = current.clamp_min(1e-8)
    previous = previous.clamp_min(1e-8)
    forward = (current * (current.log() - previous.log())).sum(dim=-1)
    backward = (previous * (previous.log() - current.log())).sum(dim=-1)
    return 0.5 * (forward + backward).mean(dim=-1)


class RecurrentTransformer(TokenBackbone):
    """One shared block with the original ``h = e + block(h)`` update."""

    def __init__(self, config: OriginalModelConfig) -> None:
        if config.architecture != "recurrent":
            raise ValueError("RecurrentTransformer requires architecture='recurrent'")
        super().__init__(config, VOCAB_SIZE)
        self.shared_block = TransformerBlock(config)
        self.classifier = Classifier(config.d_model, config.classifier_dim, N_LABELS)

    def recurrent_step(
        self,
        e: Tensor,
        h: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        # This line is intentionally the exact update proposed in the team plan.
        h = e + self.shared_block(h, attention_mask, position_ids, causal=False)
        return h * attention_mask.unsqueeze(-1).to(dtype=h.dtype)

    def _classify(self, h: Tensor, attention_mask: Tensor, slot_positions: Tensor) -> Tensor:
        normalized = self.final_norm(h) * attention_mask.unsqueeze(-1)
        return self.classifier(_gather_slots(normalized, slot_positions))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        slot_positions: Tensor,
        *,
        num_loops: int | None = None,
    ) -> Tensor:
        return self.forward_all_loops(
            input_ids,
            attention_mask,
            slot_positions,
            num_loops=num_loops,
        )[:, -1]

    def forward_all_loops(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        slot_positions: Tensor,
        *,
        num_loops: int | None = None,
    ) -> Tensor:
        """Return slot logits from every loop for optional deep supervision."""

        e, h, positions = self.prepare(input_ids, attention_mask)
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        outputs = []
        for _ in range(loops):
            h = self.recurrent_step(e, h, attention_mask, positions)
            outputs.append(self._classify(h, attention_mask, slot_positions))
        return torch.stack(outputs, dim=1)

    @torch.inference_mode()
    def forward_adaptive(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        slot_positions: Tensor,
        *,
        max_loops: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Stop samples using only consecutive output-distribution KL.

        No confidence, hidden-update norm, target, or oracle length enters the
        decision.  ``halting_patience`` merely requires the same KL condition
        for consecutive steps.
        """

        e, h, positions = self.prepare(input_ids, attention_mask)
        loops = self.config.num_loops if max_loops is None else max_loops
        if loops < self.config.min_loops:
            raise ValueError("max_loops must be at least min_loops")
        batch_size = input_ids.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        streak = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        steps_taken = torch.full((batch_size,), loops, dtype=torch.long, device=input_ids.device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        kl_history = torch.full((batch_size, loops), torch.inf, device=input_ids.device)
        previous_probabilities: Tensor | None = None
        final_logits: Tensor | None = None

        for loop_index in range(loops):
            candidate = self.recurrent_step(e, h, attention_mask, positions)
            h = torch.where(active[:, None, None], candidate, h)
            logits = self._classify(h, attention_mask, slot_positions)
            probabilities = logits.softmax(dim=-1)
            if final_logits is None:
                final_logits = logits.clone()
            final_logits = torch.where(active[:, None, None], logits, final_logits)

            if previous_probabilities is not None:
                kl = symmetric_output_kl(probabilities, previous_probabilities)
                kl_history[:, loop_index] = kl
                eligible = loop_index + 1 >= self.config.min_loops
                stable = (kl <= self.config.kl_threshold) & active if eligible else torch.zeros_like(active)
                streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
                newly_halted = active & (streak >= self.config.halting_patience)
                steps_taken = torch.where(
                    newly_halted,
                    torch.full_like(steps_taken, loop_index + 1),
                    steps_taken,
                )
                halted |= newly_halted
                active &= ~newly_halted
            previous_probabilities = probabilities
            if not bool(active.any()):
                break

        assert final_logits is not None
        return final_logits, {
            "steps_taken": steps_taken,
            "halted": halted,
            "symmetric_kl": kl_history,
        }


class ExplicitCoTTransformer(TokenBackbone):
    """Causal Transformer that externalizes the full state after every swap."""

    def __init__(self, config: OriginalModelConfig) -> None:
        if config.architecture != "cot":
            raise ValueError("ExplicitCoTTransformer requires architecture='cot'")
        super().__init__(config, COT_VOCAB_SIZE)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.lm_head = nn.Linear(config.d_model, COT_VOCAB_SIZE, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        _, h, positions = self.prepare(input_ids, attention_mask)
        for layer in self.layers:
            h = layer(h, attention_mask, positions, causal=True)
        return self.lm_head(self.final_norm(h))

    def _incremental(
        self,
        input_ids: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
        start_position: int,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Decode a new token chunk against per-layer K/V caches."""

        batch, length = input_ids.shape
        positions = torch.arange(
            start_position,
            start_position + length,
            device=input_ids.device,
            dtype=torch.long,
        ).expand(batch, -1)
        h = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        if self.config.position_encoding == "sinusoidal":
            h = h + self.sinusoidal(positions, h.dtype)
        h = self.embedding_dropout(h)
        new_caches: list[tuple[Tensor, Tensor]] = []
        for layer, cache in zip(self.layers, caches, strict=True):
            h, new_cache = layer.incremental(h, positions, cache)
            new_caches.append(new_cache)
        return self.lm_head(self.final_norm(h)), new_caches

    @torch.inference_mode()
    def generate_states(self, rows: Sequence[dict[str, object]]) -> Tensor:
        """Generate every intermediate state for an equal-swap-count row batch."""

        if not rows:
            return torch.empty(0, N_ENTITIES, dtype=torch.long, device=self.embedding.weight.device)
        swap_counts = {len(row["swaps"]) for row in rows}  # type: ignore[arg-type]
        if len(swap_counts) != 1:
            raise ValueError("rows must have the same number of swaps for batched generation")
        device = self.embedding.weight.device
        initial_tokens = [[BOS_ID, *encode_initial(row["init"])] for row in rows]  # type: ignore[arg-type]
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.layers)
        decoded_length = 0

        def consume(token_rows: Sequence[Sequence[int]]) -> Tensor:
            nonlocal caches, decoded_length
            widths = {len(tokens) for tokens in token_rows}
            if len(widths) != 1:
                raise ValueError("incremental token chunks must have equal lengths")
            input_ids = torch.tensor(token_rows, dtype=torch.long, device=device)
            logits, caches = self._incremental(input_ids, caches, decoded_length)
            decoded_length += input_ids.shape[1]
            return logits[:, -1]

        consume(initial_tokens)
        final_predictions: Tensor | None = None

        for swap_index in range(next(iter(swap_counts))):
            event_chunks = [
                [*encode_swap(row["swaps"][swap_index]), STATE_ID]  # type: ignore[index]
                for row in rows
            ]
            consume(event_chunks)
            predictions: list[Tensor] = []
            for slot_id in SLOT_TOKEN_IDS:
                prompt_logits = consume([[slot_id] for _ in rows])
                color_logits = prompt_logits[:, list(COLOR_IDS)]
                color_index = color_logits.argmax(dim=-1)
                predictions.append(color_index)
                consume([[COLOR_IDS[prediction]] for prediction in color_index.tolist()])
            final_predictions = torch.stack(predictions, dim=1)
            consume([[END_STATE_ID] for _ in rows])

        if final_predictions is None:
            raise ValueError("the original dataset must contain at least one swap")
        return final_predictions


OriginalModel = DirectTransformer | ExplicitCoTTransformer | RecurrentTransformer


def build_model(config: OriginalModelConfig) -> OriginalModel:
    if config.architecture == "direct":
        return DirectTransformer(config)
    if config.architecture == "cot":
        return ExplicitCoTTransformer(config)
    return RecurrentTransformer(config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
