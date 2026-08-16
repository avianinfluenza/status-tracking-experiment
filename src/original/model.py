"""Direct, explicit-CoT, and recurrent models for the original experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from ..data.atomic import ATOMIC_PAD_ID, ATOMIC_VOCAB_SIZE
from ..data.collate import EVENT_WIDTH
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


Architecture = Literal[
    "direct",
    "cot",
    "recurrent",
    "recurrent-r0",
    "fan-recurrent",
    "event-recurrent",
]
PositionEncoding = Literal["none", "sinusoidal", "rope"]
LoopConditioning = Literal["none", "learned"]
FanInputFormat = Literal["template", "atomic"]
DirectInputFormat = Literal["template", "atomic"]


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
    loop_conditioning: LoopConditioning = "none"
    residual_scale: float = 1.0
    recurrent_blocks: int = 1
    max_loop_embeddings: int = 64
    adaptive_update_threshold: float = 1e9
    adaptive_min_confidence: float = 0.0
    fan_input_format: FanInputFormat = "template"
    fan_positional_control: bool = False
    direct_input_format: DirectInputFormat = "template"
    direct_causal: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in (
            "direct",
            "cot",
            "recurrent",
            "recurrent-r0",
            "fan-recurrent",
            "event-recurrent",
        ):
            raise ValueError(f"unknown architecture: {self.architecture}")
        if self.position_encoding not in ("none", "sinusoidal", "rope"):
            raise ValueError(f"unknown position encoding: {self.position_encoding}")
        if self.fan_input_format not in ("template", "atomic"):
            raise ValueError(f"unknown fan_input_format: {self.fan_input_format}")
        if self.direct_input_format not in ("template", "atomic"):
            raise ValueError(f"unknown direct_input_format: {self.direct_input_format}")
        if self.architecture != "direct" and (
            self.direct_input_format != "template" or self.direct_causal
        ):
            raise ValueError("direct input and causal controls require architecture='direct'")
        if self.architecture == "fan-recurrent":
            if self.fan_positional_control:
                if self.position_encoding != "sinusoidal":
                    raise ValueError("fan positional control requires position_encoding='sinusoidal'")
            elif self.position_encoding != "none":
                raise ValueError(
                    "fan-recurrent requires position_encoding='none' unless "
                    "fan_positional_control=True"
                )
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
        if self.loop_conditioning not in ("none", "learned"):
            raise ValueError(f"unknown loop conditioning: {self.loop_conditioning}")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if not 1 <= self.recurrent_blocks <= self.num_loops:
            raise ValueError("recurrent_blocks must be in [1, num_loops]")
        if self.max_loop_embeddings < self.num_loops:
            raise ValueError("max_loop_embeddings must cover training loops")
        if self.adaptive_update_threshold < 0.0:
            raise ValueError("adaptive_update_threshold must be non-negative")
        if not 0.0 <= self.adaptive_min_confidence <= 1.0:
            raise ValueError("adaptive_min_confidence must be in [0, 1]")

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

    def forward(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
        *,
        cache_length: int | None = None,
    ) -> Tensor:
        # The caller knows the required cache length, including incremental
        # decoding offsets. Avoid extracting a CUDA scalar with .item().
        length = position_ids.shape[-1] if cache_length is None else cache_length
        self._ensure(length, position_ids.device)
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
    def __init__(self, config: OriginalModelConfig, vocab_size: int, *, pad_id: int = PAD_ID) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.sinusoidal = SinusoidalPositionEncoding(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

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
        is_atomic = config.direct_input_format == "atomic"
        vocab_size = ATOMIC_VOCAB_SIZE if is_atomic else VOCAB_SIZE
        pad_id = ATOMIC_PAD_ID if is_atomic else PAD_ID
        super().__init__(config, vocab_size, pad_id=pad_id)
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
            h = layer(h, attention_mask, positions, causal=self.config.direct_causal)
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
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Return shared-classifier logits from every recurrent loop.

        ``return_hidden_states`` exists for graph-integrity tests only.  The
        states retain their normal autograd links; targets never enter this
        method or alter the recurrence.
        """

        e, h, positions = self.prepare(input_ids, attention_mask)
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        outputs = []
        hidden_states = []
        for _ in range(loops):
            h = self.recurrent_step(e, h, attention_mask, positions)
            outputs.append(self._classify(h, attention_mask, slot_positions))
            if return_hidden_states:
                hidden_states.append(h)
        loop_logits = torch.stack(outputs, dim=1)
        if return_hidden_states:
            return loop_logits, hidden_states
        return loop_logits

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


class FanRecurrentTransformer(TokenBackbone):
    """Looped Transformer aligned with Fan et al.'s recurrent computation.

    The token embedding is re-injected before every application of the same
    depth stack::

        h_0 = 0
        h_k = F_theta(h_{k-1} + embed(x))

    ``F_theta`` is shared across loops and contains ``num_layers`` causal
    Transformer blocks. The main condition uses NoPE; the explicitly labeled
    sinusoidal positional control is admitted for either input representation
    only through ``fan_positional_control``. Targets never enter the recurrence. The
    default training objective reads only the final state;
    ``forward_all_loops`` remains available for the pre-existing
    deep-supervision ablation and evaluation-only probes.
    """

    def __init__(self, config: OriginalModelConfig) -> None:
        if config.architecture != "fan-recurrent":
            raise ValueError("FanRecurrentTransformer requires architecture='fan-recurrent'")
        is_atomic = config.fan_input_format == "atomic"
        vocab_size = ATOMIC_VOCAB_SIZE if is_atomic else VOCAB_SIZE
        pad_id = ATOMIC_PAD_ID if is_atomic else PAD_ID
        super().__init__(config, vocab_size, pad_id=pad_id)
        self.shared_layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.classifier = Classifier(config.d_model, config.classifier_dim, N_LABELS)

    def recurrent_step(
        self,
        embedding: Tensor,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        hidden = hidden + embedding
        for layer in self.shared_layers:
            hidden = layer(hidden, attention_mask, position_ids, causal=True)
        return hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)

    def _classify(self, hidden: Tensor, attention_mask: Tensor, slot_positions: Tensor) -> Tensor:
        normalized = self.final_norm(hidden) * attention_mask.unsqueeze(-1)
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
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        _, embedding, positions = self.prepare(input_ids, attention_mask)
        hidden = torch.zeros_like(embedding)
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        outputs: list[Tensor] = []
        hidden_states: list[Tensor] = []
        for _ in range(loops):
            hidden = self.recurrent_step(
                embedding,
                hidden,
                attention_mask,
                positions,
            )
            outputs.append(self._classify(hidden, attention_mask, slot_positions))
            if return_hidden_states:
                hidden_states.append(hidden)
        loop_logits = torch.stack(outputs, dim=1)
        if return_hidden_states:
            return loop_logits, hidden_states
        return loop_logits


class RecurrentR0Transformer(TokenBackbone):
    """Pure weight-tied recurrence for ball-swap final-state prediction.

    R0 is the main condition: ``h <- shared_block(h)`` with no input
    reinjection and no loop identity.  Loop embeddings, residual scaling, and
    multi-block sharing are explicit ablations configured on the model config.
    """

    def __init__(self, config: OriginalModelConfig) -> None:
        if config.architecture != "recurrent-r0":
            raise ValueError("RecurrentR0Transformer requires architecture='recurrent-r0'")
        super().__init__(config, VOCAB_SIZE)
        if config.recurrent_blocks == 1:
            self.shared_block = TransformerBlock(config)
        else:
            self.recurrent_blocks = nn.ModuleList(
                TransformerBlock(config) for _ in range(config.recurrent_blocks)
            )
        self.loop_embedding = (
            nn.Embedding(config.max_loop_embeddings, config.d_model)
            if config.loop_conditioning == "learned"
            else None
        )
        self.classifier = Classifier(config.d_model, config.classifier_dim, N_LABELS)

    def _block_for_step(self, loop_index: int) -> TransformerBlock:
        if self.config.recurrent_blocks == 1:
            return self.shared_block
        return self.recurrent_blocks[loop_index % self.config.recurrent_blocks]

    def recurrent_step(
        self,
        h: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        loop_index: int,
    ) -> Tensor:
        conditioned = h
        if self.loop_embedding is not None:
            if loop_index >= self.config.max_loop_embeddings:
                raise ValueError("loop count exceeds max_loop_embeddings")
            conditioned = conditioned + self.loop_embedding.weight[loop_index].view(1, 1, -1)
        candidate = self._block_for_step(loop_index)(
            conditioned, attention_mask, position_ids, causal=False
        )
        if self.config.residual_scale != 1.0:
            candidate = h + self.config.residual_scale * (candidate - h)
        return candidate * attention_mask.unsqueeze(-1).to(dtype=candidate.dtype)

    def _classify(self, h: Tensor, attention_mask: Tensor, slot_positions: Tensor) -> Tensor:
        normalized = self.final_norm(h) * attention_mask.unsqueeze(-1)
        return self.classifier(_gather_slots(normalized, slot_positions))

    def _slot_states(self, h: Tensor, attention_mask: Tensor, slot_positions: Tensor) -> Tensor:
        normalized = self.final_norm(h) * attention_mask.unsqueeze(-1)
        return _gather_slots(normalized, slot_positions)

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
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Return shared-classifier logits without interrupting recurrence.

        Intermediate targets are consumed exclusively by training loss code;
        this forward path never receives them and therefore cannot use teacher
        forcing, detachment, or a hidden-state reset between loops.
        """
        _, h, positions = self.prepare(input_ids, attention_mask)
        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        outputs = []
        hidden_states = []
        for loop_index in range(loops):
            h = self.recurrent_step(
                h, attention_mask, positions, loop_index=loop_index
            )
            outputs.append(self._classify(h, attention_mask, slot_positions))
            if return_hidden_states:
                hidden_states.append(h)
        loop_logits = torch.stack(outputs, dim=1)
        if return_hidden_states:
            return loop_logits, hidden_states
        return loop_logits

    @torch.inference_mode()
    def forward_adaptive(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        slot_positions: Tensor,
        *,
        max_loops: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Stop samples using KL, confidence, update ratio, and patience."""

        _, h, positions = self.prepare(input_ids, attention_mask)
        loops = self.config.num_loops if max_loops is None else max_loops
        if loops < self.config.min_loops:
            raise ValueError("max_loops must be at least min_loops")
        batch_size = input_ids.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        streak = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        steps_taken = torch.full((batch_size,), loops, dtype=torch.long, device=input_ids.device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        kl_history = torch.full((batch_size, loops), torch.inf, device=input_ids.device)
        update_history = torch.full((batch_size, loops), torch.inf, device=input_ids.device)
        confidence_history = torch.zeros((batch_size, loops), device=input_ids.device)
        previous_probabilities: Tensor | None = None
        final_logits: Tensor | None = None

        for loop_index in range(loops):
            previous_slots = self._slot_states(h, attention_mask, slot_positions)
            candidate = self.recurrent_step(
                h, attention_mask, positions, loop_index=loop_index
            )
            h = torch.where(active[:, None, None], candidate, h)
            current_slots = self._slot_states(h, attention_mask, slot_positions)
            delta = (current_slots - previous_slots).norm(dim=-1)
            base = previous_slots.norm(dim=-1).clamp_min(1e-8)
            update_ratio = (delta / base).mean(dim=-1)
            update_history[:, loop_index] = update_ratio

            logits = self._classify(h, attention_mask, slot_positions)
            probabilities = logits.softmax(dim=-1)
            confidence = probabilities.amax(dim=-1).mean(dim=-1)
            confidence_history[:, loop_index] = confidence
            if final_logits is None:
                final_logits = logits.clone()
            final_logits = torch.where(active[:, None, None], logits, final_logits)

            if previous_probabilities is not None:
                kl = symmetric_output_kl(probabilities, previous_probabilities)
                kl_history[:, loop_index] = kl
                eligible = loop_index + 1 >= self.config.min_loops
                stable = (
                    (kl <= self.config.kl_threshold)
                    & (update_ratio <= self.config.adaptive_update_threshold)
                    & (confidence >= self.config.adaptive_min_confidence)
                    & active
                    if eligible else torch.zeros_like(active)
                )
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
            "update_ratio": update_history,
            "confidence": confidence_history,
        }


class EventWiseRecurrentTransformer(nn.Module):
    """Shared one-event transition over five position-free latent registers.

    The state chain is never reset or teacher-forced.  At recurrent step ``t``
    the update block receives only the seven local tokens for event ``t``;
    neither future events nor a global event/loop position are available.
    """

    def __init__(self, config: OriginalModelConfig) -> None:
        super().__init__()
        if config.architecture != "event-recurrent":
            raise ValueError("EventWiseRecurrentTransformer requires architecture='event-recurrent'")
        self.config = config
        self.register_identity = nn.Embedding(N_ENTITIES, config.d_model)
        # The final entry represents an inactive/padded register.
        self.color_embedding = nn.Embedding(N_LABELS + 1, config.d_model)
        self.event_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD_ID)
        self.event_local_position = nn.Parameter(torch.empty(EVENT_WIDTH, config.d_model))
        self.event_type = nn.Parameter(torch.empty(config.d_model))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.shared_update = TransformerBlock(config)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = Classifier(config.d_model, config.classifier_dim, N_LABELS)
        nn.init.normal_(self.register_identity.weight, std=0.02)
        nn.init.normal_(self.color_embedding.weight, std=0.02)
        nn.init.normal_(self.event_embedding.weight, std=0.02)
        nn.init.normal_(self.event_local_position, std=0.02)
        nn.init.normal_(self.event_type, std=0.02)
        with torch.no_grad():
            self.event_embedding.weight[PAD_ID].zero_()

    def initialize_state(self, initial_colors: Tensor, register_mask: Tensor) -> Tensor:
        if initial_colors.ndim != 2 or initial_colors.shape[1] != N_ENTITIES:
            raise ValueError("initial_colors must have shape [B, 5]")
        if register_mask.shape != initial_colors.shape:
            raise ValueError("register_mask must match initial_colors")
        person_ids = torch.arange(N_ENTITIES, device=initial_colors.device).unsqueeze(0)
        state = self.register_identity(person_ids) + self.color_embedding(initial_colors)
        state = self.embedding_dropout(state)
        return state * register_mask.unsqueeze(-1).to(dtype=state.dtype)

    def recurrent_step(
        self,
        state: Tensor,
        event_input_ids: Tensor,
        register_mask: Tensor,
        event_active: Tensor,
    ) -> Tensor:
        """Apply the one shared neural transition to exactly one event."""
        if event_input_ids.ndim != 2 or event_input_ids.shape[1] != EVENT_WIDTH:
            raise ValueError(f"event_input_ids must have shape [B, {EVENT_WIDTH}]")
        event = self.event_embedding(event_input_ids) * math.sqrt(self.config.d_model)
        event = event + self.event_local_position.unsqueeze(0) + self.event_type.view(1, 1, -1)
        event = self.embedding_dropout(event)
        combined = torch.cat((state, event), dim=1)
        attention_mask = torch.cat(
            (
                register_mask,
                event_active.long().unsqueeze(1).expand(-1, EVENT_WIDTH),
            ),
            dim=1,
        )
        # All global positions are deliberately identical.  Event-local order
        # is represented only by event_local_position, which resets each step.
        position_ids = torch.zeros_like(attention_mask)
        candidate = self.shared_update(
            combined,
            attention_mask,
            position_ids,
            causal=False,
        )[:, :N_ENTITIES]
        candidate = candidate * register_mask.unsqueeze(-1).to(dtype=candidate.dtype)
        return torch.where(event_active[:, None, None].bool(), candidate, state)

    def _classify(self, state: Tensor, register_mask: Tensor) -> Tensor:
        normalized = self.final_norm(state) * register_mask.unsqueeze(-1).to(dtype=state.dtype)
        return self.classifier(normalized)

    def classify_initial_state(self, initial_colors: Tensor, register_mask: Tensor) -> Tensor:
        """Read the event-free registers with the same head used after every event."""

        return self._classify(
            self.initialize_state(initial_colors, register_mask),
            register_mask,
        )

    def forward_all_events(
        self,
        initial_colors: Tensor,
        register_mask: Tensor,
        event_input_ids: Tensor,
        event_mask: Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if event_input_ids.ndim != 3 or event_input_ids.shape[2] != EVENT_WIDTH:
            raise ValueError(f"event_input_ids must have shape [B, T, {EVENT_WIDTH}]")
        if event_mask.shape != event_input_ids.shape[:2]:
            raise ValueError("event_mask must have shape [B, T]")
        if bool((event_mask.sum(dim=1) < 1).any()):
            raise ValueError("every sample must contain at least one event")
        state = self.initialize_state(initial_colors, register_mask)
        outputs = []
        hidden_states = []
        for event_index in range(event_input_ids.shape[1]):
            state = self.recurrent_step(
                state,
                event_input_ids[:, event_index],
                register_mask,
                event_mask[:, event_index],
            )
            outputs.append(self._classify(state, register_mask))
            if return_hidden_states:
                hidden_states.append(state)
        event_logits = torch.stack(outputs, dim=1)
        if return_hidden_states:
            return event_logits, hidden_states
        return event_logits

    def forward(
        self,
        initial_colors: Tensor,
        register_mask: Tensor,
        event_input_ids: Tensor,
        event_mask: Tensor,
    ) -> Tensor:
        outputs = self.forward_all_events(
            initial_colors,
            register_mask,
            event_input_ids,
            event_mask,
        )
        assert isinstance(outputs, Tensor)
        return outputs[:, -1]


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
            h = h + self.sinusoidal(
                positions,
                h.dtype,
                cache_length=start_position + length,
            )
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


OriginalModel = (
    DirectTransformer
    | ExplicitCoTTransformer
    | RecurrentTransformer
    | FanRecurrentTransformer
    | RecurrentR0Transformer
    | EventWiseRecurrentTransformer
)


def build_model(config: OriginalModelConfig) -> OriginalModel:
    if config.architecture == "direct":
        return DirectTransformer(config)
    if config.architecture == "cot":
        return ExplicitCoTTransformer(config)
    if config.architecture == "recurrent-r0":
        return RecurrentR0Transformer(config)
    if config.architecture == "fan-recurrent":
        return FanRecurrentTransformer(config)
    if config.architecture == "event-recurrent":
        return EventWiseRecurrentTransformer(config)
    return RecurrentTransformer(config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
