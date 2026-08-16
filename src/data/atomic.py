"""Atomic structured serialization for the Fan-style control condition.

The regular ball-swap input is a Korean template whose initial assignments and
events span six and seven tokens respectively.  This module provides a second,
fixed vocabulary where each semantic assignment or event occupies one token.
It deliberately preserves the *same* symbolic task and final five colour
labels; only the input representation changes.
"""

from __future__ import annotations

from typing import Sequence

from .vocab import N_ENTITIES, N_LABELS


# Keep PAD at zero so the atomic embedding can use PyTorch's padding_idx
# directly, independently of the legacy template vocabulary's PAD id.
ATOMIC_PAD_ID = 0
_INIT_OFFSET = 1
_SWAP_OFFSET = _INIT_OFFSET + N_ENTITIES * N_LABELS
_SLOT_OFFSET = _SWAP_OFFSET + N_ENTITIES * N_ENTITIES
ATOMIC_VOCAB_SIZE = _SLOT_OFFSET + N_ENTITIES


def _entity_color_token(entity: int, color: int) -> int:
    if not 0 <= entity < N_ENTITIES or not 0 <= color < N_LABELS:
        raise ValueError("atomic initial assignment contains an invalid entity or colour")
    return _INIT_OFFSET + entity * N_LABELS + color


def _swap_token(left: int, right: int) -> int:
    if not 0 <= left < N_ENTITIES or not 0 <= right < N_ENTITIES:
        raise ValueError("atomic swap contains an invalid entity")
    # Preserve source order exactly, even though the symbolic state update is
    # symmetric. This keeps the control's input distribution aligned with the
    # original templated serialization.
    return _SWAP_OFFSET + left * N_ENTITIES + right


ATOMIC_SLOT_IDS = tuple(_SLOT_OFFSET + entity for entity in range(N_ENTITIES))


def encode_atomic_body(init: Sequence[int], swaps: Sequence[Sequence[int]]) -> list[int]:
    """Encode initial assignments and swaps as one semantic token each."""

    if len(init) != N_ENTITIES:
        raise ValueError("atomic serialization requires one initial colour per entity")
    tokens = [_entity_color_token(entity, int(color)) for entity, color in enumerate(init)]
    for pair in swaps:
        if len(pair) != 2:
            raise ValueError("each atomic swap must contain exactly two entities")
        tokens.append(_swap_token(int(pair[0]), int(pair[1])))
    return tokens
