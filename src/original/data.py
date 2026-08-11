"""Data views for the original five-person ball-swap experiments.

Direct and recurrent models reuse the repository's fixed 23-token input.  The
explicit-CoT model uses three additional control tokens and writes a complete
five-person state after every swap.  Gold trace tokens are used only for
teacher-forced training; evaluation generates them autoregressively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..collate import BallSwapDataset, encode_body
from ..vocab import COLORS, N_ENTITIES, PAD_ID, SLOTS, TOK2ID, VOCAB_SIZE


# Keep src.vocab's ID order untouched so old checkpoints and data remain valid.
BOS_ID = VOCAB_SIZE
STATE_ID = VOCAB_SIZE + 1
END_STATE_ID = VOCAB_SIZE + 2
COT_VOCAB_SIZE = VOCAB_SIZE + 3
COLOR_IDS = tuple(TOK2ID[color] for color in COLORS)
SLOT_TOKEN_IDS = tuple(TOK2ID[slot] for slot in SLOTS)


def replay_states(init: Sequence[int], swaps: Sequence[Sequence[int]]) -> list[list[int]]:
    """Return the full symbolic state after each swap."""

    if len(init) != N_ENTITIES:
        raise ValueError(f"the original experiment requires exactly {N_ENTITIES} entities")
    state = list(init)
    trace: list[list[int]] = []
    for pair in swaps:
        if len(pair) != 2:
            raise ValueError("each swap must contain two entity indices")
        left, right = map(int, pair)
        if not 0 <= left < N_ENTITIES or not 0 <= right < N_ENTITIES:
            raise ValueError("swap entity index out of range")
        state[left], state[right] = state[right], state[left]
        trace.append(list(state))
    return trace


def encode_initial(init: Sequence[int]) -> list[int]:
    """Encode only the five initial-state sentences."""

    return encode_body(init, [])


def encode_swap(pair: Sequence[int]) -> list[int]:
    """Encode one natural-language swap sentence using the base tokenizer."""

    return encode_body([], [pair])


def state_trace_tokens(state: Sequence[int]) -> list[int]:
    tokens = [STATE_ID]
    for slot_id, color in zip(SLOT_TOKEN_IDS, state, strict=True):
        tokens.extend((slot_id, COLOR_IDS[int(color)]))
    tokens.append(END_STATE_ID)
    return tokens


@dataclass(frozen=True)
class CoTExample:
    input_ids: list[int]
    lm_labels: list[int]
    final_labels: list[int]
    n_swaps: int


def build_cot_example(row: dict[str, object]) -> CoTExample:
    """Build a causal teacher-forcing sequence and color-only LM targets.

    A target is attached to each ``[SLOT_name]`` prompt position.  Since the
    causal model cannot attend to the following gold color token, this trains
    the same conditional prediction used by autoregressive evaluation.
    """

    init = [int(value) for value in row["init"]]  # type: ignore[index]
    swaps = [[int(value) for value in pair] for pair in row["swaps"]]  # type: ignore[index]
    traces = replay_states(init, swaps)
    tokens = [BOS_ID, *encode_initial(init)]
    labels = [-100] * len(tokens)

    for pair, state in zip(swaps, traces, strict=True):
        event = encode_swap(pair)
        tokens.extend(event)
        labels.extend([-100] * len(event))
        tokens.append(STATE_ID)
        labels.append(-100)
        for slot_id, color in zip(SLOT_TOKEN_IDS, state, strict=True):
            tokens.append(slot_id)
            labels.append(COLOR_IDS[color])
            tokens.append(COLOR_IDS[color])
            labels.append(-100)
        tokens.append(END_STATE_ID)
        labels.append(-100)

    final_labels = [int(value) for value in row["labels"]]  # type: ignore[index]
    if traces and traces[-1] != final_labels:
        raise ValueError("symbolic replay does not match the stored final labels")
    return CoTExample(tokens, labels, final_labels, len(swaps))


class ExplicitCoTDataset(Dataset[dict[str, object]]):
    """JSONL ball-swap rows viewed as explicit state-trace examples."""

    def __init__(self, path: str) -> None:
        base = BallSwapDataset(path)
        self.meta = base.meta
        self.rows = base.rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


def collate_cot(batch: Sequence[dict[str, object]]) -> dict[str, Tensor]:
    examples = [build_cot_example(row) for row in batch]
    max_length = max(len(example.input_ids) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    lm_labels: list[list[int]] = []
    for example in examples:
        n_pad = max_length - len(example.input_ids)
        input_ids.append(example.input_ids + [PAD_ID] * n_pad)
        attention_mask.append([1] * len(example.input_ids) + [0] * n_pad)
        lm_labels.append(example.lm_labels + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
        "labels": torch.tensor([example.final_labels for example in examples], dtype=torch.long),
        "n_swaps": torch.tensor([example.n_swaps for example in examples], dtype=torch.long),
    }


def group_rows_by_swap_count(rows: Iterable[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    """Group rows so explicit-CoT generation can batch equal-length prefixes."""

    groups: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(int(row["n_swaps"]), []).append(row)
    return groups
