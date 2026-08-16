"""Data views for the original five-person ball-swap experiments.

Direct and recurrent models reuse the repository's fixed 23-token input.  The
explicit-CoT model uses three additional control tokens and writes a complete
five-person state after every swap.  Gold trace tokens are used only for
teacher-forced training; evaluation generates them autoregressively.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data.collate import BallSwapDataset, collate_fn, encode_body
from ..data.data import GenConfig, sample_problem, to_row
from ..data.vocab import COLORS, N_ENTITIES, PAD_ID, SLOTS, TOK2ID, VOCAB_SIZE


# Keep src.vocab's ID order untouched so old checkpoints and data remain valid.
BOS_ID = VOCAB_SIZE
STATE_ID = VOCAB_SIZE + 1
END_STATE_ID = VOCAB_SIZE + 2
COT_VOCAB_SIZE = VOCAB_SIZE + 3
COLOR_IDS = tuple(TOK2ID[color] for color in COLORS)
SLOT_TOKEN_IDS = tuple(TOK2ID[slot] for slot in SLOTS)


def _stream_seed(base_seed: int, step: int, sample: int) -> int:
    """Mix online-example coordinates into a stable 64-bit RNG seed."""

    value = (
        (base_seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((step + 1) * 0x9E3779B97F4A7C15)
        ^ ((sample + 1) * 0xBF58476D1CE4E5B9)
    ) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class DeterministicOnlineBatchStream(Iterable[dict[str, Tensor]]):
    """Generate reproducible i.i.d. training batches from step coordinates.

    A step first samples one swap length from the currently available
    curriculum range, then generates a fresh batch at that exact length.  The
    shared length lets every sample use the same recurrent budget.  All RNGs
    are derived from ``(seed, step, sample_index)``; rerunning with the same
    arguments recreates every token and target exactly, independent of global
    Python or PyTorch RNG state.

    Intermediate states are deliberately removed before collation so this
    training stream can support final-state CE (and the retained deep-
    supervision ablation) but cannot accidentally use trajectory labels.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        batch_size: int,
        seed: int,
        min_swaps: int,
        max_swaps: int,
        steps_per_length: int,
        slot_first: bool = False,
        input_format: str = "template",
    ) -> None:
        if num_steps < 1 or batch_size < 1 or steps_per_length < 1:
            raise ValueError("num_steps, batch_size, and steps_per_length must be positive")
        if not 1 <= min_swaps <= max_swaps:
            raise ValueError("curriculum swap range must satisfy 1 <= min <= max")
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.seed = seed
        self.min_swaps = min_swaps
        self.max_swaps = max_swaps
        self.steps_per_length = steps_per_length
        self.slot_first = slot_first
        if input_format not in {"template", "atomic"}:
            raise ValueError("input_format must be 'template' or 'atomic'")
        self.input_format = input_format

    def __len__(self) -> int:
        return self.num_steps

    def curriculum_max_swaps(self, step: int) -> int:
        if not 0 <= step < self.num_steps:
            raise IndexError("online training step is out of range")
        return min(self.max_swaps, self.min_swaps + step // self.steps_per_length)

    def _batch_for_step(self, step: int) -> dict[str, Tensor]:
        current_max = self.curriculum_max_swaps(step)
        length_rng = random.Random(_stream_seed(self.seed, step, 0))
        n_swaps = length_rng.randint(self.min_swaps, current_max)
        config = GenConfig(
            n_entities=N_ENTITIES,
            min_swaps=n_swaps,
            max_swaps=n_swaps,
            seed=self.seed,
            profile="fan_online_curriculum_v1",
        )
        rows: list[dict[str, object]] = []
        for sample_index in range(self.batch_size):
            rng = random.Random(_stream_seed(self.seed, step, sample_index + 1))
            row = to_row(sample_problem(rng, config), config)
            row.pop("intermediate_states", None)
            rows.append(row)
        return collate_fn(rows, slot_first=self.slot_first, input_format=self.input_format)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for step in range(self.num_steps):
            yield self._batch_for_step(step)


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


class RowsDataset(Dataset[dict[str, object]]):
    """A lightweight in-memory dataset used by controlled robustness views."""

    def __init__(self, rows: Sequence[dict[str, object]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


def inject_noop_swaps(
    rows: Sequence[dict[str, object]],
    *,
    ratio: float,
    seed: int,
) -> list[dict[str, object]]:
    """Insert deterministic self-swaps without changing gold final states.

    ``ratio`` is the number of inserted noops relative to each row's original
    swap count.  The original swap order is preserved, and insertions are
    sampled independently per row from a seed-derived RNG.
    """

    if ratio <= 0.0:
        raise ValueError("noop ratio must be positive")
    augmented: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        result = copy.deepcopy(row)
        swaps = [list(pair) for pair in result["swaps"]]  # type: ignore[index]
        n_noops = max(1, round(len(swaps) * ratio))
        rng = random.Random(seed * 1_000_003 + row_index)
        for _ in range(n_noops):
            entity = rng.randrange(N_ENTITIES)
            position = rng.randrange(len(swaps) + 1)
            swaps.insert(position, [entity, entity])
        result["swaps"] = swaps
        result["n_swaps"] = len(swaps)
        if "intermediate_states" in result:
            result["intermediate_states"] = replay_states(result["init"], swaps)  # type: ignore[arg-type]
        augmented.append(result)
    return augmented


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
