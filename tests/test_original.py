from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.collate import collate_fn
from src.original.data import COLOR_IDS, build_cot_example, collate_cot, replay_states
from src.original.model import (
    DirectTransformer,
    ExplicitCoTTransformer,
    OriginalModelConfig,
    RecurrentTransformer,
)


def sample_row() -> dict[str, object]:
    return {
        "init": [0, 1, 2, 3, 4],
        "swaps": [[0, 1], [1, 4], [2, 3]],
        "labels": [1, 4, 3, 2, 0],
        "n_swaps": 3,
        "text": "",
    }


def tiny_config(architecture: str, position_encoding: str = "sinusoidal") -> OriginalModelConfig:
    return OriginalModelConfig(
        architecture=architecture,  # type: ignore[arg-type]
        position_encoding=position_encoding,  # type: ignore[arg-type]
        d_model=16,
        n_heads=4,
        d_ff=32,
        num_layers=2,
        num_loops=3,
        min_loops=2,
        dropout=0.0,
    )


def test_symbolic_trace_matches_all_five_final_labels() -> None:
    row = sample_row()
    trace = replay_states(row["init"], row["swaps"])  # type: ignore[arg-type]
    assert trace == [
        [1, 0, 2, 3, 4],
        [1, 4, 2, 3, 0],
        [1, 4, 3, 2, 0],
    ]
    assert trace[-1] == row["labels"]


@pytest.mark.parametrize("position_encoding", ["sinusoidal", "rope"])
def test_direct_supports_both_position_encodings(position_encoding: str) -> None:
    batch = collate_fn([sample_row(), sample_row()])
    model = DirectTransformer(tiny_config("direct", position_encoding))
    logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    assert logits.shape == (2, 5, 5)
    assert torch.isfinite(logits).all()


class ZeroBlock(nn.Module):
    def forward(self, h, attention_mask, position_ids, *, causal):
        return torch.zeros_like(h)


def test_recurrent_update_is_exactly_embedding_plus_block_output() -> None:
    batch = collate_fn([sample_row()])
    model = RecurrentTransformer(tiny_config("recurrent"))
    model.shared_block = ZeroBlock()
    e, h, positions = model.prepare(batch["input_ids"], batch["attn_mask"])
    updated = model.recurrent_step(e, h, batch["attn_mask"], positions)
    expected = e * batch["attn_mask"].unsqueeze(-1)
    torch.testing.assert_close(updated, expected)


def test_adaptive_recurrence_uses_kl_only_and_halts_stable_outputs() -> None:
    batch = collate_fn([sample_row(), sample_row()])
    config = tiny_config("recurrent")
    model = RecurrentTransformer(config)
    model.shared_block = ZeroBlock()
    _, diagnostics = model.forward_adaptive(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
    )
    assert set(diagnostics) == {"steps_taken", "halted", "symmetric_kl"}
    assert diagnostics["steps_taken"].tolist() == [2, 2]
    assert diagnostics["halted"].tolist() == [True, True]
    torch.testing.assert_close(diagnostics["symmetric_kl"][:, 1], torch.zeros(2))


def test_cot_targets_only_colors_after_slot_prompts() -> None:
    example = build_cot_example(sample_row())
    targets = [target for target in example.lm_labels if target != -100]
    assert len(targets) == 3 * 5
    assert set(targets).issubset(set(COLOR_IDS))
    batch = collate_cot([sample_row(), sample_row()])
    assert batch["input_ids"].shape == batch["lm_labels"].shape
    assert int((batch["lm_labels"] != -100).sum()) == 2 * 3 * 5


def test_cot_evaluation_does_not_read_stored_final_labels() -> None:
    row = sample_row()
    altered = copy.deepcopy(row)
    altered["labels"] = [4, 3, 2, 1, 0]
    model = ExplicitCoTTransformer(tiny_config("cot"))
    first = model.generate_states([row])
    second = model.generate_states([altered])
    assert first.shape == (1, 5)
    assert torch.equal(first, second)


def test_cot_forward_is_finite() -> None:
    batch = collate_cot([sample_row()])
    model = ExplicitCoTTransformer(tiny_config("cot", "rope"))
    logits = model(batch["input_ids"], batch["attention_mask"])
    assert logits.shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("position_encoding", ["sinusoidal", "rope"])
def test_cot_incremental_cache_matches_full_causal_forward(position_encoding: str) -> None:
    example = build_cot_example(sample_row())
    input_ids = torch.tensor([example.input_ids], dtype=torch.long)
    model = ExplicitCoTTransformer(tiny_config("cot", position_encoding)).eval()
    with torch.inference_mode():
        full = model(input_ids, torch.ones_like(input_ids))
        caches = [None] * len(model.layers)
        chunks = []
        offset = 0
        for width in (11, 17, input_ids.shape[1] - 28):
            logits, caches = model._incremental(
                input_ids[:, offset : offset + width], caches, offset
            )
            chunks.append(logits)
            offset += width
        incremental = torch.cat(chunks, dim=1)
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
