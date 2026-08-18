from __future__ import annotations

import copy
from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.data.atomic import ATOMIC_PAD_ID, ATOMIC_SLOT_IDS, ATOMIC_VOCAB_SIZE
from src.data.data import GenConfig, make_balanced_length_sweep
from src.original.analysis import flatten_result, summarize_rows
from src.original.data import (
    COLOR_IDS,
    DeterministicOnlineBatchStream,
    RowsDataset,
    build_cot_example,
    collate_cot,
    inject_noop_swaps,
    replay_states,
)
from src.original.experiment import (
    _prepare_run_outputs,
    InferenceComputeMeter,
    SwapCountBatchSampler,
    build_run_name,
    evaluate_classifier,
    fan_learning_rate_multiplier,
    length_proportional_loop_counts,
    load_training_checkpoint,
    maybe_compile_model,
    noop_event_input_ids,
    parse_args,
    proportional_target_indices,
    rewrite_metrics_log,
    run,
    save_training_checkpoint,
    split_train_validation,
    swap_chunk_target_indices,
    train_epoch,
    validate_epoch,
)
from src.original.model import (
    DirectTransformer,
    EventWiseRecurrentTransformer,
    ExplicitCoTTransformer,
    FanRecurrentTransformer,
    OriginalModelConfig,
    RecurrentR0Transformer,
    RecurrentTransformer,
)
from src.trainer import config_to_argv


def sample_row() -> dict[str, object]:
    return {
        "init": [0, 1, 2, 3, 4],
        "swaps": [[0, 1], [1, 4], [2, 3]],
        "labels": [1, 4, 3, 2, 0],
        "n_swaps": 3,
        "text": "",
    }


def trajectory_row() -> dict[str, object]:
    row = sample_row()
    row["intermediate_states"] = replay_states(row["init"], row["swaps"])  # type: ignore[arg-type]
    return row


def test_boundary_sweep_is_balanced_deterministic_and_unique() -> None:
    cfg = GenConfig(min_swaps=11, max_swaps=13, seed=7, profile="boundary_sweep_v1")
    first = make_balanced_length_sweep(4, cfg)
    second = make_balanced_length_sweep(4, cfg)
    assert first == second
    assert {length: sum(row["n_swaps"] == length for row in first) for length in range(11, 14)} == {
        11: 4,
        12: 4,
        13: 4,
    }
    keys = {(tuple(row["init"]), tuple(map(tuple, row["swaps"]))) for row in first}
    assert len(keys) == len(first)


def test_online_curriculum_is_reproducible_and_final_label_only() -> None:
    settings = dict(
        num_steps=3,
        batch_size=4,
        seed=17,
        min_swaps=2,
        max_swaps=4,
        steps_per_length=1,
    )
    first = DeterministicOnlineBatchStream(**settings)
    second = DeterministicOnlineBatchStream(**settings)
    assert [first.curriculum_max_swaps(step) for step in range(3)] == [2, 3, 4]
    for step, (left, right) in enumerate(zip(first, second, strict=True)):
        assert "trajectory_labels" not in left
        assert set(left) == set(right)
        for key in left:
            torch.testing.assert_close(left[key], right[key])
        assert bool((left["n_swaps"] == left["n_swaps"][0]).all())
        assert 2 <= int(left["n_swaps"][0]) <= first.curriculum_max_swaps(step)

    different = next(iter(DeterministicOnlineBatchStream(**{**settings, "seed": 18})))
    original = next(iter(DeterministicOnlineBatchStream(**settings)))
    assert not torch.equal(different["input_ids"], original["input_ids"])


def test_atomic_online_curriculum_is_reproducible_and_uses_atomic_slots() -> None:
    settings = dict(
        num_steps=2,
        batch_size=3,
        seed=17,
        min_swaps=2,
        max_swaps=2,
        steps_per_length=1,
        input_format="atomic",
    )
    first = next(iter(DeterministicOnlineBatchStream(**settings)))
    second = next(iter(DeterministicOnlineBatchStream(**settings)))
    torch.testing.assert_close(first["input_ids"], second["input_ids"])
    assert int(first["input_ids"].max()) < ATOMIC_VOCAB_SIZE
    assert first["input_ids"][0, -5:].tolist() == list(ATOMIC_SLOT_IDS)
    assert int(first["input_ids"].min()) >= ATOMIC_PAD_ID


def test_fan_lr_is_constant_through_curriculum_then_cosine_decays() -> None:
    assert fan_learning_rate_multiplier(8, train_steps=100, curriculum_steps=8) == 1.0
    assert fan_learning_rate_multiplier(54, train_steps=100, curriculum_steps=8) == pytest.approx(0.5)
    assert fan_learning_rate_multiplier(100, train_steps=100, curriculum_steps=8) == pytest.approx(0.0)


def test_cuda_compile_preserves_model_type_and_is_disabled_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DirectTransformer(tiny_config("direct"))
    original_forward = model.forward
    calls: list[object] = []

    def fake_compile(function):
        calls.append(function)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert maybe_compile_model(model, torch.device("cpu")) is model
    assert calls == []
    assert maybe_compile_model(model, torch.device("cuda")) is model
    assert isinstance(model, DirectTransformer)
    assert calls == [original_forward]


def tiny_config(architecture: str, position_encoding: str = "sinusoidal") -> OriginalModelConfig:
    if architecture == "fan-recurrent" and position_encoding == "sinusoidal":
        position_encoding = "none"
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


def test_slot_first_collation_uses_fixed_register_positions() -> None:
    short = sample_row()
    long = {
        **sample_row(),
        "swaps": sample_row()["swaps"] + [[0, 2], [2, 4]],
        "n_swaps": 5,
    }
    batch = collate_fn([short, long], slot_first=True)
    assert batch["slot_pos"].tolist() == [[0, 1, 2, 3, 4]] * 2
    assert batch["input_ids"][:, :5].tolist() == [batch["input_ids"][0, :5].tolist()] * 2
    assert batch["attn_mask"][:, :5].tolist() == [[1, 1, 1, 1, 1]] * 2
    assert batch["attn_mask"][0, -1].item() == 0
    assert batch["attn_mask"][1, -1].item() == 1


def test_event_collation_exposes_local_events_and_fixed_register_inputs() -> None:
    short = sample_row()
    short["swaps"] = short["swaps"][:1]
    short["n_swaps"] = 1
    short["labels"] = replay_states(short["init"], short["swaps"])[-1]  # type: ignore[arg-type]
    batch = collate_fn([short, sample_row()])
    assert batch["initial_colors"].shape == (2, 5)
    assert batch["register_mask"].tolist() == [[1] * 5, [1] * 5]
    assert batch["event_input_ids"].shape == (2, 3, 7)
    assert batch["event_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]


def test_event_recurrence_cannot_see_future_events_and_preserves_padded_state() -> None:
    first = sample_row()
    second = copy.deepcopy(first)
    second["swaps"][1] = [0, 3]  # type: ignore[index]
    second["labels"] = replay_states(second["init"], second["swaps"])[-1]  # type: ignore[arg-type]
    batch = collate_fn([first, second])
    model = EventWiseRecurrentTransformer(tiny_config("event-recurrent"))
    outputs = model.forward_all_events(
        batch["initial_colors"],
        batch["register_mask"],
        batch["event_input_ids"],
        batch["event_mask"],
    )
    assert isinstance(outputs, torch.Tensor)
    torch.testing.assert_close(outputs[0, 0], outputs[1, 0])

    short = copy.deepcopy(first)
    short["swaps"] = short["swaps"][:1]  # type: ignore[index]
    short["n_swaps"] = 1
    short["labels"] = replay_states(short["init"], short["swaps"])[-1]  # type: ignore[arg-type]
    padded = collate_fn([short, first])
    padded_outputs = model.forward_all_events(
        padded["initial_colors"],
        padded["register_mask"],
        padded["event_input_ids"],
        padded["event_mask"],
    )
    assert isinstance(padded_outputs, torch.Tensor)
    torch.testing.assert_close(padded_outputs[0, 0], padded_outputs[0, -1])


def test_event_recurrence_final_gradient_reaches_first_event_state() -> None:
    batch = collate_fn([sample_row()])
    model = EventWiseRecurrentTransformer(tiny_config("event-recurrent"))
    outputs, states = model.forward_all_events(
        batch["initial_colors"],
        batch["register_mask"],
        batch["event_input_ids"],
        batch["event_mask"],
        return_hidden_states=True,
    )
    gradient = torch.autograd.grad(outputs[:, -1].sum(), states[0])[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_event_initial_readout_reuses_classifier_and_updates_register_encoder() -> None:
    batch = collate_fn([sample_row()])
    model = EventWiseRecurrentTransformer(tiny_config("event-recurrent"))
    logits = model.classify_initial_state(
        batch["initial_colors"],
        batch["register_mask"],
    )
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["initial_colors"].reshape(-1),
    )
    loss.backward()
    assert model.register_identity.weight.grad is not None
    assert model.classifier.classifier[-1].weight.grad is not None
    assert model.shared_update.attention.qkv.weight.grad is None


def test_noop_probe_encoder_builds_active_self_swaps() -> None:
    encoded = noop_event_input_ids(torch.tensor([0, 4], dtype=torch.long))
    assert encoded.shape == (2, 7)
    assert torch.equal(encoded[:, 0], encoded[:, 2])


def test_trajectory_collation_pads_ragged_swap_traces_and_slots() -> None:
    short = trajectory_row()
    long = trajectory_row()
    long["swaps"] = [[0, 1], [1, 4], [2, 3], [0, 2], [2, 4]]
    long["intermediate_states"] = replay_states(long["init"], long["swaps"])  # type: ignore[arg-type]
    long["labels"] = long["intermediate_states"][-1]  # type: ignore[index]
    long["n_swaps"] = 5
    batch = collate_fn([short, long])
    assert batch["trajectory_labels"].shape == (2, 5, 5)
    assert batch["trajectory_labels"][0, :3].tolist() == short["intermediate_states"]
    assert batch["trajectory_labels"][0, 3:].tolist() == [[-100] * 5, [-100] * 5]


@pytest.mark.parametrize(
    ("loops", "swaps", "expected"),
    [
        (24, [4, 8], [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                     [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]]),
        (6, [10], [[1, 3, 4, 6, 8, 9]]),
        (4, [4], [[0, 1, 2, 3]]),
    ],
)
def test_proportional_trajectory_mapping_handles_loop_length_mismatches(
    loops: int, swaps: list[int], expected: list[list[int]]
) -> None:
    assert proportional_target_indices(loops, torch.tensor(swaps)).tolist() == expected


def test_swap_chunk_targets_and_batch_sampler_share_length_proportional_budget() -> None:
    n_swaps = torch.tensor([3, 4, 5, 6])
    assert length_proportional_loop_counts(n_swaps, 2).tolist() == [2, 2, 3, 3]
    assert length_proportional_loop_counts(torch.tensor([2, 3, 10]), 0.5).tolist() == [4, 6, 20]
    assert swap_chunk_target_indices(4, torch.tensor([4, 7]), 2).tolist() == [
        [1, 3, 3, 3],
        [1, 3, 5, 6],
    ]
    assert swap_chunk_target_indices(8, torch.tensor([4]), 0.5).tolist() == [
        [0, 0, 1, 1, 2, 2, 3, 3],
    ]
    rows = []
    for count in n_swaps.tolist():
        row = trajectory_row()
        swaps = [[0, 1]] * count
        row["swaps"] = swaps
        row["n_swaps"] = count
        row["intermediate_states"] = replay_states(row["init"], swaps)  # type: ignore[arg-type]
        row["labels"] = row["intermediate_states"][-1]  # type: ignore[index]
        rows.append(row)
    sampler = SwapCountBatchSampler(
        RowsDataset(rows), batch_size=2, swaps_per_loop=2, shuffle=False, seed=0
    )
    for indices in sampler:
        loop_counts = {(int(rows[index]["n_swaps"]) + 1) // 2 for index in indices}
        assert len(loop_counts) == 1


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


class IdentityRecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.causal_values: list[bool] = []

    def forward(self, h, attention_mask, position_ids, *, causal):
        self.causal_values.append(causal)
        return h * attention_mask.unsqueeze(-1)


def test_recurrent_update_is_exactly_embedding_plus_block_output() -> None:
    batch = collate_fn([sample_row()])
    model = RecurrentTransformer(tiny_config("recurrent"))
    model.shared_block = ZeroBlock()
    e, h, positions = model.prepare(batch["input_ids"], batch["attn_mask"])
    updated = model.recurrent_step(e, h, batch["attn_mask"], positions)
    expected = e * batch["attn_mask"].unsqueeze(-1)
    torch.testing.assert_close(updated, expected)


def test_fan_recurrence_starts_at_zero_and_reinjects_nope_embedding() -> None:
    batch = collate_fn([sample_row()])
    model = FanRecurrentTransformer(tiny_config("fan-recurrent"))
    blocks = nn.ModuleList([IdentityRecordingBlock(), IdentityRecordingBlock()])
    model.shared_layers = blocks
    _, embedding, _ = model.prepare(batch["input_ids"], batch["attn_mask"])
    _, hidden_states = model.forward_all_loops(
        batch["input_ids"],
        batch["attn_mask"],
        batch["slot_pos"],
        return_hidden_states=True,
    )
    mask = batch["attn_mask"].unsqueeze(-1)
    for loop_index, hidden in enumerate(hidden_states, start=1):
        torch.testing.assert_close(hidden, loop_index * embedding * mask)
    assert all(value is True for block in blocks for value in block.causal_values)


def test_fan_model_rejects_position_encodings() -> None:
    with pytest.raises(ValueError, match="requires position_encoding='none'"):
        OriginalModelConfig(architecture="fan-recurrent", position_encoding="sinusoidal")


def test_fan_controls_require_explicit_compatible_configurations() -> None:
    positional = OriginalModelConfig(
        architecture="fan-recurrent",
        position_encoding="sinusoidal",
        fan_positional_control=True,
    )
    atomic = OriginalModelConfig(
        architecture="fan-recurrent",
        position_encoding="none",
        fan_input_format="atomic",
    )
    atomic_positional = OriginalModelConfig(
        architecture="fan-recurrent",
        position_encoding="sinusoidal",
        fan_input_format="atomic",
        fan_positional_control=True,
    )
    assert positional.fan_positional_control is True
    assert atomic.fan_input_format == "atomic"
    assert atomic_positional.fan_positional_control is True
    with pytest.raises(ValueError, match="requires position_encoding='none'"):
        OriginalModelConfig(
            architecture="fan-recurrent",
            position_encoding="sinusoidal",
            fan_input_format="atomic",
        )


def test_atomic_fan_model_reads_atomic_collation() -> None:
    batch = collate_fn([sample_row()], input_format="atomic")
    model = FanRecurrentTransformer(
        OriginalModelConfig(
            architecture="fan-recurrent",
            position_encoding="none",
            fan_input_format="atomic",
            d_model=16,
            n_heads=4,
            d_ff=32,
            num_layers=1,
            num_loops=2,
        )
    )
    logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    assert logits.shape == (1, 5, 5)


def test_atomic_causal_direct_model_reads_atomic_collation() -> None:
    batch = collate_fn([sample_row()], input_format="atomic")
    model = DirectTransformer(
        OriginalModelConfig(
            architecture="direct",
            position_encoding="none",
            direct_input_format="atomic",
            direct_causal=True,
            d_model=16,
            n_heads=4,
            d_ff=32,
            num_layers=2,
        )
    )
    blocks = nn.ModuleList([IdentityRecordingBlock(), IdentityRecordingBlock()])
    model.layers = blocks
    logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    assert logits.shape == (1, 5, 5)
    assert all(value is True for block in blocks for value in block.causal_values)


def test_online_direct_baseline_requires_atomic_causal_nope() -> None:
    with pytest.raises(ValueError, match="atomic input, causal attention, and NoPE"):
        run(
            parse_args(
                [
                    "--architecture",
                    "direct",
                    "--online-training",
                    "--smoke",
                    "--no-progress",
                ]
            )
        )


def test_recurrent_r0_update_does_not_reinject_embedding() -> None:
    batch = collate_fn([sample_row()])
    model = RecurrentR0Transformer(tiny_config("recurrent-r0"))
    model.shared_block = ZeroBlock()
    _, h, positions = model.prepare(batch["input_ids"], batch["attn_mask"])
    updated = model.recurrent_step(
        h, batch["attn_mask"], positions, loop_index=0
    )
    torch.testing.assert_close(updated, torch.zeros_like(h))


def test_recurrent_r0_loop_conditioning_is_explicit_ablation() -> None:
    batch = collate_fn([sample_row()])
    model = RecurrentR0Transformer(tiny_config("recurrent-r0"))
    conditioned = RecurrentR0Transformer(
        OriginalModelConfig(
            **{
                **tiny_config("recurrent-r0").to_dict(),
                "loop_conditioning": "learned",
                "max_loop_embeddings": 4,
            }
        )
    )
    plain_logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    conditioned_logits = conditioned(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
    assert plain_logits.shape == conditioned_logits.shape == (1, 5, 5)


def test_recurrent_r0_adaptive_halting_reports_confidence_and_update_ratio() -> None:
    batch = collate_fn([sample_row(), sample_row()])
    model = RecurrentR0Transformer(tiny_config("recurrent-r0"))
    logits, diagnostics = model.forward_adaptive(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"], max_loops=3
    )
    assert logits.shape == (2, 5, 5)
    assert set(diagnostics) == {
        "steps_taken",
        "halted",
        "symmetric_kl",
        "update_ratio",
        "confidence",
    }
    assert diagnostics["update_ratio"].shape == (2, 3)
    assert diagnostics["confidence"].shape == (2, 3)


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


def test_recurrent_exposes_every_loop_for_optional_deep_supervision() -> None:
    batch = collate_fn([sample_row(), sample_row()])
    model = RecurrentTransformer(tiny_config("recurrent"))
    logits = model.forward_all_loops(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
    )
    assert logits.shape == (2, 3, 5, 5)
    torch.testing.assert_close(model(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
    ), logits[:, -1])


@pytest.mark.parametrize("architecture", ["recurrent", "recurrent-r0", "fan-recurrent"])
def test_final_readout_gradient_reaches_first_recurrent_hidden_state(architecture: str) -> None:
    batch = collate_fn([trajectory_row()])
    model = {
        "recurrent": RecurrentTransformer,
        "recurrent-r0": RecurrentR0Transformer,
        "fan-recurrent": FanRecurrentTransformer,
    }[architecture](tiny_config(architecture))
    loop_logits, hidden_states = model.forward_all_loops(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"], return_hidden_states=True
    )
    first_loop_gradient = torch.autograd.grad(loop_logits[:, -1].sum(), hidden_states[0])[0]
    assert torch.isfinite(first_loop_gradient).all()
    assert first_loop_gradient.abs().sum().item() > 0.0


@pytest.mark.parametrize("architecture", ["recurrent", "fan-recurrent"])
def test_deep_supervision_training_ablation_runs(architecture: str) -> None:
    rows = [sample_row(), sample_row()]
    loader = DataLoader(RowsDataset(rows), batch_size=2, collate_fn=collate_fn)
    model = {
        "recurrent": RecurrentTransformer,
        "fan-recurrent": FanRecurrentTransformer,
    }[architecture](tiny_config(architecture))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        deep_supervision_weight=0.5,
    )
    assert loss > 0.0


def test_event_recurrent_training_uses_final_ce_only() -> None:
    rows = [sample_row(), sample_row()]
    loader = DataLoader(RowsDataset(rows), batch_size=2, collate_fn=collate_fn)
    model = EventWiseRecurrentTransformer(tiny_config("event-recurrent"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
    )
    assert loss > 0.0


def test_cot_targets_only_colors_after_slot_prompts() -> None:
    example = build_cot_example(sample_row())
    targets = [target for target in example.lm_labels if target != -100]
    assert len(targets) == 3 * 5
    assert set(targets).issubset(set(COLOR_IDS))
    batch = collate_cot([sample_row(), sample_row()])
    assert batch["input_ids"].shape == batch["lm_labels"].shape
    assert int((batch["lm_labels"] != -100).sum()) == 2 * 3 * 5


def test_noop_robustness_view_preserves_gold_state() -> None:
    row = trajectory_row()
    augmented = inject_noop_swaps([row], ratio=0.5, seed=7)[0]
    assert augmented["labels"] == row["labels"]
    assert augmented["n_swaps"] == 5
    assert replay_states(augmented["init"], augmented["swaps"])[-1] == row["labels"]  # type: ignore[arg-type]
    assert any(left == right for left, right in augmented["swaps"])  # type: ignore[assignment]
    assert augmented["intermediate_states"] == replay_states(augmented["init"], augmented["swaps"])  # type: ignore[arg-type]


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


def test_original_result_aggregation_keeps_seed_and_swap_axes() -> None:
    results = []
    for seed, score in enumerate((0.2, 0.4, 0.6)):
        results.append({
            "config": {"architecture": "direct", "position_encoding": "sinusoidal"},
            "seed": seed,
            "splits": {
                "id_test": {
                    "exact_match": score,
                    "slot_accuracy": score + 0.1,
                    "n_samples": 10,
                    "by_swaps": {
                        "2": {"exact_match": score, "correct": int(score * 10), "total": 10}
                    },
                }
            },
        })
    raw = [row for result in results for row in flatten_result(result)]
    summary = summarize_rows(raw)
    swap_summary = next(row for row in summary if row["n_swaps"] == 2)
    assert swap_summary["n_seeds"] == 3
    assert swap_summary["mean"] == pytest.approx(0.4)
    assert swap_summary["std"] == pytest.approx(0.2)


def test_team_yaml_config_maps_to_canonical_trainer_cli() -> None:
    arguments = config_to_argv({
        "architecture": "looped",
        "position_encoding": "rope",
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "num_loops": 2,
        "training": {"epochs": 1, "batch_size": 8, "seed": 3},
        "adaptive_halting": {
            "enabled_at_evaluation": True,
            "threshold": 0.01,
            "min_loops": 2,
            "patience": 1,
        },
    })
    assert arguments[:2] == ["--architecture", "recurrent"]
    assert "--position-encoding" in arguments
    assert "rope" in arguments
    assert "--adaptive-kl-eval" in arguments


def test_validation_split_is_deterministic_and_disjoint() -> None:
    rows = []
    for n_swaps in (2, 3, 4, 5):
        rows.extend(dict(sample_row(), row_id=f"{n_swaps}-{index}", n_swaps=n_swaps)
                    for index in range(10))
    dataset = RowsDataset(rows)
    train_a, validation_a = split_train_validation(
        dataset, validation_ratio=0.2, seed=11, max_samples=None
    )
    train_b, validation_b = split_train_validation(
        dataset, validation_ratio=0.2, seed=11, max_samples=None
    )
    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert len(validation_a) == 8
    assert set(train_a.indices).isdisjoint(validation_a.indices)
    validation_lengths = [dataset[index]["n_swaps"] for index in validation_a.indices]
    assert {length: validation_lengths.count(length) for length in set(validation_lengths)} == {
        2: 2, 3: 2, 4: 2, 5: 2,
    }


def test_validation_and_full_training_checkpoint_roundtrip(tmp_path) -> None:
    rows = [sample_row(), sample_row()]
    loader = DataLoader(RowsDataset(rows), batch_size=2, collate_fn=collate_fn)
    model = DirectTransformer(tiny_config("direct"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    metrics = validate_epoch(model, loader, torch.device("cpu"))
    assert metrics["target_count"] == 10
    checkpoint = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        history=[{"epoch": 1, "validation": metrics}],
        best_validation_loss=float(metrics["loss"]),
        best_epoch=1,
        training_seconds=0.5,
        run_config={"seed": 0},
    )
    loaded = load_training_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded["epoch"] == 1
    assert loaded["optimizer_state"] is not None
    assert loaded["scheduler_state"] is not None
    assert loaded["best_epoch"] == 1
    assert all(not key.startswith("_orig_mod.") for key in loaded["model_state"])


def test_yaml_wires_training_infrastructure_options() -> None:
    arguments = config_to_argv({
        "architecture": "direct",
        "training": {
            "optimizer": "AdamW",
            "scheduler": {"name": "cosine", "warmup_epochs": 2, "min_lr": 1e-6},
        },
        "validation": {"ratio": 0.15},
        "checkpointing": {"save_every": 3, "resume": "auto", "overwrite": True},
        "performance": {
            "amp": True,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "evaluation": {"batch_size": 64, "splits": ["id_test"]},
    })
    for flag in (
        "--optimizer", "--scheduler", "--validation-ratio", "--checkpoint-every",
        "--resume", "--amp", "--num-workers", "--pin-memory",
        "--persistent-workers", "--eval-batch-size", "--eval-splits", "--overwrite",
    ):
        assert flag in arguments


def test_resume_log_reconciliation_removes_duplicates(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 99}\n{"epoch": 99}\n', encoding="utf-8")
    history = [{"epoch": 1, "train_loss": 1.0}, {"epoch": 2, "train_loss": 0.5}]
    rewrite_metrics_log(path, history)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == history


def test_existing_run_requires_resume_or_explicit_overwrite(tmp_path) -> None:
    args = Namespace(output_dir=tmp_path, resume=None, overwrite=False)
    run_dir, _checkpoints, _metrics, _result = _prepare_run_outputs(args, "test-run")
    run_dir.mkdir(parents=True)
    marker = run_dir / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_run_outputs(args, "test-run")
    args.overwrite = True
    _prepare_run_outputs(args, "test-run")
    assert not run_dir.exists()


def test_runtime_inference_meter_counts_classifier_and_cot_generation_flops() -> None:
    row = sample_row()
    batch = collate_fn([row, row])
    direct = DirectTransformer(tiny_config("direct")).eval()
    with InferenceComputeMeter(direct, torch.device("cpu")) as meter:
        meter.measure(lambda: direct(batch["input_ids"], batch["attn_mask"], batch["slot_pos"]))
    direct_summary = meter.summary(n_samples=2)
    assert direct_summary["flops"] > 0
    assert direct_summary["forward_calls"] == 1

    cot = ExplicitCoTTransformer(tiny_config("cot")).eval()
    with InferenceComputeMeter(cot, torch.device("cpu")) as meter:
        meter.measure(lambda: cot.generate_states([row]))
    cot_summary = meter.summary(n_samples=1)
    assert cot_summary["flops"] > 0
    assert meter.attention_flops > 0


def test_evaluation_persists_forward_compute_and_time_metrics() -> None:
    rows = [sample_row(), sample_row()]
    loader = DataLoader(RowsDataset(rows), batch_size=2, collate_fn=collate_fn)
    result = evaluate_classifier(
        DirectTransformer(tiny_config("direct")),
        loader,
        torch.device("cpu"),
        adaptive_kl=False,
    )
    compute = result["inference_compute"]
    assert isinstance(compute, dict)
    assert compute["flops"] > 0
    assert compute["inference_seconds"] >= 0.0


def test_slot_first_cli_accepts_snake_case_boolean_value() -> None:
    arguments = parse_args(["--architecture", "direct", "--slot_first", "True"])
    assert arguments.slot_first is True


def test_slot_first_run_name_records_the_ablation() -> None:
    arguments = parse_args(["--architecture", "recurrent-r0", "--slot-first"])
    assert "slotfirst" in build_run_name(arguments)


def test_slot_first_yaml_config_maps_to_cli() -> None:
    arguments = config_to_argv({"architecture": "direct", "slot_first": True})
    assert "--slot-first" in arguments


def test_event_recurrent_yaml_alias_maps_to_cli() -> None:
    arguments = config_to_argv({"architecture": "event_recurrent"})
    assert arguments[:2] == ["--architecture", "event-recurrent"]


def test_event_probe_yaml_config_maps_to_cli() -> None:
    arguments = config_to_argv({
        "architecture": "event-recurrent",
        "evaluation": {"event_trajectory_probe": True},
    })
    assert "--event-intermediate-weight" not in arguments
    assert "--event-initial-state-weight" not in arguments
    assert "--event-noop-consistency-weight" not in arguments
    assert "--event-trajectory-probe" in arguments


def test_trajectory_probe_yaml_config_maps_to_cli() -> None:
    arguments = config_to_argv({
        "architecture": "recurrent",
        "evaluation": {"trajectory_probe_eval": True},
    })
    assert "--trajectory-supervision" not in arguments
    assert "--trajectory-supervision-weight" not in arguments
    assert "--trajectory-probe-eval" in arguments


def test_removed_auxiliary_loss_cli_flags_are_rejected() -> None:
    for removed_flag in (
        "--trajectory-supervision",
        "--trajectory-supervision-weight",
        "--event-intermediate-weight",
        "--event-initial-state-weight",
        "--event-noop-consistency-weight",
    ):
        with pytest.raises(SystemExit):
            parse_args(["--architecture", "recurrent", removed_flag, "0.1"])


def test_recurrent_cli_rejects_invalid_combinations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require a recurrent model"):
        run(parse_args([
            "--architecture", "direct", "--trajectory-probe-eval", "--output-dir", str(tmp_path),
        ]))
    with pytest.raises(ValueError, match="requires --adaptive-kl-eval"):
        run(parse_args([
            "--architecture", "recurrent", "--adaptive-max-loops", "24", "--output-dir", str(tmp_path),
        ]))
    with pytest.raises(ValueError, match="requires --swaps-per-loop"):
        run(parse_args([
            "--architecture", "recurrent", "--length-matched-eval", "--output-dir", str(tmp_path),
        ]))
    with pytest.raises(ValueError, match="does not apply"):
        run(parse_args([
            "--architecture", "event-recurrent", "--slot-first", "--output-dir", str(tmp_path),
        ]))
    with pytest.raises(ValueError, match="requires --architecture event-recurrent"):
        run(parse_args([
            "--architecture", "recurrent", "--event-trajectory-probe",
            "--output-dir", str(tmp_path),
        ]))


def test_event_trajectory_probe_runs_in_smoke_experiment(tmp_path: Path) -> None:
    result = run(parse_args([
        "--architecture", "event-recurrent",
        "--smoke",
        "--device", "cpu",
        "--no-progress",
        "--event-trajectory-probe",
        "--output-dir", str(tmp_path),
    ]))
    probe = result["splits"]["id_test"]["event_trajectory_probe"]
    assert probe["evaluation_mode"] == "event_aligned_trajectory"
    assert probe["initial_state"]["n_samples"] == 4
    assert probe["update_norms"]["real_relative_l2"] > 0.0
    assert probe["update_norms"]["noop_relative_l2"] > 0.0
    assert probe["events"]


def test_trajectory_probe_records_each_requested_loop_count(tmp_path: Path) -> None:
    result = run(parse_args([
        "--architecture", "recurrent",
        "--smoke",
        "--device", "cpu",
        "--no-progress",
        "--trajectory-probe-eval",
        "--swaps-per-loop", "2",
        "--length-matched-eval",
        "--eval-loop-counts", "1", "2",
        "--output-dir", str(tmp_path),
    ]))
    probe = result["splits"]["id_test"]["trajectory_probe"]
    assert set(probe) == {"1", "2"}
    assert len(probe["1"]["loops"]) == 1
    assert len(probe["2"]["loops"]) == 2
    oracle = result["splits"]["id_test"]["length_matched_oracle"]
    assert oracle["evaluation_mode"] == "length_matched_oracle"
    assert oracle["swaps_per_loop"] == 2
    assert sum(oracle["loop_count_histogram"].values()) == oracle["n_samples"]


def test_fan_yaml_maps_online_curriculum_to_cli() -> None:
    arguments = config_to_argv({
        "architecture": "fan_recurrent",
        "position_encoding": "none",
        "training": {
            "online_training": True,
            "train_steps": 100,
            "swaps_per_loop": 1.0,
            "curriculum_min_swaps": 2,
            "curriculum_max_swaps": 10,
            "curriculum_steps_per_length": 5,
        },
    })
    assert arguments[:2] == ["--architecture", "fan-recurrent"]
    assert "--online-training" in arguments
    assert arguments[arguments.index("--train-steps") + 1] == "100"
    assert arguments[arguments.index("--curriculum-max-swaps") + 1] == "10"


def test_fan_smoke_uses_length_matched_final_only_training(tmp_path: Path) -> None:
    result = run(parse_args([
        "--architecture", "fan-recurrent",
        "--position-encoding", "none",
        "--online-training",
        "--swaps-per-loop", "1",
        "--smoke",
        "--device", "cpu",
        "--no-progress",
        "--output-dir", str(tmp_path),
    ]))
    assert result["track"] == "fan_aligned"
    assert result["training_regime"]["objective"] == "final_ce_only"
    assert result["training_regime"]["optimizer_steps"] == 2
    assert result["splits"]["id_test"]["evaluation_mode"] == "length_matched"


def test_r0_yaml_config_maps_advanced_ball_swap_options() -> None:
    arguments = config_to_argv({
        "architecture": "r0",
        "num_loops": 6,
        "loop_conditioning": "learned",
        "residual_scale": 0.5,
        "recurrent_blocks": 2,
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "seed": 3,
            "random_loops": True,
            "random_min_loops": 2,
            "random_max_loops": 6,
        },
        "adaptive_halting": {
            "enabled_at_evaluation": True,
            "threshold": 0.01,
            "update_threshold": 0.05,
            "min_confidence": 0.7,
            "min_loops": 2,
            "patience": 2,
        },
        "evaluation": {"loop_counts": [1, 2, 4, 6, 8]},
    })
    assert arguments[:2] == ["--architecture", "recurrent-r0"]
    assert "--loop-conditioning" in arguments
    assert "learned" in arguments
    assert "--random-loops" in arguments
    assert "--eval-loop-counts" in arguments
    assert "--adaptive-kl-eval" in arguments


def test_original_run_writes_config_and_requires_collision_policy(tmp_path: Path) -> None:
    argv = [
        "--architecture", "direct",
        "--run-name", "repeat",
        "--smoke",
        "--device", "cpu",
        "--output-dir", str(tmp_path),
        "--no-progress",
    ]
    first = run(parse_args(argv))
    with pytest.raises(FileExistsError):
        run(parse_args(argv))

    run_dir = Path(str(first["run_dir"]))
    assert run_dir == tmp_path / "repeat"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "args.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "checkpoint.pt").is_file()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert config["architecture"] == "direct"
    assert result["run_dir"] == str(run_dir)
    assert result["paths"]["config"] == str(run_dir / "config.json")
