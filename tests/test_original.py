from __future__ import annotations

import copy
from argparse import Namespace
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.original.analysis import flatten_result, summarize_rows
from src.original.data import (
    COLOR_IDS,
    RowsDataset,
    build_cot_example,
    collate_cot,
    inject_noop_swaps,
    replay_states,
)
from src.original.experiment import (
    _prepare_run_outputs,
    InferenceComputeMeter,
    evaluate_classifier,
    load_training_checkpoint,
    rewrite_metrics_log,
    save_training_checkpoint,
    split_train_validation,
    train_epoch,
    validate_epoch,
)
from src.original.model import (
    DirectTransformer,
    ExplicitCoTTransformer,
    OriginalModelConfig,
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


def test_deep_supervision_training_ablation_runs() -> None:
    rows = [sample_row(), sample_row()]
    loader = DataLoader(RowsDataset(rows), batch_size=2, collate_fn=collate_fn)
    model = RecurrentTransformer(tiny_config("recurrent"))
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


def test_cot_targets_only_colors_after_slot_prompts() -> None:
    example = build_cot_example(sample_row())
    targets = [target for target in example.lm_labels if target != -100]
    assert len(targets) == 3 * 5
    assert set(targets).issubset(set(COLOR_IDS))
    batch = collate_cot([sample_row(), sample_row()])
    assert batch["input_ids"].shape == batch["lm_labels"].shape
    assert int((batch["lm_labels"] != -100).sum()) == 2 * 3 * 5


def test_noop_robustness_view_preserves_gold_state() -> None:
    row = sample_row()
    augmented = inject_noop_swaps([row], ratio=0.5, seed=7)[0]
    assert augmented["labels"] == row["labels"]
    assert augmented["n_swaps"] == 5
    assert replay_states(augmented["init"], augmented["swaps"])[-1] == row["labels"]  # type: ignore[arg-type]
    assert any(left == right for left, right in augmented["swaps"])  # type: ignore[assignment]


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
