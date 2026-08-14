from __future__ import annotations

import copy
import json
from pathlib import Path

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
from src.original.experiment import parse_args, run, train_epoch
from src.original.model import (
    DirectTransformer,
    ExplicitCoTTransformer,
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


def test_slot_first_collation_uses_fixed_register_positions() -> None:
    short = sample_row()
    long = {**sample_row(), "swaps": sample_row()["swaps"] + [[0, 2], [2, 4]]}
    batch = collate_fn([short, long], slot_first=True)
    assert batch["slot_pos"].tolist() == [[0, 1, 2, 3, 4]] * 2
    assert batch["input_ids"][:, :5].tolist() == [batch["input_ids"][0, :5].tolist()] * 2
    assert batch["attn_mask"][:, :5].tolist() == [[1, 1, 1, 1, 1]] * 2
    assert batch["attn_mask"][0, -1].item() == 0
    assert batch["attn_mask"][1, -1].item() == 1


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


def test_slot_first_cli_accepts_snake_case_boolean_value() -> None:
    arguments = parse_args(["--architecture", "direct", "--slot_first", "True"])
    assert arguments.slot_first is True


def test_slot_first_yaml_config_maps_to_cli() -> None:
    arguments = config_to_argv({"architecture": "direct", "slot_first": True})
    assert "--slot-first" in arguments


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


def test_original_run_writes_unique_directory_with_config(tmp_path: Path) -> None:
    argv = [
        "--architecture", "direct",
        "--run-name", "repeat",
        "--smoke",
        "--device", "cpu",
        "--output-dir", str(tmp_path),
        "--no-progress",
    ]
    first = run(parse_args(argv))
    second = run(parse_args(argv))

    first_dir = Path(str(first["run_dir"]))
    second_dir = Path(str(second["run_dir"]))
    assert first_dir != second_dir
    assert first_dir.parent == tmp_path
    assert second_dir.parent == tmp_path
    assert first_dir.name.startswith("repeat__")
    assert second_dir.name.startswith("repeat__")

    for run_dir in (first_dir, second_dir):
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "args.json").is_file()
        assert (run_dir / "result.json").is_file()
        assert (run_dir / "checkpoint.pt").is_file()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert config["architecture"] == "direct"
        assert result["run_dir"] == str(run_dir)
        assert result["paths"]["config"] == str(run_dir / "config.json")
