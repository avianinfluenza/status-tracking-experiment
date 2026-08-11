from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.collate import BallSwapDataset, collate_fn
from src.classifier import Classifier
from src.model import (
    ModelConfig,
    RecurrentTransformerEncoder,
    StateTrackingModel,
    count_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def small_config(model_type: str, **overrides) -> ModelConfig:
    values = {
        "model_type": model_type,
        "d_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "num_layers": 2,
        "recurrent_steps": 3,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_both_models_follow_output_contract() -> None:
    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[0], dataset[1]])
    for model_type in ("vanilla", "recurrent"):
        model = StateTrackingModel(small_config(model_type))
        hidden = model.encode(batch["input_ids"], batch["attn_mask"])
        logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
        assert hidden.shape == (*batch["input_ids"].shape, 32)
        assert logits.shape == (2, 5, 5)
        loss = F.cross_entropy(logits.reshape(-1, 5), batch["labels"].reshape(-1))
        assert torch.isfinite(loss)


def test_parameter_matched_pair_has_identical_trainable_size() -> None:
    vanilla = StateTrackingModel(small_config("vanilla", num_layers=1, recurrent_steps=4))
    recurrent = StateTrackingModel(small_config("recurrent", num_layers=1, recurrent_steps=4))
    assert count_parameters(vanilla) == count_parameters(recurrent)


def test_feature_classifier_mlp_contract() -> None:
    model = StateTrackingModel(small_config("recurrent", classifier_dim=2))
    assert isinstance(model.classifier, Classifier)
    slot_states = torch.randn(3, 5, 32)
    logits = model.classifier(slot_states)
    assert logits.shape == (3, 5, 5)
    linear_layers = [
        layer for layer in model.classifier.classifier if isinstance(layer, torch.nn.Linear)
    ]
    assert [layer.in_features for layer in linear_layers] == [32, 64]
    assert [layer.out_features for layer in linear_layers] == [64, 5]


def test_residual_scaling_damps_each_recurrent_update() -> None:
    torch.manual_seed(3)
    full = RecurrentTransformerEncoder(
        small_config("recurrent", recurrent_steps=1, residual_scale=1.0)
    ).eval()
    half = RecurrentTransformerEncoder(
        small_config("recurrent", recurrent_steps=1, residual_scale=0.5)
    ).eval()
    half.load_state_dict(full.state_dict())

    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[0], dataset[1]])
    x, padding_mask, valid = full.prepare(batch["input_ids"], batch["attn_mask"])
    full_next, _ = full.recurrent_step(x, padding_mask, valid, step=1)
    half_next, _ = half.recurrent_step(x, padding_mask, valid, step=1)
    torch.testing.assert_close(half_next - x, 0.5 * (full_next - x), atol=1e-6, rtol=1e-5)


def test_loop_conditioning_changes_shared_block_computation() -> None:
    torch.manual_seed(4)
    conditioned = RecurrentTransformerEncoder(
        small_config("recurrent", loop_conditioning="sinusoidal")
    ).eval()
    naive = RecurrentTransformerEncoder(
        small_config("recurrent", loop_conditioning="none")
    ).eval()
    naive.load_state_dict(conditioned.state_dict())

    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[0]])
    conditioned_hidden = conditioned(batch["input_ids"], batch["attn_mask"])
    naive_hidden = naive(batch["input_ids"], batch["attn_mask"])
    assert not torch.allclose(conditioned_hidden, naive_hidden)


def test_adaptive_halting_is_per_sample_and_bounded() -> None:
    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[0], dataset[1], dataset[2]])
    model = StateTrackingModel(
        small_config(
            "recurrent",
            recurrent_steps=4,
            min_recurrent_steps=2,
            halting_patience=1,
        )
    ).eval()
    logits, diagnostics = model.forward_adaptive(
        batch["input_ids"],
        batch["attn_mask"],
        batch["slot_pos"],
        threshold=1e9,
        min_confidence=0.0,
        update_threshold=1e9,
    )
    assert logits.shape == (3, 5, 5)
    assert diagnostics["halted"].all()
    assert diagnostics["steps_taken"].tolist() == [2, 2, 2]
    assert diagnostics["symmetric_kl"].shape == (3, 2)
    assert torch.isfinite(diagnostics["symmetric_kl"][:, 1:]).all()
    assert torch.isfinite(diagnostics["update_ratio"]).all()
    assert torch.isfinite(diagnostics["confidence"]).all()


def test_adaptive_halting_rejects_stable_but_unconfident_predictions() -> None:
    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[0], dataset[1]])
    model = StateTrackingModel(small_config("recurrent", recurrent_steps=4)).eval()
    _, diagnostics = model.forward_adaptive(
        batch["input_ids"],
        batch["attn_mask"],
        batch["slot_pos"],
        threshold=1e9,
        patience=1,
        min_confidence=0.99,
        update_threshold=1e9,
    )
    assert not diagnostics["halted"].any()
    assert diagnostics["steps_taken"].tolist() == [4, 4]


def test_padding_before_slots_does_not_change_prediction() -> None:
    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    rows = [dataset[i] for i in range(100)]
    short = min(rows, key=lambda row: row["n_swaps"])
    long = max(rows, key=lambda row: row["n_swaps"])
    single = collate_fn([short])
    mixed = collate_fn([short, long])
    assert mixed["input_ids"].shape[1] > single["input_ids"].shape[1]

    model = StateTrackingModel(small_config("vanilla")).eval()
    with torch.inference_mode():
        single_logits = model(single["input_ids"], single["attn_mask"], single["slot_pos"])
        mixed_logits = model(mixed["input_ids"], mixed["attn_mask"], mixed["slot_pos"])
    torch.testing.assert_close(single_logits[0], mixed_logits[0], atol=1e-5, rtol=1e-5)


def test_sinusoidal_positions_support_ood_length() -> None:
    model = StateTrackingModel(small_config("recurrent", recurrent_steps=2)).eval()
    input_ids = torch.zeros((1, 600), dtype=torch.long)
    attn_mask = torch.ones_like(input_ids)
    slot_pos = torch.tensor([[595, 596, 597, 598, 599]])
    with torch.inference_mode():
        logits = model(input_ids, attn_mask, slot_pos)
    assert logits.shape == (1, 5, 5)
    assert torch.isfinite(logits).all()


def test_model_can_overfit_a_tiny_batch() -> None:
    torch.manual_seed(0)
    dataset = BallSwapDataset(ROOT / "data" / "train.jsonl")
    batch = collate_fn([dataset[i] for i in range(8)])
    model = StateTrackingModel(
        small_config("vanilla", d_model=64, n_heads=4, dim_feedforward=128, num_layers=2)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(200):
        logits = model(batch["input_ids"], batch["attn_mask"], batch["slot_pos"])
        loss = F.cross_entropy(logits.reshape(-1, 5), batch["labels"].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        predictions = model(
            batch["input_ids"], batch["attn_mask"], batch["slot_pos"]
        ).argmax(dim=-1)
    exact_match = (predictions == batch["labels"]).all(dim=1).float().mean()
    assert float(loss.item()) < 0.05
    assert float(exact_match.item()) == 1.0
