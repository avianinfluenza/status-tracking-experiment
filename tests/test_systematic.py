from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from src.systematic.data import (
    StateTrackingDataset,
    StateTrackingGenerator,
    TokenVocabulary,
    apply_event,
    collate_examples,
)
from src.systematic.analysis import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_sign_flip_pvalue,
    spearman_rho,
)
from src.systematic.experiment import (
    best_loop_by_depth,
    matched_length_grid,
    ood_degradation_slope,
)
from src.systematic.model import (
    ModelConfig,
    StateTrackingTransformer,
    closest_parameter_matched_width,
    count_parameters,
)


def test_generator_separates_target_depth_from_context_length():
    generator = StateTrackingGenerator(seed=7)
    shallow = generator.generate(target_depth=2, num_distractors=10, num_entities=5)
    deep = generator.generate(target_depth=10, num_distractors=2, num_entities=5)

    assert shallow.total_events == deep.total_events == 12
    assert len(shallow.target_trajectory) == 3
    assert len(deep.target_trajectory) == 11
    assert sum(event.entity == shallow.target for event in shallow.events) == 2
    assert sum(event.entity != shallow.target for event in shallow.events) == 10


def test_symbolic_replay_always_produces_gold_answer():
    generator = StateTrackingGenerator(seed=11)
    example = generator.generate(target_depth=7, num_distractors=13, num_entities=6)
    state = [None] * example.num_entities
    # Every entity's first source is its initial state; replay each entity chain.
    for event in example.events:
        if state[event.entity] is None:
            state[event.entity] = event.src
        state = apply_event(state, event)
    assert state[example.target] == example.answer


def test_schema_vocab_covers_held_out_templates_without_unk():
    generator = StateTrackingGenerator(seed=13)
    example = generator.generate(
        target_depth=3, num_distractors=3, num_entities=5,
        linguistic_variation=True, template_split="ood",
    )
    vocab = TokenVocabulary.from_schema(num_locations=8)
    assert vocab.stoi[vocab.UNK] not in vocab.encode(example.text)


def test_generator_split_dedup_and_metadata():
    seen = set()
    first = StateTrackingGenerator(seed=20).generate_unique(
        20, seen=seen, target_depth=3, num_distractors=4, num_entities=5,
    )
    second = StateTrackingGenerator(seed=21).generate_unique(
        20, seen=seen, target_depth=3, num_distractors=4, num_entities=5,
    )
    assert len({example.example_id for example in [*first, *second]}) == 40
    row = first[0].to_dict()
    assert row["generator_version"] == "systematic-v2"
    assert len(row["initial_state"]) == 5
    assert row["template_ids"]


def tiny_batch():
    generator = StateTrackingGenerator(seed=3)
    examples = [
        generator.generate(target_depth=depth, num_distractors=2, num_entities=4)
        for depth in (1, 2, 3)
    ]
    vocab = TokenVocabulary.from_examples(examples)
    dataset = StateTrackingDataset(examples, vocab)
    return next(iter(DataLoader(
        dataset, batch_size=3,
        collate_fn=partial(collate_examples, pad_id=vocab.pad_id),
    ))), vocab


def test_r0_allows_more_inference_loops_and_returns_each_state():
    batch, vocab = tiny_batch()
    model = StateTrackingTransformer(ModelConfig(
        vocab_size=len(vocab), d_model=16, n_heads=4, d_ff=32,
        train_loops=2, architecture="recurrent",
        loop_conditioning="none", residual_scale=1.0,
    ))
    logits, states = model(
        batch["input_ids"], batch["attention_mask"],
        num_loops=5, return_hidden_states=True,
    )
    assert logits.shape == (3, 8)
    assert len(states) == 5
    assert model.loop_embedding is None
    assert sum(1 for name, _ in model.named_modules() if name == "shared_block") == 1


def test_learned_loop_identity_is_an_explicit_ablation():
    batch, vocab = tiny_batch()
    model = StateTrackingTransformer(ModelConfig(
        vocab_size=len(vocab), d_model=16, n_heads=4, d_ff=32,
        train_loops=2, max_loop_embeddings=4, architecture="recurrent",
        loop_conditioning="learned",
    ))
    assert model.loop_embedding is not None
    model(batch["input_ids"], batch["attention_mask"], num_loops=4)
    with pytest.raises(ValueError):
        model(batch["input_ids"], batch["attention_mask"], num_loops=5)


def test_adaptive_halting_is_optional_and_per_sample():
    batch, vocab = tiny_batch()
    model = StateTrackingTransformer(ModelConfig(
        vocab_size=len(vocab), d_model=16, n_heads=4, d_ff=32,
        train_loops=2, architecture="recurrent",
    )).eval()
    logits, diagnostics = model.forward_adaptive(
        batch["input_ids"], batch["attention_mask"],
        max_loops=6, min_loops=2, patience=1,
        kl_threshold=1e9, update_threshold=1e9, min_confidence=0.0,
    )
    assert logits.shape == (3, 8)
    assert diagnostics["halted"].all()
    assert diagnostics["steps_taken"].tolist() == [2, 2, 2]


def test_standard_and_recurrent_compute_match_but_not_parameter_match():
    _, vocab = tiny_batch()
    common = dict(vocab_size=len(vocab), d_model=16, n_heads=4, d_ff=32)
    standard = StateTrackingTransformer(ModelConfig(
        **common, architecture="standard", num_layers=4,
    ))
    recurrent = StateTrackingTransformer(ModelConfig(
        **common, architecture="recurrent", train_loops=4,
    ))
    assert count_parameters(recurrent) < count_parameters(standard)
    width, matched_count = closest_parameter_matched_width(
        recurrent.config, count_parameters(standard), max_d_model=64
    )
    assert width % recurrent.config.n_heads == 0
    assert matched_count > count_parameters(recurrent)


def test_multi_block_recurrence_cycles_shared_operators():
    batch, vocab = tiny_batch()
    model = StateTrackingTransformer(ModelConfig(
        vocab_size=len(vocab), d_model=16, n_heads=4, d_ff=32,
        train_loops=6, architecture="recurrent", recurrent_blocks=2,
    ))
    _, states = model(
        batch["input_ids"], batch["attention_mask"],
        num_loops=6, return_hidden_states=True,
    )
    assert len(model.recurrent_blocks) == 2
    assert len(states) == 6


def test_matched_length_and_best_loop_helpers():
    cells = matched_length_grid(24, [2, 8, 20])
    assert all(cell["target_depth"] + cell["num_distractors"] == 24 for cell in cells)
    rows = [
        {"target_depth": 2, "num_loops": 2, "accuracy": 0.9},
        {"target_depth": 2, "num_loops": 4, "accuracy": 0.9},
        {"target_depth": 8, "num_loops": 2, "accuracy": 0.5},
        {"target_depth": 8, "num_loops": 4, "accuracy": 0.8},
    ]
    assert best_loop_by_depth(rows) == {2: 2, 8: 4}
    assert ood_degradation_slope({8: 0.9, 10: 0.8, 12: 0.6}, 8) == pytest.approx(-0.1)


def test_statistical_helpers_are_deterministic_and_adjust_multiple_tests():
    assert spearman_rho([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    low, high = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=500, seed=4)
    assert low <= 0.2 <= high
    assert paired_sign_flip_pvalue([0.1, 0.2, 0.3], samples=500, seed=4) <= 1.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.04)
