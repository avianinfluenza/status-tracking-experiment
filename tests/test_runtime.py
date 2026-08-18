from __future__ import annotations

import torch
from torch import nn

from src.runtime import maybe_compile_model, uncompiled_state_dict


def test_compile_is_forward_only_and_keeps_state_dict_names(
    monkeypatch,
) -> None:
    model = nn.Linear(2, 3)
    original_forward = model.forward
    calls: list[object] = []

    def fake_compile(function):
        calls.append(function)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert maybe_compile_model(model, torch.device("cuda")) is model
    assert calls == [original_forward]
    assert set(uncompiled_state_dict(model)) == {"weight", "bias"}
    assert all(not key.startswith("_orig_mod.") for key in uncompiled_state_dict(model))


def test_uncompiled_state_dict_unwraps_compiled_module_shape() -> None:
    class WrappedModule(nn.Module):
        def __init__(self, original: nn.Module) -> None:
            super().__init__()
            self._orig_mod = original

    wrapped = WrappedModule(nn.Linear(2, 3))
    assert set(uncompiled_state_dict(wrapped)) == {"weight", "bias"}
    assert all(not key.startswith("_orig_mod.") for key in uncompiled_state_dict(wrapped))
