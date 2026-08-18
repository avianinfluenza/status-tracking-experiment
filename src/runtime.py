"""Runtime-only acceleration helpers.

Checkpoint payloads should describe the eager model.  Compilation is an
execution optimization applied after the model is constructed or restored.
"""

from __future__ import annotations

import torch
from torch import nn


def uncompiled_model(model: nn.Module) -> nn.Module:
    """Return the eager module even if a caller passed a compiled wrapper."""

    return getattr(model, "_orig_mod", model)


def uncompiled_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict with eager-model parameter names, never ``_orig_mod.*`` keys."""

    return uncompiled_model(model).state_dict()


def maybe_compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Compile the forward path on CUDA while preserving checkpoint names."""

    if device.type != "cuda":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("CUDA execution requires a PyTorch build with torch.compile")
    model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
    return model
