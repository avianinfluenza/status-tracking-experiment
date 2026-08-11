"""Shared MLP classification head for slot-based state prediction.

This module keeps the constructor used by ``feature/classifier`` while making
the input contract explicit: callers pass only gathered slot states with shape
``[batch, n_slots, d_model]``.  Slot selection belongs to the model wrapper,
because ``n_slots`` and ``n_class`` are separate concepts even though both are
five in the current dataset.
"""

from __future__ import annotations

from torch import Tensor, nn


class Classifier(nn.Module):
    """Apply one shared MLP to every slot state."""

    def __init__(self, d_model: int, classifier_dim: int, n_class: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if classifier_dim < 1:
            raise ValueError("classifier_dim must be at least 1")
        if n_class < 2:
            raise ValueError("n_class must be at least 2")

        hidden_dim = classifier_dim * d_model
        self.d_model = d_model
        self.classifier_dim = classifier_dim
        self.n_class = n_class
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, slot_states: Tensor) -> Tensor:
        if slot_states.ndim != 3 or slot_states.shape[-1] != self.d_model:
            raise ValueError(
                "slot_states must have shape [B, n_slots, d_model] "
                f"with d_model={self.d_model}"
            )
        return self.classifier(slot_states)
